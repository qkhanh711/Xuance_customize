from gymnasium import spaces
import numpy as np
from argparse import Namespace
from typing import Tuple, Dict, Any, Optional
from xuance.environment import RawEnvironment

class UAVLLMOffloadingEnv(RawEnvironment):

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        if isinstance(config, Namespace):
            self.config = vars(config)
        elif isinstance(config, dict):
            self.config = config
        else:
            raise TypeError(f"Unsupported config type: {type(config)}")

        cfg_get = self.config.get
        self.render_mode = cfg_get('render_mode', 'human')
        self.seed_value = int(cfg_get('env_seed', cfg_get('seed', 1)))
        self.rng = np.random.default_rng(self.seed_value)

        # ==================== CONFIGURATION ====================
        self.N = cfg_get('num_users', 5)  # Number of users
        self.T = cfg_get('num_timeslots', 10)  # Episode length
        self.tau = cfg_get('timeslot_duration', 3.0)  # seconds

        # Environment bounds
        self.x_min, self.x_max = -100, 100
        self.y_min, self.y_max = -100, 100
        self.z_min, self.z_max = 10, 50

        # UAV parameters
        self.V_max = 50.0  # m/s
        self.E_bg = 1e6  # J (battery capacity)
        self.E_min = 1e4  # J (minimum energy)

        # Flying energy parameters (from Table 1)
        self.P_0 = 158.76  # W (blade profile power)
        self.P_1 = 88.63   # W (induced power)
        self.G_tip = 120.0  # m/s
        self.v_0 = 4.03    # m/s
        self.d_0 = 0.3
        self.epsilon = 1.225  # kg/m³
        self.s = 0.05
        self.A = 0.503  # m²

        # Communication parameters
        self.B_i = 0.512e6  # Hz (bandwidth per user)
        self.B_U_ED = 1e6   # Hz (UAV-server bandwidth)
        self.p_up_i = 10**(17/10) * 1e-3  # W (user uplink power)
        self.p_up_U = 10**(27/10) * 1e-3  # W (UAV uplink power)
        self.p_down_U = 10**(27/10) * 1e-3  # W (UAV downlink power)
        self.p_ED = 10**(40/10) * 1e-3  # W (server power)
        self.N_0 = 10**(-60/10) * 1e-3  # W/Hz (noise PSD)
        self.beta_path = 2.2  # Path loss exponent
        self.phi = 1.8e9  # Hz (carrier frequency)
        self.c = 3e8  # m/s (speed of light)
        self.epsilon_LoS = 0.5

        # LLM parameters
        self.B_batch = cfg_get('llm_batch_size', 8)  # Effective batch size for latency estimate
        self.h = 1024  # Hidden dimension
        self.theta_min = 32  # Min transformer layers
        self.theta_max = 64  # Max transformer layers
        self.f_max = 2.0  # GHz (max GPU frequency)
        self.C_E_ED = 3840  # GPU cores (e.g., V100)
        self.D_E_ED = 1.5  # FLOPs per cycle per core
        self.kappa_ED = 1e-38  # Energy coefficient

        # Utility parameters (from fitted model)
        self.varphi = 0.01
        self.varpi = -1.172
        self.varrho = 47.72

        # Laser charging parameters
        self.P_pb_min = 10.0  # W
        self.P_pb_max = 500.0  # W
        self.zeta_el = 0.35  # Electricity-to-laser efficiency
        self.zeta_le = 0.45  # Laser-to-electricity efficiency
        self.gamma_pb = 0.1  # Cost per W

        # Laser attenuation (clear air, 810nm)
        self.psi = 3.92
        self.xi = 550e-9  # m
        self.omega = 10000  # m (visibility)
        self.rho = 1.3
        self.alpha = (self.psi / self.omega) * (810e-9 / self.xi)**(-self.rho)

        # Laser conversion coefficients (810nm)
        self.b_1 = 0.445
        self.d_1 = -0.75
        self.b_2 = 0.5414
        self.d_2 = -0.2313

        # Edge server and PB positions
        self.server_pos = np.array([0.0, 0.0, 0.0])
        self.pb_pos = np.array([0.0, 50.0, 50.0])

        # Reward weights (normalized)
        self.w_U = cfg_get('w_utility', 0.01)
        self.w_T = cfg_get('w_latency', 1.0)
        self.w_P = cfg_get('w_power', 0.1)
        self.reward_clip = float(cfg_get('reward_clip', 1e4))
        self.reward_scale = float(cfg_get('reward_scale', 1.0))
        self.reward_bias = float(cfg_get('reward_bias', 0.0))
        self.penalty_energy_low_coef = float(cfg_get('penalty_energy_low_coef', 100.0))
        self.penalty_energy_high_coef = float(cfg_get('penalty_energy_high_coef', 50.0))
        self.penalty_latency_coef = float(cfg_get('penalty_latency_coef', 10.0))

        # Observation layout:
        #   8 global features + (3 user_pos + task + token + prev_latency + prev_utility) * N
        # => 8 + 7N
        state_dim = 8 + 7 * self.N
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Action space: continuous [-1, 1]
        # [delta_x, delta_y, delta_z, velocity, P_pb, theta_1, ..., theta_N]
        action_dim = 5 + self.N
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32
        )

        self.current_step = 0
        self.max_episode_steps = self.T
        self.uav_pos = None
        self.uav_energy = None
        self.user_positions = None
        self.task_sizes = None
        self.token_lengths = None
        self.prev_latencies = None
        self.prev_utilities = None

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)

        self.uav_pos = np.array([0.0, 0.0, 55.0])
        self.uav_energy = self.E_bg
        self.current_step = 0

        # Random user positions
        self.user_positions = self.rng.uniform(
            low=[self.x_min, self.y_min, 0],
            high=[self.x_max, self.y_max, 0],
            size=(self.N, 3)
        )

        # Random task arrivals
        self.task_sizes = self.rng.uniform(512, 1024, size=self.N)  # bits
        self.token_lengths = self.rng.uniform(512, 1024, size=self.N)

        # Initialize history
        self.prev_latencies = np.zeros(self.N)
        self.prev_utilities = np.zeros(self.N)

        obs = self._get_observation()
        info = {'episode': 0, 'step': 0}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self.action_space.shape[0]:
            raise ValueError(f"Expected action dim {self.action_space.shape[0]}, got {action.shape[0]}")

        uav_pos_new, velocity, P_pb, thetas = self._convert_action(action)

        T_fly = np.linalg.norm(uav_pos_new - self.uav_pos) / (velocity + 1e-6)
        E_fly = self._compute_flying_energy(velocity, T_fly)
        E_hover = self._compute_hovering_energy(max(self.tau - T_fly, 0.0))
        E_tot = E_fly + E_hover

        channel_gains_users = self._compute_channel_gains(uav_pos_new, self.user_positions)
        channel_gain_server = self._compute_channel_gain_server(uav_pos_new)

        R_up_users = self._compute_uplink_rates(channel_gains_users)
        R_down_users = self._compute_downlink_rates(channel_gains_users)
        R_up_server = self._compute_uplink_rate_server(channel_gain_server)
        R_down_server = self._compute_downlink_rate_server(channel_gain_server)

        latencies = np.zeros(self.N)
        utilities = np.zeros(self.N)
        energies_relay = np.zeros(self.N)

        for i in range(self.N):
            # Upload latency
            T_up = self.task_sizes[i] * (1/R_up_users[i] + 1/R_up_server)

            psi = self._compute_flops(self.token_lengths[i])
            f_i = self.f_max / self.N
            T_cmp_layer = psi / (f_i * self.C_E_ED * self.D_E_ED * 1e9)  # 1e9 for GHz
            T_inf = thetas[i] * T_cmp_layer

            s_out = self.task_sizes[i]
            T_down = s_out * (1/R_down_server + 1/R_down_users[i])

            latencies[i] = T_up + T_inf + T_down

            utilities[i] = (self.varphi * thetas[i]**2 +
                          self.varpi * thetas[i] +
                          self.varrho)

            energies_relay[i] = (self.task_sizes[i] * self.p_up_U / R_up_server +
                                s_out * self.p_down_U / R_down_users[i])

        d_pb = np.linalg.norm(uav_pos_new - self.pb_pos)
        zeta_ls = np.exp(-self.alpha * d_pb)

        if P_pb >= self.P_pb_min:
            P_rv = (self.b_1 * self.b_2 * zeta_ls * P_pb +
                   self.b_2 * self.d_1 * zeta_ls + self.d_2)
            P_rv = P_rv * self.zeta_el * self.zeta_le
        else:
            P_rv = 0.0

        E_rv = P_rv * self.tau
        power_cost = self.gamma_pb * P_pb

        E_relay_total = float(np.sum(energies_relay))
        self.uav_energy = self.uav_energy + E_rv - E_relay_total - E_tot
        self.uav_pos = uav_pos_new
        self.prev_latencies = latencies
        self.prev_utilities = utilities
        self.current_step += 1

        reward, info = self._compute_reward(
            utilities, latencies, power_cost,
            T_fly, thetas, E_rv, E_relay_total, E_tot
        )

        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        terminated = (self.current_step >= self.max_episode_steps) or (self.uav_energy <= 0)
        truncated = False

        if not terminated:
            self.task_sizes = self.rng.uniform(512, 1024, size=self.N)
            self.token_lengths = self.rng.uniform(512, 1024, size=self.N)

        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"UAV Position: {self.uav_pos}")
        print(f"UAV Energy: {self.uav_energy:.2f} J")
        print(f"User Positions: {self.user_positions}")
        print(f"Task Sizes: {self.task_sizes}")
        print(f"Token Lengths: {self.token_lengths}")
        print(f"Previous Latencies: {self.prev_latencies}")
        print(f"Previous Utilities: {self.prev_utilities}")
        
    def close(self):
        return None

    def _convert_action(self, action: np.ndarray) -> Tuple:
        action = np.clip(action, -1.0, 1.0)
        velocity = self.V_max * (1 + action[3]) / 2
        velocity = np.clip(velocity, 0, self.V_max)

        max_delta = velocity * self.tau
        delta_pos = action[:3] * max_delta
        new_pos = self.uav_pos + delta_pos
        new_pos = np.clip(new_pos,
                         [self.x_min, self.y_min, self.z_min],
                         [self.x_max, self.y_max, self.z_max])


        P_pb = self.P_pb_max * (1 + action[4]) / 2
        P_pb = max(0, P_pb)

        thetas = self.theta_min + (self.theta_max - self.theta_min) * (1 + action[5:]) / 2
        thetas = np.clip(np.round(thetas), self.theta_min, self.theta_max).astype(int)

        return new_pos, velocity, P_pb, thetas

    def _compute_flying_energy(self, v, T_fly):
        """Compute UAV flying energy (Eq. 15)"""
        P_blade = self.P_0 * (1 + 3*v**2 / self.G_tip**2)
        P_induced = self.P_1 * (np.sqrt(1 + v**4/(4*self.v_0**4)) - v**2/(2*self.v_0**2))**0.5
        P_parasite = 0.5 * self.d_0 * self.s * self.epsilon * self.A * v**3
        P_fly = P_blade + P_induced + P_parasite
        return P_fly * T_fly

    def _compute_hovering_energy(self, T_hover):
        """Hovering energy"""
        P_hover = self.P_0 + self.P_1
        return P_hover * T_hover

    def _compute_channel_gains(self, uav_pos, user_positions):
        """Channel gains UAV-users (Eq. 2a)"""
        distances = np.maximum(np.linalg.norm(user_positions - uav_pos, axis=1), 1.0)
        path_loss = (4 * np.pi * self.phi / self.c)**(-2)
        gains = path_loss * self.epsilon_LoS * (distances**2)**(-self.beta_path/2)
        gains *= 1.5
        return np.maximum(gains, 1e-12)

    def _compute_channel_gain_server(self, uav_pos):
        """Channel gain UAV-server"""
        distance = max(np.linalg.norm(uav_pos - self.server_pos), 1.0)
        path_loss = (4 * np.pi * self.phi / self.c)**(-2)
        gain = path_loss * self.epsilon_LoS * (distance**2)**(-self.beta_path/2)
        gain *= 1.5
        return max(gain, 1e-12)

    def _compute_uplink_rates(self, gains):
        """Uplink rates users→UAV (Eq. 4a)"""
        SNR = self.p_up_i * gains / (self.B_i * self.N_0)
        rates = self.B_i * np.log2(1 + SNR)
        return np.maximum(rates, 1e3)

    def _compute_downlink_rates(self, gains):
        """Downlink rates UAV→users (Eq. 4b)"""
        SNR = self.p_down_U * gains / (self.B_i * self.N_0)
        rates = self.B_i * np.log2(1 + SNR)
        return np.maximum(rates, 1e3)

    def _compute_uplink_rate_server(self, gain):
        """Uplink rate UAV→server"""
        SNR = self.p_up_U * gain / (self.B_U_ED * self.N_0)
        return max(self.B_U_ED * np.log2(1 + SNR), 1e3)

    def _compute_downlink_rate_server(self, gain):
        """Downlink rate server→UAV"""
        SNR = self.p_ED * gain / (self.B_U_ED * self.N_0)
        return max(self.B_U_ED * np.log2(1 + SNR), 1e3)

    def _compute_flops(self, d):
        """FLOPs per token (before Eq. 6)"""
        return 24 * self.B_batch * d * self.h**2 + 4 * self.B_batch * d**2 * self.h

    def _compute_reward(self, utilities, latencies, power_cost,
                       T_fly, thetas, E_rv, E_relay, E_tot):
        """Compute reward with penalties"""
        # Utility is treated as positive benefit; latency/power are costs.
        utility_term = self.w_U * float(np.mean(utilities))
        latency_term = self.w_T * float(np.mean(latencies))
        power_term = self.w_P * float(power_cost)
        reward = utility_term - latency_term - power_term

        penalties = 0
        info = {'violations': []}

        if self.uav_energy < self.E_min:
            penalty = self.penalty_energy_low_coef * (self.E_min - self.uav_energy) / self.E_bg
            penalty = float(min(penalty, self.reward_clip))
            penalties -= penalty
            info['violations'].append(f'energy_low:{penalty:.2f}')
        elif self.uav_energy > self.E_bg:
            penalty = self.penalty_energy_high_coef * (self.uav_energy - self.E_bg) / self.E_bg
            penalty = float(min(penalty, self.reward_clip))
            penalties -= penalty
            info['violations'].append(f'energy_high:{penalty:.2f}')

        latency_budget = max(self.tau, 1e-3)
        for i, lat in enumerate(latencies):
            if lat > latency_budget:
                # Ratio-based penalty is more stable than raw-second penalty when latency scale is large.
                penalty = self.penalty_latency_coef * ((lat / latency_budget) - 1.0)
                penalty = float(min(penalty, self.reward_clip / max(self.N, 1)))
                penalties -= penalty
                info['violations'].append(f'latency_user{i}:{penalty:.2f}')

        if self.E_min < self.uav_energy < 0.8 * self.E_bg:
            reward += 10

        if len(latencies) > 1:
            latency_std = np.std(latencies)
            reward -= 0.1 * latency_std

        reward_total = (reward + penalties) * self.reward_scale + self.reward_bias

        info.update({
            'total_utility': np.sum(utilities),
            'total_latency': np.sum(latencies),
            'power_cost': power_cost,
            'energy_remaining': self.uav_energy,
            'energy_received': E_rv,
            'avg_layers': np.mean(thetas),
            'reward_utility_term': utility_term,
            'reward_latency_term': latency_term,
            'reward_power_term': power_term,
            'reward_penalty_term': penalties,
            'reward_before_scale': reward + penalties,
            'reward_after_scale': reward_total,
        })

        return reward_total, info

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector"""
        uav_pos_norm = self.uav_pos.copy()
        uav_pos_norm[:2] /= 100.0
        uav_pos_norm[2] = (uav_pos_norm[2] - 55) / 45

        energy_norm = (self.uav_energy - self.E_min) / (self.E_bg - self.E_min)

        distance_pb = np.linalg.norm(self.uav_pos - self.pb_pos) / 150.0  # Normalize by max distance

        time_remaining = (self.T - self.current_step) / self.T

        user_pos_norm = self.user_positions.copy()
        user_pos_norm[:, :2] /= 100.0

        task_sizes_norm = np.log1p(self.task_sizes) / 10.0
        token_lengths_norm = self.token_lengths / 1024.0

        latencies_norm = self.prev_latencies / self.tau
        utilities_norm = self.prev_utilities / 50.0
        obs = np.concatenate([
            uav_pos_norm,
            [energy_norm],
            [distance_pb],
            [time_remaining],
            [0, 0],
            user_pos_norm.flatten(),
            task_sizes_norm,
            token_lengths_norm,
            latencies_norm,
            utilities_norm,
        ])

        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0)
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs.astype(np.float32)