import numpy as np
from scipy.spatial.distance import euclidean

class WillowNoiseModel:
    """
    Implements the Composite Noise Equation:
    P_err(t) = P_base + N_drift(t) + C_xtalk(q, t) + B_burst(t) + P_periodic(t)
    Ref: Survivor QEC Report, Section 4.1 
    """
    def __init__(self, num_qubits, distance):
        self.num_qubits = num_qubits
        self.distance = distance
        self.t = 0
        
        # Base Parameters
        self.p_base = 0.005
        
        # 1. Pink Noise Generator State (Voss-McCartney algorithm for 1/f) [cite: 159]
        self.pink_noise_state = np.random.randn(num_qubits, 16) 
        self.drift_scale = 0.002
        
        # 2. Crosstalk Matrix (Sycamore-style spatial coupling) [cite: 167]
        self.crosstalk_matrix = self._generate_crosstalk_map()
        
        # 3. Periodic components (Hidden Physics)
        self.phase = 0.0
        self.freq = 0.06  # Normalized frequency (60Hz baseline) [cite: 188]
        self.amplitude = 0.003
        
        # 4. Burst State
        self.is_bursting = False
        self.burst_timer = 0
        self.burst_multiplier = 1.0

    def _generate_crosstalk_map(self):
        # Simplified spatial lattice generation for Distance-d code
        coords = [(i % self.distance, i // self.distance) for i in range(self.num_qubits)]
        mat = np.zeros((self.num_qubits, self.num_qubits))
        for i in range(self.num_qubits):
            for j in range(self.num_qubits):
                if i != j:
                    dist = euclidean(coords[i], coords[j])
                    # Neighbor coupling decays with distance
                    if dist <= 1.5: 
                        mat[i, j] = 0.15  # Strong neighbor crosstalk
        return mat

    def _update_pink_noise(self):
        # Simplified 1/f update
        row = np.random.randint(0, 16)
        self.pink_noise_state[:, row] = np.random.randn(self.num_qubits)
        return np.sum(self.pink_noise_state, axis=1) * 0.1

    def step(self, config_overrides=None):
        """
        Returns the error probability vector for the current timestep.
        """
        self.t += 1
        
        # Apply catastrophic overrides (if any) from the Environment Wrapper
        current_freq = config_overrides.get('freq', self.freq) if config_overrides else self.freq
        drift_mult = config_overrides.get('drift_mult', 1.0) if config_overrides else 1.0
        
        # --- Component 1: 1/f Drift [cite: 155] ---
        drift = self._update_pink_noise() * self.drift_scale * drift_mult
        
        # --- Component 2: Periodic Interference [cite: 31] ---
        self.phase += 2 * np.pi * current_freq
        periodic = self.amplitude * np.sin(self.phase)
        
        # --- Component 3: Burst Logic [cite: 173] ---
        # Check trigger or explicit override
        trigger_burst = config_overrides.get('burst', False) if config_overrides else False
        
        if trigger_burst and not self.is_bursting:
            self.is_bursting = True
            self.burst_timer = 5 # 5 round duration [cite: 177]
            self.burst_multiplier = 5.0 # 5x Amplitude
            
        if self.is_bursting:
            self.burst_timer -= 1
            if self.burst_timer <= 0:
                self.is_bursting = False
                self.burst_multiplier = 1.0
        
        # Summation
        raw_probs = self.p_base + drift + periodic
        
        # Apply Burst Multiplier
        raw_probs *= self.burst_multiplier
        
        # --- Component 4: Crosstalk Mixing [cite: 163] ---
        # P_final = P_raw + (Crosstalk_Matrix * P_raw)
        xtalk_contribution = np.dot(self.crosstalk_matrix, raw_probs) * 0.1
        total_probs = raw_probs + xtalk_contribution
        
        return np.clip(total_probs, 0.0, 0.5) # Clip to physical limits