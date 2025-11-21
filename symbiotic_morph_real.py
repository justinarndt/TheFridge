import torch
import torch.nn as nn
import torch.optim as optim
import stim
import pymatching
import numpy as np
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- THE BRAIN (Metamorphic Fridge) ---
class MetamorphicFridge(nn.Module):
    def __init__(self):
        super().__init__()
        # D=7 rounds=7 produces 336 detectors. 
        self.input_size = 336 
        self.hidden_size = 512
        
        self.detector_embed = nn.Linear(self.input_size, 256)
        self.gru = nn.GRU(input_size=256, hidden_size=self.hidden_size, num_layers=2, batch_first=False)
        
        self.storm_head = nn.Linear(self.hidden_size, 1)      
        self.topology_head = nn.Linear(self.hidden_size, 2)   
        
        self.register_buffer('h', torch.zeros(2, 1, self.hidden_size))

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        
        # Pad input for D=3 cases
        if x.shape[-1] < self.input_size:
            padding = torch.zeros(x.shape[0], x.shape[1], self.input_size - x.shape[-1], device=x.device)
            x = torch.cat([x, padding], dim=-1)
            
        x_emb = self.detector_embed(x)
        _, self.h = self.gru(x_emb, self.h.detach())
        
        # FIX: Do NOT squeeze the batch dimension. 
        # We need shape (1, 512) for the heads to output (1, N) for CrossEntropyLoss.
        h_flat = self.h[-1] 
        
        storm = torch.sigmoid(self.storm_head(h_flat))
        topology_logits = self.topology_head(h_flat)
        return storm, topology_logits

# --- THE PHYSICS ENGINE (With PyMatching) ---
class PhysicsEngine:
    def __init__(self):
        print("Initializing Physics Engine & Decoders...")
        # Pre-compile decoders at a reference noise level (0.005)
        # This simulates a calibrated decoder.
        self.circuit_d3_ref = self._make_circuit(3, 0.005)
        self.circuit_d7_ref = self._make_circuit(7, 0.005)
        
        self.matcher_d3 = pymatching.Matching.from_detector_error_model(self.circuit_d3_ref.detector_error_model(decompose_errors=True))
        self.matcher_d7 = pymatching.Matching.from_detector_error_model(self.circuit_d7_ref.detector_error_model(decompose_errors=True))
        print("Decoders Online.")

    def _make_circuit(self, distance, p):
        return stim.Circuit.generated(
            "surface_code:rotated_memory_z",
            distance=distance,
            rounds=distance,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p
        )

    def run_shot(self, distance, p_noise):
        # 1. Build Circuit with current weather
        circuit = self._make_circuit(distance, p_noise)
        
        # 2. Run Physics
        sampler = circuit.compile_detector_sampler()
        dets, obs_flips = sampler.sample(1, separate_observables=True)
        
        # 3. Run Decoder (The Correction)
        matcher = self.matcher_d3 if distance == 3 else self.matcher_d7
        predicted_flip = matcher.decode(dets[0])[0]
        
        # 4. Verdict: Did the decoder match the actual error?
        # If predicted_flip != actual_flip, we have a logical error.
        actual_flip = obs_flips[0][0]
        logical_error = 1 if predicted_flip != actual_flip else 0
        
        return torch.tensor(dets[0], dtype=torch.float32), logical_error

# --- THE LOOP ---
def run_real_metamorphosis():
    print(f"--- LEVEL 2: CODE METAMORPHOSIS (DECODED) ---")
    physics = PhysicsEngine()
    fridge = MetamorphicFridge().to(DEVICE)
    optimizer = optim.Adam(fridge.parameters(), lr=0.002)
    
    history = {'noise': [], 'topology': [], 'static_err': 0, 'dynamic_err': 0}
    current_distance = 3
    
    print("Streaming 3,000 Cycles...") # Shortened for speed/impact
    
    for t in range(3000):
        # 1. Weather Pattern (Peaks at 0.009 - Sub-threshold)
        phase = t / 300.0
        p_noise = 0.001 + 0.008 * (np.sin(phase)**2)
        
        # 2. Run Parallel Universes
        # Universe A: Static D=3
        dets_d3, err_d3 = physics.run_shot(3, p_noise)
        history['static_err'] += err_d3
        
        # Universe B: Dynamic Metamorphosis
        if current_distance == 3:
            active_dets = dets_d3
            active_err = err_d3
        else:
            dets_d7, err_d7 = physics.run_shot(7, p_noise)
            active_dets = dets_d7
            active_err = err_d7
            
        history['dynamic_err'] += active_err
        
        # 3. Fridge Learning
        optimizer.zero_grad()
        storm_pred, top_logits = fridge(active_dets.to(DEVICE))
        
        # Training Signals
        density = active_dets.sum() / len(active_dets)
        is_storm = density > 0.08
        
        # Policy: Switch to D=7 if noise > 0.004
        ideal_top = 1 if p_noise > 0.004 else 0
        
        loss = nn.MSELoss()(storm_pred, torch.tensor([[float(is_storm)]], device=DEVICE)) + \
               nn.CrossEntropyLoss()(top_logits, torch.tensor([ideal_top], device=DEVICE))
        
        loss.backward()
        optimizer.step()
        
        # 4. Act
        choice = torch.argmax(top_logits).item()
        current_distance = 7 if choice == 1 else 3
        
        history['noise'].append(p_noise)
        history['topology'].append(current_distance)
        
        if t % 500 == 0:
            mode = "D=3" if current_distance == 3 else "D=7"
            print(f"Cycle {t} | p={p_noise:.4f} | Mode: {mode} | Dynamic Err: {history['dynamic_err']} vs Static: {history['static_err']}")

    # FINAL REPORT
    print("\n--- DEBRIEF ---")
    print(f"Static D=3 Errors:  {history['static_err']}")
    print(f"Dynamic Errors:     {history['dynamic_err']}")
    improvement = history['static_err'] / max(history['dynamic_err'], 1)
    print(f"IMPROVEMENT FACTOR: {improvement:.2f}x")
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Noise (p)', color='red')
    ax1.plot(history['noise'], color='red', alpha=0.5, label='Noise')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Distance', color='blue')
    ax2.plot(history['topology'], color='blue', label='Topology')
    
    plt.title(f"Metamorphosis Verification: {improvement:.2f}x Improvement")
    plt.savefig('metamorphosis_real_proof.png')
    print("Proof saved.")

if __name__ == "__main__":
    run_real_metamorphosis()
