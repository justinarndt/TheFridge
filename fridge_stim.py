import torch
import torch.nn as nn
import stim
import numpy as np
import time
import os

# --- The Fridge Architecture (Matched to Checkpoint) ---
class TheFridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 18
        self.embed_dim = 128
        self.hidden_size = 256
        self.output_size = 76
        
        self.detector_embed = nn.Linear(self.input_size, self.embed_dim)
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=False)
        self.correction_head = nn.Linear(self.hidden_size, self.output_size)
        self.next_syndrome_head = nn.Linear(self.hidden_size, 18) 
        
        self.register_buffer('h', torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        x_emb = self.detector_embed(x)
        if x_emb.dim() == 2: x_emb = x_emb.unsqueeze(0)

        _, self.h = self.gru(x_emb, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        next_syndrome_pred = torch.sigmoid(self.next_syndrome_head(h_flat))
        return next_syndrome_pred

def generate_stim_circuit(rounds=10000, noise_level=0.005):
    circuit = stim.Circuit()
    
    # 1. Initialization
    circuit.append("R", range(18))
    
    # 2. PRIMER ROUND (The Fix)
    # We measure once before the loop so the first detector has a "past" to compare to.
    circuit.append("M", range(18)) 
    
    # 3. The Loop
    loop_body = stim.Circuit()
    loop_body.append("DEPOLARIZE1", range(18), noise_level)
    loop_body.append("M", range(18))
    
    for i in range(18):
        # Compare current (rec[-18]) with previous (rec[-36])
        # This is now safe because we added the primer measurements.
        loop_body.append("DETECTOR", [stim.target_rec(-18 + i), stim.target_rec(-36 + i)])
        
    loop_body.append("SHIFT_COORDS", [], [0, 0, 1])
    
    circuit += loop_body * rounds
    return circuit

def run_stim_bridge(model_path='fridge.pt'):
    device = torch.device("cpu")
    print(f"--- Connecting The Fridge to Stim (Device: {device}) ---")
    
    fridge = TheFridge().to(device)
    if not os.path.exists(model_path):
        print(f"Error: {model_path} missing."); return
        
    try:
        fridge.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"Load Error: {e}"); return
    fridge.eval()

    print("Initializing Stim Circuit (D=3 Toric Topology, 18 Detectors)...")
    chunk_size = 1000
    # Generate circuit with valid boundary conditions
    circuit = generate_stim_circuit(rounds=chunk_size)
    sampler = circuit.compile_detector_sampler()
    
    print("\n--- LIVE STREAM STARTED ---")
    print("The model is now trying to predict quantum errors before they happen.\n")
    
    total_events = 0
    total_correct_bits = 0
    total_bits = 0
    
    try:
        while True:
            batch = sampler.sample(shots=1) 
            full_stream = torch.from_numpy(batch[0].astype(np.float32)).view(-1, 18)
            
            t0 = time.time_ns()
            
            for i in range(full_stream.shape[0] - 1):
                current_syndrome = full_stream[i]
                future_syndrome = full_stream[i+1]
                
                with torch.no_grad():
                    prediction = fridge(current_syndrome)
                
                predicted_bits = (prediction > 0.5).float()
                matches = (predicted_bits == future_syndrome).sum().item()
                
                total_correct_bits += matches
                total_bits += 18
                total_events += 1
                
                if total_events % 500 == 0:
                    latency_ns = (time.time_ns() - t0) / 500
                    acc = (total_correct_bits / total_bits) * 100
                    t0 = time.time_ns()
                    
                    mse = torch.mean((prediction - future_syndrome).pow(2)).item()
                    status = "PREDICTING" if acc > 85.0 else "LEARNING"
                    
                    print(f"Event #{total_events:,} | Latency: {latency_ns/1000:.2f} µs | Accuracy: {acc:.2f}% | Surprise: {mse:.4f} [{status}]")

    except KeyboardInterrupt:
        print("\nDisconnected.")

if __name__ == "__main__":
    run_stim_bridge()
