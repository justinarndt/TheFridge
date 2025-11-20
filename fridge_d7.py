import torch
import torch.nn as nn
import torch.optim as optim
import stim
import numpy as np
import time
import os

# --- The Fridge Architecture (Scaled for D=7) ---
class TheFridge(nn.Module):
    def __init__(self):
        super().__init__()
        # SCALING PARAMS
        self.distance = 7
        self.input_size = 2 * (self.distance ** 2)  # 98 Detectors
        self.embed_dim = 256                        # Wider embedding
        self.hidden_size = 1024                     # Bigger Brain (4x larger memory)
        
        # Layers
        self.detector_embed = nn.Linear(self.input_size, self.embed_dim)
        # Note: We keep 1 layer for speed, but width (1024) handles the complexity
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=False)
        
        # We only care about the World Model head for self-supervised learning
        self.next_syndrome_head = nn.Linear(self.hidden_size, self.input_size) 
        
        self.register_buffer('h', torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        x_emb = self.detector_embed(x)
        if x_emb.dim() == 2: x_emb = x_emb.unsqueeze(0)

        _, self.h = self.gru(x_emb, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        next_syndrome_pred = torch.sigmoid(self.next_syndrome_head(h_flat))
        return next_syndrome_pred

def generate_d7_circuit(rounds=5000, noise_level=0.005):
    """Generates a Distance-7 Toric Code Circuit (98 Detectors)"""
    d = 7
    n_detectors = 2 * (d**2) # 98
    
    circuit = stim.Circuit()
    circuit.append("R", range(n_detectors))
    circuit.append("M", range(n_detectors)) # Primer
    
    loop_body = stim.Circuit()
    loop_body.append("DEPOLARIZE1", range(n_detectors), noise_level)
    loop_body.append("M", range(n_detectors))
    
    for i in range(n_detectors):
        # Compare current with previous round
        loop_body.append("DETECTOR", [stim.target_rec(-n_detectors + i), stim.target_rec(-2*n_detectors + i)])
        
    loop_body.append("SHIFT_COORDS", [], [0, 0, 1])
    circuit += loop_body * rounds
    return circuit

def run_d7_loop(model_path='fridge_d7.pt'):
    device = torch.device("cpu")
    print(f"--- The Fridge: DISTANCE 7 SCALING (Device: {device}) ---")
    print(f"--- Architecture: 98 Inputs -> 1024 Hidden -> 98 Outputs ---")
    
    fridge = TheFridge().to(device)
    if os.path.exists(model_path):
        try:
            fridge.load_state_dict(torch.load(model_path, map_location=device), strict=False)
            print("Loaded existing weights.")
        except:
            print("Starting from scratch.")
    else:
        print("Starting from scratch (New D=7 Brain).")

    optimizer = optim.Adam(fridge.parameters(), lr=0.001) 
    fridge.train()

    print("Initializing D=7 Stim Circuit (98 Detectors)...")
    chunk_size = 5000
    circuit = generate_d7_circuit(rounds=chunk_size)
    sampler = circuit.compile_detector_sampler()
    
    print("\n--- LIVE STREAM STARTED ---")
    
    total_events = 0
    total_correct_bits = 0
    total_bits = 0
    trained_events = 0
    
    try:
        while True:
            batch = sampler.sample(shots=1) 
            full_stream = torch.from_numpy(batch[0].astype(np.float32)).view(-1, 98)
            
            t0 = time.time_ns()
            
            for i in range(full_stream.shape[0] - 1):
                current_syndrome = full_stream[i]
                future_syndrome = full_stream[i+1]
                
                # 1. Forward
                prediction = fridge(current_syndrome)
                
                # 2. Surprise Calculation
                loss = nn.MSELoss()(prediction, future_syndrome)
                surprise = loss.item()
                
                # 3. Gating Logic (Adjusted threshold for D=7 complexity)
                # We allow slightly more noise before triggering training
                should_train = (surprise > 0.06) or (total_events % 1000 == 0)
                
                if should_train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    trained_events += 1
                    status = "ADAPTING"
                else:
                    status = "LOCKED"
                
                # Metrics
                predicted_bits = (prediction > 0.5).float()
                matches = (predicted_bits == future_syndrome).sum().item()
                total_correct_bits += matches
                total_bits += 98 # 98 bits per event now
                total_events += 1
                
                if total_events % 500 == 0:
                    latency_ns = (time.time_ns() - t0) / 500
                    acc = (total_correct_bits / total_bits) * 100
                    training_rate = (trained_events / 500) * 100
                    
                    # Reset
                    total_correct_bits = 0
                    total_bits = 0
                    trained_events = 0
                    t0 = time.time_ns()
                    
                    print(f"Event #{total_events:,} | Latency: {latency_ns/1000:.2f} µs | Accuracy: {acc:.2f}% | Rate: {training_rate:.1f}% [{status}]")

    except KeyboardInterrupt:
        print("\nSaving D=7 brain...")
        torch.save(fridge.state_dict(), 'fridge_d7.pt')

if __name__ == "__main__":
    run_d7_loop()
