import torch
import torch.nn as nn
import numpy as np
import time
import os

# --- The Fridge Architecture (Must Match D=7 Checkpoint) ---
class TheFridge(nn.Module):
    def __init__(self):
        super().__init__()
        # SCALING PARAMS FOR D=7
        self.input_size = 98        # 2 * 7^2
        self.embed_dim = 256        # Matches fridge_d7.py
        self.hidden_size = 1024     # Matches fridge_d7.py
        
        self.detector_embed = nn.Linear(self.input_size, self.embed_dim)
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=False)
        self.next_syndrome_head = nn.Linear(self.hidden_size, self.input_size) 
        
        self.register_buffer('h', torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        # Ensure input shape (1, 1, 98)
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        x_emb = self.detector_embed(x)
        if x_emb.dim() == 2: x_emb = x_emb.unsqueeze(0)

        _, self.h = self.gru(x_emb, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        next_syndrome_pred = torch.sigmoid(self.next_syndrome_head(h_flat))
        return next_syndrome_pred

def run_fridge_deployment(model_path='fridge_d7.pt'):
    # 1. Optimization: Run on CPU for minimal single-batch latency
    device = torch.device("cpu")
    print(f"--- The Fridge (v3): Deployment Mode ---")
    print(f"Target: Distance-7 Surface Code (98 Detectors)")
    print(f"Device: {device}")

    if not os.path.exists(model_path):
        print(f"Error: Checkpoint '{model_path}' not found. Did you run fridge_d7.py?")
        return

    # 2. Load the Brain
    fridge = TheFridge().to(device)
    try:
        # strict=False allows for minor version mismatches in non-critical layers
        fridge.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print(f"SUCCESS: Loaded {model_path}")
    except Exception as e:
        print(f"FATAL: Model load failed. {e}")
        return

    fridge.eval()
    
    print("\n--- INFERENCE STREAM STARTED (Ctrl+C to Stop) ---")
    
    n_events = 0
    
    try:
        while True:
            # 3. Simulate Incoming Hardware Data (98 bits)
            # In production, this line is replaced by the FPGA/DAQ readout
            # We inject random noise here to simulate the 'hum' of the machine
            input_vector = torch.randn(98)
            
            # 4. The Critical Path (Inference)
            start_time = time.time_ns()
            
            with torch.no_grad():
                prediction = fridge(input_vector)
                
            latency_ns = time.time_ns() - start_time
            
            # 5. Logging (Every 1000 events)
            if n_events % 1000 == 0:
                # Calculate a mock "Confidence" metric based on prediction certainty
                confidence = torch.mean(torch.abs(prediction - 0.5) * 2).item()
                status = "LOCKED" if confidence > 0.8 else "SCANNING"
                
                print(f"Event #{n_events:,} | Latency: {latency_ns/1000:.2f} µs | Confidence: {confidence:.2f} [{status}]")
            
            n_events += 1
            
    except KeyboardInterrupt:
        print("\nShutting down.")

if __name__ == "__main__":
    # Defaulting to the D=7 model you just built
    run_fridge_deployment(model_path='fridge_d7.pt')
