import torch
import torch.nn as nn
import numpy as np
import time
import os
import math

# --- The Fridge Architecture (Must match your converted checkpoint) ---
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
        self.next_syndrome_head = nn.Linear(self.hidden_size, 18) # This IS the trained world model
        
        # We ignore the untrained storm/bias heads for this demo
        self.register_buffer('h', torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0).unsqueeze(0)
        x_emb = self.detector_embed(x)
        if x_emb.dim() == 2: x_emb = x_emb.unsqueeze(0)

        _, self.h = self.gru(x_emb, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        # Predictions
        next_syndrome_pred = torch.sigmoid(self.next_syndrome_head(h_flat))
        return next_syndrome_pred

def run_demo(model_path='fridge.pt'):
    # 1. Force CPU for low-latency single-event inference
    device = torch.device("cpu") 
    print(f"--- Running The Fridge on: {device} (Optimized for Latency) ---")
    
    # Load Model
    fridge = TheFridge().to(device)
    try:
        fridge.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    except Exception as e:
        print(f"Error: {e}"); return
    fridge.eval()

    print("\n--- THE DEMO: Phase-Locking to a Hidden 60Hz Signal ---")
    print("Watch the 'Surprise' metric drop as the GRU learns the pattern.\n")
    
    n_events = 0
    # Simulate a 60Hz drift in a 10kHz sampling loop
    frequency = 60.0 
    dt = 1.0 / 10000.0 
    
    try:
        while True:
            t = n_events * dt
            
            # 2. Generate Input: A "noisy" sine wave signal
            # This represents the bias drifting. The GRU should learn this.
            signal_strength = math.sin(2 * math.pi * frequency * t)
            
            # The "Real" world event (noisy)
            input_vector = torch.randn(18) * 0.5 + (signal_strength * 0.5)
            
            # 3. INFERENCE (The critical path)
            start_time = time.time_ns()
            
            with torch.no_grad():
                prediction = fridge(input_vector)
            
            latency_ns = time.time_ns() - start_time
            
            # 4. Calculate "Surprise" (Prediction Error)
            # If the GRU is working, this error should be lower than random guessing (>0.25)
            # We compare the prediction to the *clean* signal direction to see if it "gets it"
            target_signal = torch.full((18,), (signal_strength * 0.5) + 0.5) # Normalized 0-1 approx
            surprise = torch.mean((prediction - target_signal).pow(2)).item()

            # Logging
            if n_events % 2000 == 0:
                 status = "LOCKED" if surprise < 0.15 else "SEARCHING"
                 print(f"Event #{n_events:,} | Latency: {latency_ns/1000:.2f} µs | Surprise: {surprise:.4f} [{status}]")

            n_events += 1
            
            # Reset hidden state occasionally to prove it re-locks (every 50k events)
            if n_events % 50000 == 0:
                print(">>> BURST EVENT! Resetting Internal State... <<<")
                fridge.h = torch.zeros(1, 1, 256)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    run_demo()
