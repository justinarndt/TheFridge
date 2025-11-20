import torch
import torch.nn as nn
import numpy as np
import stim
import time
import os

# Must match convert_to_fridge.py EXACTLY
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
        self.storm_head = nn.Linear(self.hidden_size, 1)
        self.bias_head  = nn.Linear(self.hidden_size, 18)
        
        self.register_buffer('h', torch.zeros(1, 1, self.hidden_size))

    def forward(self, x):
        # Support both (1,1,18) inputs
        if x.dim() == 1:
             x = x.unsqueeze(0).unsqueeze(0)
        
        x_emb = self.detector_embed(x)
        if x_emb.dim() == 2: # Handle if embed flattens
             x_emb = x_emb.unsqueeze(0)

        _, self.h = self.gru(x_emb, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        correction = self.correction_head(h_flat)
        storm = torch.sigmoid(self.storm_head(h_flat))
        return correction, storm

def run_fridge_simulation(model_path='fridge.pt'):
    if not os.path.exists(model_path):
        print(f"Error: '{model_path}' not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Running The Fridge on: {device} ---")
    
    fridge = TheFridge().to(device)
    try:
        fridge.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    fridge.eval()

    print("\nStarting Continuous-Time Event Stream (Ctrl+C to stop)...")
    
    n_events = 0
    while True:
        # Simulate a "Volume" event (Vector of 18 syndromes)
        # In a real scenario, this would come from Stim
        input_vector = torch.randn(18).to(device) # Random input to keep the GRU humming

        start_time = time.time_ns()
        with torch.no_grad():
            correction, storm = fridge(input_vector)
        latency_ns = time.time_ns() - start_time
        
        if n_events % 1000 == 0:
             print(f"Event #{n_events:,} | Latency: {latency_ns/1000:.2f} µs | Storm: {storm.item():.2f}")

        n_events += 1

if __name__ == "__main__":
    run_fridge_simulation()
