import torch
import torch.nn as nn
import numpy as np
import time
import os
from collections import deque

class SymbioticFridge(nn.Module):
    def __init__(self, distance=7, hidden_size=1024, device='cpu'):
        super().__init__()
        self.d = distance
        self.n_data = distance * distance
        # FIX: Match the checkpoint's topology (2 * d^2 = 98)
        self.n_detectors = 2 * (distance ** 2)  
        self.hidden_size = hidden_size
        self.device = device
        
        # Embedding
        self.detector_embed = nn.Linear(self.n_detectors, 256)
        
        # Core persistent world model (your exact trained GRU)
        self.gru = nn.GRU(input_size=256, hidden_size=hidden_size, num_layers=1, batch_first=False)
        
        # New symbiotic heads
        self.correction_head = nn.Linear(hidden_size, self.n_data * 3)    # X, Y, Z per data qubit
        self.storm_head = nn.Linear(hidden_size, 1)                       # scalar 0-1 burst forecast
        self.bias_head = nn.Linear(hidden_size, self.n_data * 2)          # per-qubit X/Z bias shift
        self.next_syndrome_head = nn.Linear(hidden_size, self.n_detectors)  # keeps physics honest
        
        # Persistent hidden state - NEVER RESET
        self.register_buffer('h', torch.zeros(1, 1, hidden_size))
        
        # Lazy Pauli frame (free corrections)
        self.register_buffer('pauli_frame_x', torch.zeros(self.n_data))
        self.register_buffer('pauli_frame_z', torch.zeros(self.n_data))
        
        # Imagination buffer for multi-step rollout
        self.imag_buffer = deque(maxlen=10)
        
        self.to(device)

    def imagine_forward(self, steps=10):
        """Dreamer-style rollout using only the learned dynamics"""
        imagines = []
        h_imag = self.h.clone().detach()
        for _ in range(steps):
            # Use tiny learned transition (approximated from next syndrome → embed → gru step)
            fake_next = self.next_syndrome_head(h_imag[-1]).detach()
            emb = self.detector_embed(fake_next)
            _, h_imag = self.gru(emb.unsqueeze(0), h_imag)
            storm = torch.sigmoid(self.storm_head(h_imag[-1])).item()
            imagines.append(storm)
        return np.mean(imagines)  # forecasted storm over next ~10 events

    def forward(self, detector_vector: torch.Tensor):
        """
        Input: raw syndrome vector (98,) float32, sparse or dense
        Output: dict with everything the algorithm needs to negotiate
        """
        if detector_vector.dim() == 1:
            detector_vector = detector_vector.unsqueeze(0).unsqueeze(0)  # (1,1,98)
        
        x_emb = self.detector_embed(detector_vector.to(self.device))
        _, self.h = self.gru(x_emb, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        correction_soft = torch.sigmoid(self.correction_head(h_flat))   # (n_data*3,)
        storm_level = torch.sigmoid(self.storm_head(h_flat))           # (1,)
        bias_forecast = torch.tanh(self.bias_head(h_flat)).view(self.n_data, 2)  # (n_data, X/Z)
        next_synd_pred = torch.sigmoid(self.next_syndrome_head(h_flat))
        
        # 10-step imagined storm
        forecast_storm = self.imagine_forward(10)
        
        # Pre-correction logic
        corr_x, corr_y, corr_z = correction_soft.chunk(3, dim=-1)
        high_conf_x = corr_x.squeeze() > 0.96
        high_conf_z = corr_z.squeeze() > 0.96
        
        # Apply to Pauli frame (free)
        self.pauli_frame_x[high_conf_x] = 1 - self.pauli_frame_x[high_conf_x]
        self.pauli_frame_z[high_conf_z] = 1 - self.pauli_frame_z[high_conf_z]
        
        physical_flips_needed = high_conf_x | high_conf_z if storm_level < 0.4 else torch.zeros_like(high_conf_x)
        
        return {
            'correction_soft': correction_soft,
            'storm_level': storm_level.item(),
            'forecast_storm': forecast_storm,
            'bias_forecast': bias_forecast,
            'physical_flips_x': high_conf_x.int(),
            'physical_flips_z': high_conf_z.int(),
            'needs_physical': physical_flips_needed,
            'next_syndrome_pred': next_synd_pred
        }

def load_symbiotic(model_path='fridge_d7.pt', device='cpu'):
    model = SymbioticFridge(device=device)
    if os.path.exists(model_path):
        old_state = torch.load(model_path, map_location=device)
        # Map only the shared weights (GRU + Embeds + Next Syndrome)
        new_state = model.state_dict()
        transfer_count = 0
        
        for k, v in old_state.items():
            # Strict shape checking ensures we only load what fits
            if k in new_state and v.shape == new_state[k].shape:
                new_state[k] = v
                transfer_count += 1
            else:
                if k in new_state:
                    print(f"Skipping {k}: Shape mismatch {v.shape} vs {new_state[k].shape}")
        
        model.load_state_dict(new_state, strict=False)
        print(f"Symbiotic Fridge v3.1 loaded - {transfer_count} weight layers transferred.")
    else:
        print("Warning: No checkpoint found. Initializing fresh brain.")
    return model

# Demo loop
if __name__ == "__main__":
    print("--- Spinning up Symbiotic Precog (v3.1) ---")
    fridge = load_symbiotic('fridge_d7.pt', device='cpu')
    
    print("--- Running 10,000 event Imagination Test ---")
    t0 = time.time_ns()
    for i in range(10000):
        # Simulated input (98 inputs)
        fake_synd = torch.randn(98) * 0.01 
        out = fridge(fake_synd)
        
        if i % 2000 == 0:
            latency = (time.time_ns() - t0) / 2000 / 1000
            t0 = time.time_ns()
            print(f"Event {i:,} | Storm: {out['storm_level']:.3f} | Forecast (10-step): {out['forecast_storm']:.3f} | Latency: {latency:.2f}µs")
