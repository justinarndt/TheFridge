import torch
import torch.nn as nn
import collections
import os
import pickle
import numpy as np

# --- The Fridge Class (Matched to Checkpoint DNA) ---
class TheFridge(nn.Module):
    def __init__(self):
        super().__init__()
        # Dimensions extracted from your FATAL ERROR logs:
        self.input_size = 18        # "size mismatch... copying param with shape [128, 18]"
        self.embed_dim = 128        # "copying param with shape [128, 18]"
        self.hidden_size = 256      # "gru.weight_hh_l0... shape [768, 256]" (768/3 = 256)
        self.output_size = 76       # "correction_head... shape [76, 256]"
        
        # 1. Embedding
        self.detector_embed = nn.Linear(self.input_size, self.embed_dim)
        
        # 2. Core GRU (1 Layer, Hidden 256)
        self.gru = nn.GRU(input_size=self.embed_dim, hidden_size=self.hidden_size, num_layers=1, batch_first=False)
        
        # 3. Heads
        self.correction_head = nn.Linear(self.hidden_size, self.output_size)
        self.next_syndrome_head = nn.Linear(self.hidden_size, 18) # Matches input size 18
        
        # The Storm and Bias heads are new features, so we initialize them randomly
        self.storm_head = nn.Linear(self.hidden_size, 1)
        self.bias_head  = nn.Linear(self.hidden_size, 18) # Placeholder size
        
        # Persistent state
        self.register_buffer('h', torch.zeros(1, 1, self.hidden_size))
        # Placeholder frame
        self.pauli_frame = torch.zeros(18, 3) 

    def forward(self, x):
        # x shape: (1, 1, 18)
        x = self.detector_embed(x).unsqueeze(0).unsqueeze(0)
        _, self.h = self.gru(x, self.h.detach())
        h_flat = self.h[-1].squeeze(0)
        
        correction = self.correction_head(h_flat)
        storm = torch.sigmoid(self.storm_head(h_flat))
        return correction, storm

def convert_checkpoint_to_fridge(checkpoint_path, save_path='fridge.pt'):
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
        
    source_weights = data['weights']
    print(f"Loaded {len(source_weights)} weights from checkpoint.")

    fridge = TheFridge()
    new_state_dict = collections.OrderedDict()
    
    # Exact mapping based on your error log
    mapping = {
        'encoder.0.weight':              'detector_embed.weight',
        'encoder.0.bias':                'detector_embed.bias',
        'gru.weight_ih_l0':              'gru.weight_ih_l0',
        'gru.weight_hh_l0':              'gru.weight_hh_l0',
        'gru.bias_ih_l0':                'gru.bias_ih_l0',
        'gru.bias_hh_l0':                'gru.bias_hh_l0',
        'logits.weight':                 'correction_head.weight',
        'logits.bias':                   'correction_head.bias',
        'world_model.weight':            'next_syndrome_head.weight',
        'world_model.bias':              'next_syndrome_head.bias'
    }
    
    for src, tgt in mapping.items():
        if src in source_weights:
            new_state_dict[tgt] = torch.tensor(source_weights[src])
            print(f"Mapped: {src} -> {tgt}")
        else:
            print(f"⚠️ Warning: Source key '{src}' not found.")

    try:
        fridge.load_state_dict(new_state_dict, strict=False)
        torch.save(fridge.state_dict(), save_path)
        print(f"\nSUCCESS: The Fridge (v3) model saved to {save_path}")
    except RuntimeError as e:
        print(f"\nFATAL SHAPE MISMATCH: {e}")

if __name__ == '__main__':
    CHECKPOINT_FILE = 'results/checkpoints/policies/default_policy/policy_state.pkl'
    if os.path.exists(CHECKPOINT_FILE):
        convert_checkpoint_to_fridge(CHECKPOINT_FILE)
    else:
        print(f"Check path: {CHECKPOINT_FILE}")
