import numpy as np
import torch
import torch.nn as nn
import math
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class QEC_Transformer_Model(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.d_model = model_config["custom_model_config"].get("d_model", 128)
        self.nhead = model_config["custom_model_config"].get("nhead", 4)
        self.num_layers = model_config["custom_model_config"].get("num_layers", 2)
        self.aux_weight = model_config["custom_model_config"].get("aux_weight", 3.0)
        
        # Config for reshaping
        self.context_len = 16 
        # Calculate single frame size: Total Obs / Context Length
        self.frame_dim = obs_space.shape[0] // self.context_len
        
        # 1. Spatial Encoder (Frame -> Latent)
        self.embedding = nn.Sequential(
            nn.Linear(self.frame_dim, self.d_model),
            nn.ReLU(),
            nn.LayerNorm(self.d_model)
        )
        
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=100)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.nhead, 
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        
        # Heads
        self.logits = nn.Linear(self.d_model, num_outputs)
        self.value_branch = nn.Linear(self.d_model, 1)
        # World Model predicts just the NEXT frame, not the whole history
        self.world_model = nn.Linear(self.d_model, self.frame_dim) 

        self._cur_value = None
        self._last_prediction = None

    @override(TorchModelV2)
    def get_initial_state(self):
        return []

    @override(TorchModelV2)
    def value_function(self):
        return self._cur_value.reshape(-1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs_flat = input_dict["obs_flat"] if "obs_flat" in input_dict else input_dict["obs"]
        
        # UNFLATTEN THE HISTORY
        # Input: [Batch, Total_Flat_Dim]
        # Desired: [Batch, Context_Len, Frame_Dim]
        x = obs_flat.reshape(-1, self.context_len, self.frame_dim)
        
        # 1. Embed & Position
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # 2. Transformer Pass (No mask needed, history is already causal by Env definition)
        # output: [Batch, Context_Len, d_model]
        output = self.transformer_encoder(x)
        
        # 3. Focus on the FINAL Step for Action
        # We take the last token in the sequence as the "summary"
        # [Batch, d_model]
        final_token = output[:, -1, :]
        
        # 4. Heads
        self._cur_value = self.value_branch(final_token).squeeze(1)
        logits = self.logits(final_token)
        self._last_prediction = self.world_model(final_token)
        
        return logits, []

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        # For Aux Loss, we need the NEXT REAL FRAME.
        # The 'new_obs' is also a flattened history.
        # We want the LAST frame of 'new_obs' which is the newest data point.
        
        next_obs_flat = loss_inputs["new_obs"]
        # Reshape: [Batch, Context_Len, Frame_Dim]
        next_obs_seq = next_obs_flat.reshape(-1, self.context_len, self.frame_dim)
        # Target is the newest frame (last index)
        target = next_obs_seq[:, -1, :]
        
        pred = self._last_prediction
        reconstruction_loss = torch.mean((pred - target) ** 2)
        
        total_loss = [loss + (self.aux_weight * reconstruction_loss) for loss in policy_loss]
        return total_loss
