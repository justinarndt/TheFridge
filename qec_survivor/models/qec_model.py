import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
import torch
import torch.nn as nn

class QEC_Physics_Model(TorchModelV2, nn.Module):
    """
    Robust Implementation that bypasses the deprecated RecurrentNetwork wrapper.
    Manually handles time-dimension reshaping to avoid 'seq_lens=None' crashes.
    """
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        self.hidden_dim = model_config["custom_model_config"].get("hidden_dim", 256)
        self.aux_weight = model_config["custom_model_config"].get("aux_weight", 3.0)
        
        input_dim = obs_space.shape[0]
        
        # 1. Feature Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # 2. The Physics Engine (GRU)
        self.gru = nn.GRU(128, self.hidden_dim, batch_first=True)
        
        # 3. Heads
        self.logits = nn.Linear(self.hidden_dim, num_outputs)
        self.value_branch = nn.Linear(self.hidden_dim, 1)
        self.world_model = nn.Linear(self.hidden_dim, input_dim)
        
        # State storage for value function and loss
        self._cur_value = None
        self._last_prediction = None

    @override(TorchModelV2)
    def get_initial_state(self):
        # Crucial: Initialize the hidden state for the GRU
        # Returns list [Tensor(1, B, Hidden)]
        # We return shape [Hidden] because RLlib auto-batches it.
        return [torch.zeros(self.hidden_dim)]

    @override(TorchModelV2)
    def value_function(self):
        assert self._cur_value is not None, "must call forward() first"
        return self._cur_value.reshape(-1)

    @override(TorchModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Manual Forward Pass.
        Handles both 'Dummy Batches' (seq_lens=None) and 'Train Batches' (seq_lens=Tensor).
        """
        # Extract observations (flattened)
        obs = input_dict["obs_flat"] if "obs_flat" in input_dict else input_dict["obs"]
        
        # 1. Encode features
        x = self.encoder(obs) # Shape: [Batch * Time, 128]
        
        # 2. Recurrent Logic (The "Safe" Way)
        if seq_lens is None:
            # CASE A: Initialization / Dummy Batch
            # Assume simple mapping: 1 step per sequence
            # x is [Batch, 128], state[0] is [Batch, Hidden]
            output, new_h = self.gru(x.unsqueeze(1), state[0].unsqueeze(0))
            output = output.squeeze(1)
            new_state = [new_h.squeeze(0)]
        else:
            # CASE B: Training Batch
            # We have flattened inputs [B*T, 128] and seq_lens [B]
            # We must reshape to [B, T, 128] for the GRU
            batch_size = seq_lens.shape[0]
            max_seq_len = x.shape[0] // batch_size
            
            # Unflatten
            x_time = x.reshape(batch_size, max_seq_len, -1)
            
            # GRU Step
            # state[0] is [Batch, Hidden]. GRU expects [Layers, Batch, Hidden]
            h_in = state[0].unsqueeze(0)
            output_time, h_out = self.gru(x_time, h_in)
            
            # Flatten output back to [B*T, Hidden] for the linear heads
            output = output_time.reshape(-1, self.hidden_dim)
            new_state = [h_out.squeeze(0)]

        # 3. Compute Outputs
        self._cur_value = self.value_branch(output).squeeze(1)
        logits = self.logits(output)
        
        # 4. World Model Prediction (for Aux Loss)
        self._last_prediction = self.world_model(output)
        
        return logits, new_state

    @override(TorchModelV2)
    def custom_loss(self, policy_loss, loss_inputs):
        """
        Calculates the Physics Prediction Loss.
        """
        next_obs = loss_inputs["new_obs"]
        
        # Get our prediction from the forward pass
        pred_obs = self._last_prediction
        
        # Ensure target shape matches prediction (handle potential flattening issues)
        target = next_obs.reshape(pred_obs.shape)
        
        # Mean Squared Error: Prediction vs Reality
        reconstruction_loss = torch.mean((pred_obs - target) ** 2)
        
        # Add weighted aux loss to the policy loss(es)
        # policy_loss is a list (one per tower), usually len 1
        total_loss = [loss + (self.aux_weight * reconstruction_loss) for loss in policy_loss]
        
        return total_loss
