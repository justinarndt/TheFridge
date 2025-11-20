import os
import ray
from ray import tune
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from sklearn.decomposition import PCA
from ray.rllib.algorithms.ppo import PPO
from qec_survivor.env.catastrophe_env import QEC_Catastrophe_Env
from qec_survivor.models.qec_model import QEC_Physics_Model
from ray.rllib.models import ModelCatalog

# 1. REGISTRATION (Critical Step previously missing)
# We must teach this script what "QEC_Catastrophe_Env" means
tune.register_env("QEC_Catastrophe_Env", lambda config: QEC_Catastrophe_Env(config))
ModelCatalog.register_custom_model("qec_physics_model", QEC_Physics_Model)

ray.init()

def run_forensic_analysis():
    print("--- STARTING FORENSIC ANALYSIS ---")
    
    # 2. Restore the Agent
    # Pointing directly to the checkpoint folder
    checkpoint_path = "./results/checkpoints"
    
    print(f"Loading Checkpoint from: {checkpoint_path}")
    algo = PPO.from_checkpoint(checkpoint_path)
    
    # 3. Setup the Test Bed (Catastrophe Environment)
    env_config = {
        "distance": 5,
        "schedule": { 
            50: {"freq_mult": 2.0},      # The Jump
            100: {"drift_mult": 3.0},    # The Drift
            150: {"burst": True, "add_freq": 0.18} # The Burst
        }
    }
    env = QEC_Catastrophe_Env(env_config)
    
    # 4. Run Episode and Spy on the Brain (GRU)
    obs, _ = env.reset()
    done = False
    
    # Initialize hidden state (zeros)
    # shape: [1, hidden_dim] (Batch=1)
    h_t = [np.zeros(256, dtype=np.float32)] 
    
    hidden_states = []
    rewards = []
    
    print("Running Test Episode (300 Rounds)...")
    while not done:
        # Compute action and get next hidden state
        action, state_out, _ = algo.compute_single_action(
            observation=obs,
            state=h_t,
            explore=False # Deterministic for analysis
        )
        
        # Store the hidden state
        hidden_states.append(h_t[0])
        
        # Step Env
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(reward)
        
        h_t = state_out
        if truncated: done = True

    print(f"Episode Complete. Total Reward: {sum(rewards)}")

    # 5. Data Processing (The "Smoking Gun" Logic)
    H = np.array(hidden_states)
    
    print("Performing PCA on Hidden States...")
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(H).flatten()
    
    # STFT: Time-Frequency Analysis
    f, t, Sxx = spectrogram(pc1, fs=1.0, nperseg=30, noverlap=28)
    
    # 6. Visualization (The Kill Shot)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='inferno')
    plt.ylabel('Frequency (Hz normalized)')
    plt.xlabel('Time (Rounds)')
    plt.title('The Smoking Gun: Survivor QEC Adaptive Phase-Locking')
    plt.colorbar(label='Neural Activation Power')
    
    # Annotate the Events
    plt.axvline(x=50, color='cyan', linestyle='--', alpha=0.8, label='Freq Jump (60->120Hz)')
    plt.axvline(x=150, color='white', linestyle='--', alpha=0.8, label='Burst Event')
    
    plt.legend(loc='upper right')
    
    output_path = "analysis/smoking_gun.png"
    plt.savefig(output_path, dpi=300)
    print(f"SUCCESS: Analysis saved to {output_path}")
    print("Check this image to verify the frequency jump at t=50!")

if __name__ == "__main__":
    run_forensic_analysis()
