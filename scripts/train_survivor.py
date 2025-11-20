import ray
from ray.rllib.algorithms.ppo import PPOConfig
from qec_survivor.env.catastrophe_env import QEC_Catastrophe_Env
from qec_survivor.models.qec_model import QEC_Physics_Model
from ray.rllib.models import ModelCatalog
import os
import ray.tune as tune

# Register Environment
tune.register_env("QEC_Catastrophe_Env", lambda config: QEC_Catastrophe_Env(config))

# Register Custom Model
ModelCatalog.register_custom_model("qec_physics_model", QEC_Physics_Model)

# Create Config Object
config = PPOConfig()

# --- CRITICAL FIX: DISABLE NEW API STACK ---
# This forces Ray to use the standard ModelV2 API compatible with our custom model
config = config.api_stack(
    enable_rl_module_and_learner=False, 
    enable_env_runner_and_connector_v2=False
)

# 1. Environment
config = config.environment(
    env="QEC_Catastrophe_Env",
    env_config={
        "distance": 5,
        "schedule": { 
            50: {"freq_mult": 2.0},
            100: {"drift_mult": 3.0},
            150: {"burst": True, "add_freq": 0.18}
        }
    }
)

# 2. Framework & Resources
config = config.framework("torch")
config = config.resources(num_gpus=1)

# 3. Env Runners
config = config.env_runners(
    num_env_runners=8,
    num_envs_per_env_runner=1
)

# 4. Training
config = config.training(
    model={
        "custom_model": "qec_physics_model",
        "custom_model_config": {
            "hidden_dim": 256,
            "aux_weight": 3.0,
            "history_length": 8
        },
        "fcnet_hiddens": [],
        "fcnet_activation": "relu",
    },
    lr=1e-4,
    gamma=0.99,
    train_batch_size=4000
)

# 5. PPO Hyperparameters
config.sgd_minibatch_size = 256
config.num_sgd_iter = 30
config.kl_coeff = 0.2
config.lambda_ = 0.95

if __name__ == "__main__":
    ray.init()
    
    print("----------------------------------------------------------------")
    print("INITIATING: SURVIVOR QEC (Hard Fork v2)")
    print("MODE: Catastrophe Regime - MANUAL CONTROL LOOP")
    print("DEVICE: NVIDIA RTX 4060 (via WSL)")
    print("API STACK: Legacy (ModelV2)")
    print("----------------------------------------------------------------")
    
    # Build Algorithm
    algo = config.build()
    
    save_dir = "./results/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting Training Loop (200 Iterations)...")
    for i in range(1, 201):
        result = algo.train()
        
        # Robust reward extraction
        if 'env_runners' in result:
            reward = result['env_runners']['episode_reward_mean']
        else:
            reward = result['episode_reward_mean']
            
        print(f"Iter: {i:3d} | Mean Reward: {reward:.4f}")
        
        if i % 10 == 0:
            checkpoint_path = algo.save(save_dir)
            print(f"Saved Checkpoint: {checkpoint_path}")
