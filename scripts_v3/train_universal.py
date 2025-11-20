import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray import tune
import os

from qec_universal.env.adversarial_env import Adversarial_QEC_Env
from qec_universal.models.transformer_model import QEC_Transformer_Model

tune.register_env("Adversarial_QEC_Env", lambda config: Adversarial_QEC_Env(config))
ModelCatalog.register_custom_model("qec_transformer", QEC_Transformer_Model)

config = PPOConfig()
config = config.api_stack(
    enable_rl_module_and_learner=False, 
    enable_env_runner_and_connector_v2=False
)

config = config.environment(
    env="Adversarial_QEC_Env",
    env_config={
        "distance": 5,
        "adversary_strength": 0.10, # REDUCED to 10% for learning phase
        "context_len": 16           # CRITICAL: Matches Model Logic
    }
)

config = config.framework("torch")
config = config.resources(num_gpus=1)

config = config.env_runners(num_env_runners=8, num_envs_per_env_runner=1)

config = config.training(
    model={
        "custom_model": "qec_transformer",
        "custom_model_config": {
            "d_model": 128,
            "nhead": 4,
            "num_layers": 2,
            "aux_weight": 3.0
        },
        "fcnet_hiddens": [],
        "fcnet_activation": "relu",
    },
    lr=5e-5,
    gamma=0.99,
    train_batch_size=4000
)
config.sgd_minibatch_size = 128
config.num_sgd_iter = 20

if __name__ == "__main__":
    ray.init()
    print("----------------------------------------------------------------")
    print("PROJECT INVINCIBLE (v3.1): CONTEXT-AWARE UNIVERSAL DECODER")
    print("Architecture: Transformer with 16-Frame Memory Buffer")
    print("Difficulty: Medium (10% Adversarial Noise)")
    print("----------------------------------------------------------------")
    
    algo = config.build()
    save_dir = "./results_v3/checkpoints"
    os.makedirs(save_dir, exist_ok=True)

    print("Starting Training...")
    for i in range(1, 201):
        result = algo.train()
        if 'env_runners' in result:
            reward = result['env_runners']['episode_reward_mean']
        else:
            reward = result['episode_reward_mean']
        print(f"Iter: {i:3d} | Mean Reward: {reward:.4f}")
        
        if i % 10 == 0:
            algo.save(save_dir)

    print("DONE.")
