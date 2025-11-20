import torch
import ray

print("--- CUDA CHECK ---")
print(f"Torch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: Torch cannot see the GPU!")

print("\n--- RAY CHECK ---")
ray.init(num_gpus=1)
print(f"Ray Resources: {ray.cluster_resources()}")
