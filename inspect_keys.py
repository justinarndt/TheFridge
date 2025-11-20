import pickle
import os

checkpoint_path = 'results/checkpoints/policies/default_policy/policy_state.pkl'

if not os.path.exists(checkpoint_path):
    print(f"❌ File not found: {checkpoint_path}")
    exit()

print(f"--- INSPECTING: {checkpoint_path} ---")
try:
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle different RLlib structures
    if isinstance(data, dict) and 'module' in data:
        state_dict = data['module']
        print("✅ Found state_dict under 'module' key.")
    elif isinstance(data, dict):
        state_dict = data
        print("⚠️ 'module' key not found. Using root dictionary.")
    else:
        print("❌ Data is not a dictionary.")
        exit()

    print("\n--- FIRST 20 KEYS ---")
    keys = list(state_dict.keys())
    for k in keys[:20]:
        print(f"'{k}'")
        
except Exception as e:
    print(f"❌ Error: {e}")
