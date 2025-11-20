import pickle
import os

checkpoint_path = 'results/checkpoints/policies/default_policy/policy_state.pkl'

print(f"--- DRILLING INTO 'weights': {checkpoint_path} ---")
try:
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    weights = data.get('weights')
    
    if isinstance(weights, dict):
        print(f"✅ 'weights' is a dictionary with {len(weights)} entries.")
        print("\n--- FIRST 20 WEIGHT KEYS ---")
        for i, k in enumerate(weights.keys()):
            if i < 20: print(f"'{k}'")
    else:
        print(f"⚠️ 'weights' is type: {type(weights)}. Printing raw value (truncated): {str(weights)[:200]}")

except Exception as e:
    print(f"❌ Error: {e}")
