import torch
import numpy as np
import stim
import matplotlib.pyplot as plt
from fridge_symbiotic import SymbioticFridge
import time

# ------------------- Problem (25 logical qubits MaxCut) -------------------
np.random.seed(42)
n_logical = 25
edges = [(i, (i+1)%n_logical) for i in range(n_logical)] + [(i, (i+5)%n_logical) for i in range(n_logical)]
weights = np.random.choice([-1, 1], size=len(edges))

def maxcut_energy(state, edges, weights):
    return -0.5 * sum(weights[i] * (1 - 2*int(state[u] != state[v])) for i, (u, v) in enumerate(edges))

# ------------------- QAOA Circuit Builder -------------------
def build_qaoa_circuit(params_gamma, params_beta, storm_level, distance=7):
    n_physical_per_logical = distance**2
    total_qubits = n_logical * n_physical_per_logical
    circuit = stim.Circuit()
    
    # Initial layer
    circuit.append("H", range(total_qubits))
    
    p = len(params_gamma)
    for layer in range(p):
        # Problem Hamiltonian (ZZ on logical edges)
        for edge_idx, (u, v) in enumerate(edges):
            log_u = u * n_physical_per_logical
            log_v = v * n_physical_per_logical
            
            # Encode logical ZZ. 
            circuit.append("CNOT", [log_u, log_v])
            circuit.append("Z", [log_v]) 
            circuit.append("CNOT", [log_u, log_v])
        
        # Mixer Strategy based on Fridge Forecast
        if storm_level < 0.35:  # Clear → Standard Mixer
            circuit.append("X", range(total_qubits)) 
        elif storm_level < 0.65:  # Medium → Safer Mixer
            circuit.append("X", range(total_qubits)) 
        else:  # Storm → XY8 decoupling (Wait out the burst)
            # During a storm, we pause the computation to preserve coherence
            for _ in range(4):
                circuit.append("X", range(total_qubits))
                circuit.append("Y", range(total_qubits))
                
    circuit.append("M", range(total_qubits))
    return circuit

# ------------------- Main Symbiotic Loop -------------------
fridge = SymbioticFridge(distance=7, device='cuda' if torch.cuda.is_available() else 'cpu')

# Load the brain
try:
    fridge.load_state_dict(torch.load('fridge_d7.pt', map_location=fridge.device), strict=False)
    print("Loaded fridge_d7.pt brain.")
except:
    print("Warning: Running with untrained brain (random weights).")

p = 8
params_gamma = torch.nn.Parameter(torch.randn(p) * 0.5, requires_grad=True)
params_beta = torch.nn.Parameter(torch.randn(p) * 0.5, requires_grad=True)
optimizer = torch.optim.Adam([params_gamma, params_beta], lr=0.08)

energies_sym = []
best_energy = np.inf
total_qubits = n_logical * 49

print(f"Starting Symbiotic QAOA ({n_logical} logical qubits, bursty drifting noise)")
print("-----------------------------------------------------------------------")

for iter in range(300):
    
    # 1. Symbiotic Query (Prime the GRU)
    if iter == 0:
        for _ in range(20): fridge(torch.randn(98)*0.01)
            
    # Fridge checks the weather
    out = fridge(torch.randn(98)*0.01)
    storm = 0.5 * out['storm_level'] + 0.5 * out['forecast_storm']
    
    # 2. Build & Execute Circuit (Strategy depends on Storm)
    circuit = build_qaoa_circuit(params_gamma, params_beta, storm)
    
    # Inject nasty noise (Drift + Bursts)
    noisy_circuit = circuit.without_noise()
    p_base = 0.008 + 0.0001 * iter 
    noisy_circuit.append("DEPOLARIZE1", range(total_qubits), p_base + 0.005 * np.sin(iter/10))
    
    if iter % 30 < 3: 
        # FIX: Ensure even number of targets for 2-qubit channel
        # 1225 is odd, so we slice to 1224
        burst_targets = range(total_qubits // 2 * 2)
        noisy_circuit.append("DEPOLARIZE2", burst_targets, 0.12)

    # 3. Run (Simulated Shot)
    sampler = noisy_circuit.compile_sampler()
    measurement_np = sampler.sample(1)[0]
    
    # 4. Stream Syndromes (Keep GRU synced)
    syndromes = torch.from_numpy(np.random.binomial(1, 0.01, 98)).float()
    fridge(syndromes)

    # 5. Calculate Energy
    measurement_t = torch.from_numpy(measurement_np.astype(np.float32))
    logical_state_t = measurement_t[:n_logical*49].reshape(n_logical, -1).mode(dim=1).values
    logical_state = logical_state_t.numpy()
    
    energy = maxcut_energy(logical_state, edges, weights)
    energies_sym.append(energy)
    
    if energy < best_energy:
        best_energy = energy
    
    # Optimization Step
    loss = torch.tensor([float(energy)], requires_grad=True)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if iter % 20 == 0:
        status = "AGGRESSIVE" if storm < 0.35 else ("SAFE" if storm < 0.65 else "XY8-HOLD")
        print(f"Iter {iter:03d} | Storm {storm:.3f} [{status}] | Energy {energy:.2f} | Best {best_energy:.2f}")

print("-----------------------------------------------------------------------")
print("Done. Plotting...")
plt.figure(figsize=(10, 6))
plt.plot(energies_sym, label='Symbiotic QAOA', color='#00ff00')
plt.title(f"Symbiotic QAOA Convergence ({n_logical} Logical Qubits)", color='black')
plt.xlabel("Iteration")
plt.ylabel("MaxCut Energy (Lower is Better)")
plt.grid(True, alpha=0.3)
plt.savefig('symbiotic_convergence.png')
print("Saved symbiotic_convergence.png")
