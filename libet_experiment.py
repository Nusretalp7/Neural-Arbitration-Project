import numpy as np
import matplotlib.pyplot as plt

class WongWangAttractor:
    def __init__(self, dt=0.0005): 
        self.dt = dt
        
        # --- PARAMETERS (Wong & Wang, 2006) ---
        self.tau_S = 0.100        
        self.a = 270.0  
        self.b = 108.0  
        self.d = 0.154  
        self.J_11 = 0.2609  
        self.J_22 = 0.2609  
        self.J_12 = 0.0497  
        self.J_21 = 0.0497  
        self.gamma = 0.641 

        # Input Current
        self.I_0 = 0.335 
        
        # Noise Level
        self.sigma_noise = 0.05 
        
        # Initial Values
        self.S1 = 0.1
        self.S2 = 0.1
        
        self.history = {'t': [], 'r1': [], 'r2': []}
        self.time = 0

    def H_function(self, x):
        numerator = self.a * x - self.b
        denominator = 1 - np.exp(-self.d * (self.a * x - self.b))
        if np.abs(denominator) < 1e-9: return 0
        return numerator / denominator

    def step(self, I_stim1, I_stim2):
        # Noise (Scaled for dt)
        noise1 = np.random.normal(0, 1) * np.sqrt(self.dt) * self.sigma_noise / np.sqrt(self.tau_S) 
        noise2 = np.random.normal(0, 1) * np.sqrt(self.dt) * self.sigma_noise / np.sqrt(self.tau_S)

        # Current Calculation
        x1 = self.J_11 * self.S1 - self.J_12 * self.S2 + I_stim1 + self.I_0 + noise1
        x2 = self.J_22 * self.S2 - self.J_21 * self.S1 + I_stim2 + self.I_0 + noise2
        
        # Firing Rate (Hz)
        r1 = self.H_function(x1)
        r2 = self.H_function(x2)
        
        # Synaptic Variable Update
        dS1 = (-self.S1 / self.tau_S + (1 - self.S1) * self.gamma * r1) * self.dt
        dS2 = (-self.S2 / self.tau_S + (1 - self.S2) * self.gamma * r2) * self.dt
        
        self.S1 = np.clip(self.S1 + dS1, 0, 1)
        self.S2 = np.clip(self.S2 + dS2, 0, 1)
        
        self.time += self.dt
        self.history['t'].append(self.time)
        self.history['r1'].append(r1)
        self.history['r2'].append(r2)
        
        return r1, r2

# --- EXPERIMENT: SPONTANEOUS DECISION ---
if __name__ == "__main__":
    model = WongWangAttractor(dt=0.0005) 
    
    # Duration of the simulation
    duration = 4.0  
    steps = int(duration / model.dt)
    
    stimulus_1 = np.zeros(steps)
    stimulus_2 = np.zeros(steps)
    
    print(f"Simulation running... (I_0={model.I_0} nA)")
    
    decision_threshold = 20.0 # Hz (Decision threshold)
    decision_time = None
    winner = None

    for i in range(steps):
        r1, r2 = model.step(stimulus_1[i], stimulus_2[i])
        
        if decision_time is None:
            if r1 > decision_threshold:
                decision_time = model.time
                winner = "Population 1 (Blue)"
                print(f"Decision REACHED! Winner: {winner} at {decision_time:.3f}s")
            elif r2 > decision_threshold:
                decision_time = model.time
                winner = "Population 2 (Red)"
                print(f"Decision REACHED! Winner: {winner} at {decision_time:.3f}s")

    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    plt.plot(model.history['t'], model.history['r1'], label='Population 1', color='blue', linewidth=2, alpha=0.8)
    plt.plot(model.history['t'], model.history['r2'], label='Population 2', color='red', linewidth=2, linestyle='--', alpha=0.8)
    plt.axhline(y=decision_threshold, color='green', linestyle=':', label='Threshold')
    
    if decision_time:
        plt.axvline(x=decision_time, color='black', linestyle='-.')
        plt.title(f'Spontaneous Decision \nWinner: {winner} at {decision_time:.3f}s', fontsize=14)
    else:
        plt.title(f'No Decision (Increase I_0 slightly more)', fontsize=14)
        
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Firing Rate (Hz)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()