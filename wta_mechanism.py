import numpy as np
import matplotlib.pyplot as plt

class NeuralArbitrator:
    def __init__(self, dt=0.001, tau=0.1, noise_amp=0.02):
        """
        Implementation of the Wong & Wang (2006) Reduced Attractor Model.
        
        Parameters:
        -----------
        dt : float
            Simulation time step in seconds (e.g., 0.001 = 1ms).
        tau : float
            Synaptic time constant (approx. 100ms for NMDA receptors).
        noise_amp : float
            Amplitude of the stochastic noise term (simulating neural fluctuations).
        """
        self.dt = dt
        self.tau = tau
        self.noise_amp = noise_amp
        
        # Model Parameters (Adapted from Wong & Wang, 2006 reduced model)
        # These weights determine the attractor dynamics.
        self.w_self = 1.0      # Weight of self-excitation (enables 'Latching' / Memory)
        self.w_inhibit = 1.0   # Weight of mutual inhibition (enables 'Competition' / WTA)
        
        # Initial State (Firing Rates - represented in arbitrary units or Hz)
        # We start with low baseline activity to simulate spontaneous background noise.
        self.r1 = 0.1 
        self.r2 = 0.1
        
        # History log for visualization purposes
        self.history = {'t': [], 'r1': [], 'r2': []}
        self.time = 0

    def transfer_function(self, x):
        """
        Frequency-Current (F-I) curve. 
        Acts as a Rectified Linear Unit (ReLU) to ensure firing rates remain non-negative.
        Simulates the threshold-linear response of neural populations.
        """
        return np.maximum(0, x)

    def step(self, input_1, input_2):
        """
        Advances the simulation by one time step (dt).
        
        Parameters:
        -----------
        input_1 : float
            External sensory evidence driving Population 1 (e.g., Left Choice).
        input_2 : float
            External sensory evidence driving Population 2 (e.g., Right Choice).
            
        Returns:
        --------
        r1, r2 : float
            Updated firing rates for both populations.
        """
        
        # 1. Noise Generation (Stochastic Term)
        # Scaled by sqrt(dt) for proper integration of the stochastic differential equation (Euler-Maruyama method).
        noise1 = np.random.normal(0, 1) * np.sqrt(self.dt) * self.noise_amp
        noise2 = np.random.normal(0, 1) * np.sqrt(self.dt) * self.noise_amp
        
        # 2. Compute Total Synaptic Input
        # Input = External Stimulus + (Self-Excitation) - (Cross-Inhibition)
        total_input_1 = input_1 + (self.w_self * self.r1) - (self.w_inhibit * self.r2)
        total_input_2 = input_2 + (self.w_self * self.r2) - (self.w_inhibit * self.r1)
        
        # 3. Compute Derivatives (The Dynamics)
        # Equation: tau * dR/dt = -R + F(Input)
        dr1 = (-self.r1 + self.transfer_function(total_input_1)) / self.tau
        dr2 = (-self.r2 + self.transfer_function(total_input_2)) / self.tau
        
        # 4. Update State (Euler Integration)
        self.r1 += dr1 * self.dt + noise1
        self.r2 += dr2 * self.dt + noise2
        
        # 5. Enforce Biological Constraint (Non-negative firing rates)
        self.r1 = max(0, self.r1)
        self.r2 = max(0, self.r2)
        
        # Advance internal clock and log history
        self.time += self.dt
        self.history['t'].append(self.time)
        self.history['r1'].append(self.r1)
        self.history['r2'].append(self.r2)
        
        return self.r1, self.r2

# --- SIMULATION AND TESTING ---
# This section generates the Figure for the 'Results' section of your report.

if __name__ == "__main__":
    # Initialize the Neural Arbitrator
    model = NeuralArbitrator(dt=0.001, tau=0.1, noise_amp=0.05)

    # Simulation Settings
    duration = 2.0  # seconds
    steps = int(duration / model.dt)

    # Generate Input Signals (Simulating Human Intent)
    # Scenario: "Perceptual Decision"
    # First 0.5s: Silence (Baseline)
    # After 0.5s: Strong evidence for Option 1 (Left), weak evidence for Option 2 (Right)
    inputs_1 = np.zeros(steps)
    inputs_2 = np.zeros(steps)

    inputs_1[500:] = 0.5  # Stronger signal for Left (Target)
    inputs_2[500:] = 0.4  # Weaker signal for Right (Distractor)

    # Run Simulation Loop
    print("Running simulation...")
    for i in range(steps):
        model.step(inputs_1[i], inputs_2[i])

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Plot firing rates
    plt.plot(model.history['t'], model.history['r1'], label='Population 1 (Left - Winner)', color='blue', linewidth=2)
    plt.plot(model.history['t'], model.history['r2'], label='Population 2 (Right - Loser)', color='red', linewidth=2, linestyle='--')
    
    # Mark stimulus onset
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Stimulus Onset')
    
    # Formatting
    plt.title('Winner-Take-All Dynamics: Decision Latching', fontsize=14)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Firing Rate (Activity)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show plot
    plt.show()