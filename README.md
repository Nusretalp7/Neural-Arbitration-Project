# Bio-Inspired Neural Arbitration for BCI Control

This project implements a simulation of neural arbitration mechanisms for Brain-Computer Interface (BCI) applications. It explores the use of attractor dynamics (specifically the Wong & Wang model) to filter noisy control signals and improve cursor control stability compared to raw signal mapping.

## Overview

The core of the project is a comparison between two control modes:
1.  **RAW Mode**: Direct mapping of noisy "neural" signals/intents to cursor velocity. This simulates the jitter and instability often found in raw BCI decoding.
2.  **WTA (Winner-Take-All) Mode**: Signals are passed through a neural attractor network. The network's integration and competition dynamics (mutual inhibition) help "latch" onto a decision and filter out high-frequency noise, resulting in smoother control.

## Project Structure

- **`simulation.py`**: The main interactive simulation built with **Pygame**.
  - **Visuals**: Displays the cursor, target/distractor, and real-time plots of internal neural firing rates.
  - **Controls**:
    - `SPACE`: Pause simulation.
    - `R`: Toggle between **RAW** and **WTA** modes.
    - `C`: Toggle intended target (Left/Right).
    - `I`: Reset the neural attractor.
    - `Up`/`Down`: Increase/Decrease input noise levels.
    - `H`: Toggle internal variables HUD.
- **`wta_mechanism.py`**: A standalone script visualizing the Wong & Wang Reduced Attractor Model dynamics using **Matplotlib**. Useful for understanding the decision-making latching effect.
- **`libet_experiment.py`**: Simulates a "Spontaneous Decision" experiment to observe how the model breaks symmetry under noise (similar to the Libet experiment).

## Requirements

- Python 3.x
- `pygame`
- `numpy`
- `matplotlib`

You can install the dependencies via pip:

```bash
pip install pygame numpy matplotlib
```

## Usage

To run the main simulation:

```bash
python simulation.py
```

### Simulation Controls
- The **Goal** is to move the cursor to the Right (Blue) or Left (Red) target based on the "Brain" intent.
- Observe how **RAW** mode jitters significantly with high noise.
- Switch to **WTA** mode (`R`) to see how the neural dynamics potential well stabilizes the movement.

## Theory

This project is based on the **Wong & Wang (2006)** reduced attractor model. The model consists of two competing populations of neurons (Left vs Right):
- **Self-Excitation**: Maintains active memory (integration).
- **Mutual Inhibition**: Ensures only one population wins (decision making).
- **Stochastic noise**: Simulates the noisy nature of neural firing.

## License

This project is part of the CMPE 58I coursework.
