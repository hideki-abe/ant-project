import matplotlib.pyplot as plt
import numpy as np


def plot_sine_waves(params, timesteps=1000, separate_plots=False):
    """
    Plot the sine waves used for controlling the ant

    Args:
        params (list): List of 24 parameters (8 frequencies, 8 amplitudes, and 8 phase shifts)
        timesteps (int): Number of timesteps to plot
        separate_plots (bool): If True, plot each sine wave in a separate subplot
    """

    # Extract frequencies, amplitudes, and phase shifts
    frequencies = params[:8]
    amplitudes = params[8:16]
    phase_shifts = params[16:]

    # Ensure amplitudes are between 0 and 1
    amplitudes = np.clip(amplitudes, 0, 1)

    # Create time array with the same time_factor used in evaluate_ant_with_sine_waves
    time = np.arange(timesteps) / 10.0  # Scale time for reasonable frequencies

    if separate_plots:
        # Create a figure with 8 subplots (2x4 grid)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True)
        axes = axes.flatten()

        # Plot each sine wave in its own subplot
        for i in range(8):
            actions = amplitudes[i] * np.sin(frequencies[i] * time + phase_shifts[i])
            axes[i].plot(time, actions)
            axes[i].set_title(f"Joint {i+1}: f={frequencies[i]:.2f}, A={amplitudes[i]:.2f}, φ={phase_shifts[i]:.2f}")
            axes[i].grid(True)

        # Set common labels with proper padding
        fig.text(0.5, 0.01, "Time (scaled by factor 1/20)", ha='center')
        fig.text(0.04, 0.5, "Joint Action", va='center', rotation='vertical')
        plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])  # Adjust the layout to leave room for labels
        plt.savefig("sine_waves_individual_plots.png")
    else:
        # Create a single figure with all sine waves
        plt.figure(figsize=(12, 8))

        # Plot each sine wave
        for i in range(8):
            actions = amplitudes[i] * np.sin(frequencies[i] * time + phase_shifts[i])
            plt.plot(time, actions, label=f"Joint {i+1}: f={frequencies[i]:.2f}, A={amplitudes[i]:.2f}, φ={phase_shifts[i]:.2f}")

        plt.title("Sine Wave Control Signals for Ant Joints")
        plt.xlabel("Time (scaled by factor 1/20)")
        plt.ylabel("Joint Action")
        plt.legend()
        plt.grid(True)
        plt.savefig("sine_waves_plot.png")