import numpy as np
from scipy.fft import ifft, fftfreq
import matplotlib.pyplot as plt


def compute_ifft(fft_values):
    N = len(fft_values)
    time_signal = ifft(fft_values) * N
    return time_signal.real  # Return the real part


def plot_time_signal(t, time_signal):
    plt.figure(figsize=(10, 4))
    plt.plot(t, time_signal)
    plt.title('Reconstructed Time Domain Signal from IFFT')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    plt.show()
if __name__ == "__main__":
    # Example FFT values (can be replaced with actual FFT data)
    sampling_rate = 1000
    duration = 1.0
    N = int(sampling_rate * duration)
    t = np.linspace(0, duration, N, endpoint=False)

    # Create a sample signal and compute its FFT
    frequencies = [5, 50, 120]
    amplitudes = [1.0, 0.5, 0.2]
    phases = [0, np.pi/4, np.pi/2]
    
    signal = sum(amplitudes[i] * np.sin(2 * np.pi * frequencies[i] * t + phases[i]) for i in range(len(frequencies)))
    fft_values = np.fft.fft(signal) / N

    # Compute IFFT
    time_signal = compute_ifft(fft_values)

    # Plot the reconstructed time signal
    plot_time_signal(t, time_signal)

