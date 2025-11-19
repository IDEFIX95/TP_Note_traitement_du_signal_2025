import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
def plot_signals(original_signal, filtered_signal, t):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, original_signal, label='Original Signal')
    plt.title('Original Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.subplot(2, 1, 2)
    plt.plot(t, filtered_signal, label='Filtered Signal', color='orange')
    plt.title('Filtered Signal (Low-pass)')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid()
    
    plt.tight_layout()
    plt.show()
    if __name__ == "__main__":
    # Sample parameters
    fs = 500.0       # Sampling frequency
    cutoff = 50.0    # Desired cutoff frequency of the filter, Hz
    order = 6        # Order of the filter
    duration = 2.0   # seconds
    N = int(fs * duration)  # Total number of samples
    t = np.linspace(0, duration, N, endpoint=False)

    # Create a sample signal with multiple frequencies
    freqs = [5, 50, 120]
    amplitudes = [1.0, 0.5, 0.2]
    signal = sum(amplitudes[i] * np.sin(2 * np.pi * freqs[i] * t) for i in range(len(freqs)))

    # Apply low-pass filter
    filtered_signal = lowpass_filter(signal, cutoff, fs, order)

    # Plot original and filtered signals
    plot_signals(signal, filtered_signal, t)


