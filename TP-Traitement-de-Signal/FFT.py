import numpy as np 
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq




def compute_fft(signal, sampling_rate):
    N = len(signal)
    fft_values = fft(signal)
    fft_magnitudes = np.abs(fft_values) / N
    fft_frequencies = fftfreq(N, 1/sampling_rate)
    return fft_frequencies[:N//2], fft_magnitudes[:N//2]            

def plot_signal_and_spectrum(t, signal, fft_frequencies, fft_magnitudes):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Time Domain Signal')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 1, 2)
    plt.stem(fft_frequencies, fft_magnitudes, use_line_collection=True)
    plt.title('Frequency Domain Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":  
    frequencies = [5, 50, 120]
    amplitudes = [1.0, 0.5, 0.2]
    phases = [0, np.pi/4, np.pi/2]
    sampling_rate = 1000
    duration = 1.0

    t, signal = generate_signal(frequencies, amplitudes, phases, sampling_rate, duration)
    fft_frequencies, fft_magnitudes = compute_fft(signal, sampling_rate)
    plot_signal_and_spectrum(t, signal, fft_frequencies, fft_magnitudes)














        
