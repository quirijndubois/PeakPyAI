import matplotlib.pyplot as plt
import numpy as np

def mapRange(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def normal_distribution(x,mu,sigma):
    return np.exp(-((x-mu)**2)/(2*sigma**2))

def plot_data(signal=None,actual_peaks=None,detected_peaks=None, probabilities=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if signal is not None:
        ax1.plot(signal)
        ax1.set_title('Signal')

    if probabilities is not None:
        ax2.plot(probabilities)
        ax2.set_title('Peak Probabilities')
    

    if actual_peaks is not None:
        ax1.axvline(x=actual_peaks[0], color='r', linestyle='--',label='actual peaks')
        for peak in actual_peaks:
            ax1.axvline(x=peak, color='r', linestyle='--')

    if detected_peaks is not None:
        ax2.axvline(x=detected_peaks[0], color='g', linestyle='--',label='detected peaks')
        for peak in detected_peaks:
            ax2.axvline(x=peak, color='g', linestyle='--')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')

    plt.show()

def generate_random_data(
        resolution=200,
        peaks_range=[20,20],
        peak_height_range=[0.1,1],
        std_range=[0.1,2],
        noise_strength=0.02):
    
    n_peaks = int(mapRange(np.random.random(),0,1,peaks_range[0],peaks_range[1]))

    signal = np.zeros(resolution)
    peaks = []
    for _ in range(n_peaks):
        y_peak = mapRange(np.random.random(),0,1,peak_height_range[0],peak_height_range[1])
        x_peak = mapRange(np.random.random(),0,1,0,resolution)
        std = mapRange(np.random.random(),0,1,std_range[0],std_range[1])
        peaks.append(round(x_peak))
        signal += normal_distribution(np.arange(resolution),x_peak,std) * y_peak
    signal += (np.random.randn(resolution)*2 - 1)*noise_strength

    return [signal,peaks]