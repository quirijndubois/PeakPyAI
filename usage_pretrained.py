from PeakPyAI import PeakDetector, generate_random_data, plot_data

# load model
detector = PeakDetector()
detector.load_model('PeakPyAI/trained_model.pkl')

# Prediction
example_signal, example_peaks = generate_random_data()
detector.predict_peak_probabilities(example_signal)
detector.calculate_peaks_by_probability_threshold()
detector.calculate_peaks_by_probability_peaks()

#plotting
plot_data(
    signal=example_signal,
    actual_peaks=example_peaks,
    detected_peaks=detector.detected_peaks,
    probabilities=detector.peak_probabilities
)
