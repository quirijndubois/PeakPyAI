from PeakPyAI import PeakDetector, generate_random_data, plot_data

# Training
detector = PeakDetector()
detector.generate_random_training_data()
model = detector.train_peak_detection_model()

# Prediction
example_signal, example_peaks = generate_random_data()
detector.predict_peak_probabilities(example_signal)
detector.calculate_peaks_by_probability_threshold()

#plotting
plot_data(
    signal=example_signal,
    actual_peaks=example_peaks,
    detected_peaks=detector.detected_peaks,
    probabilities=detector.peak_probabilities
)
