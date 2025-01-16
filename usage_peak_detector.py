from PeakPyAI import PeakDetector, generate_random_data, plot_data

# Training
detector = PeakDetector(
    max_estimators=100,
    step=10,
    test_size=0.1,
    random_state=42
)

detector.generate_random_training_data(
    data_set_size=200,
    resolution=200,
    peaks_range=[20,20],
    peak_height_range=[0.1,1],
    std_range=[0.1,2],
    noise_strength=0.01
)

detector.train_model()
detector.save_model('PeakPyAI/trained_model.pkl')

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