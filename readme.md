# PeakPyAI

A simple project to do some peak detection in scientific data. Very quick implementation, so use with caution!

## Usage

Import the libary

```python
from PeakPyAI import PeakDetector, generate_random_data, plot_data
```

Either train a model, or load an existing one

```python
# Training
detector = PeakDetector()
detector.generate_random_training_data()
model = detector.train_peak_detection_model()
```

or 

```python
# load model
detector = PeakDetector()
detector.load_model('PeakPyAI/trained_model.pkl')
```

Use the model to find peaks

```python
# Prediction
example_signal, example_peaks = generate_random_data()
detector.predict_peak_probabilities(example_signal)
detector.calculate_peaks_by_probability_threshold()
```

Now you can plot the results

```python 
#plotting
plot_data(
    signal=example_signal,
    actual_peaks=example_peaks,
    detected_peaks=detector.detected_peaks,
    probabilities=detector.peak_probabilities
)
```