# PeakPyAI

A simple project to do some peak detection in scientific data. Very quick implementation, so use with caution!

## Usage

The module has two types of peak detectors. The first one is a general peak detector called PeakDetector. The second, GuassianPeakDetector, is made for smaller intervals that contain three peaks, this one is made with the intention of adding peakheight detection of overlapping gaussian peaks in the future as well.

### 1. PeakDetector

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

### 2. GaussianPeakDetector

From usage_gaussian_peak_detector.py

```python
from PeakPyAI.gaussian_detector import *

dataset_size = 50
epoch_amount = 200
learning_rate = 0.0001

# Training
dataset = GaussianPeakDataset(dataset_size)
model = GaussianPeakDetector()
pipeline = GaussianTrainingPipeline(model, dataset, lr=learning_rate)

pipeline.train(epochs=epoch_amount)
pipeline.plot_losses()
pipeline.save_model('PeakPyAI/trained_gaussian_model.pt')

# Testing
test_signal, test_peaks, test_heights = GaussianDataGenerator.generate_random_data()
predicted_peaks = GaussianTesting.test_model(model, test_signal)

GaussianTesting.visualize_results(test_signal, test_peaks, predicted_peaks)
```
