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

# Visualization
GaussianTesting.visualize_results(test_signal, test_peaks, predicted_peaks)