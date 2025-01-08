from .utils import plot_data, generate_random_data
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class PeakDetector:
    max_estimators = 100
    step = 1
    test_size = 0.1
    random_state = 42

    def create_features_and_labels(self):

        features = []
        labels = []

        # Generate features and labels
        for signal, peaks in self.training_data:
            for i in range(len(signal)):
                # Example features: the value at the current index, the previous and next values
                # We handle edge cases when the index is at the start or end of the signal
                prev_val = signal[i - 1] if i > 0 else 0
                next_val = signal[i + 1] if i < len(signal) - 1 else 0
                features.append([signal[i], prev_val, next_val])
                # Label is 1 if it's a peak, 0 otherwise
                labels.append(1 if i in peaks else 0)
        
        return np.array(features), np.array(labels)

    def generate_random_training_data(self,data_set_size = 100,**kwargs):
        self.training_data = [generate_random_data(**kwargs) for _ in range(data_set_size)]

    def load_training_data(self,data_set):
        self.training_data = data_set

    def train_peak_detection_model(self):
        # Prepare features and labels
        X, y = self.create_features_and_labels()
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        
        # Initialize the model (Random Forest) with warm_start=True for incremental training
        model = RandomForestClassifier(n_estimators=self.step, warm_start=True, random_state=self.random_state)
        
        max_estimators = self.max_estimators  # Total number of trees
        step = self.step        # Number of trees to add in each iteration
        training_accuracies = []
        testing_accuracies = []
        estimator_range = range(step, max_estimators + step, step)
        
        for n_estimators in estimator_range:
            model.n_estimators = n_estimators
            model.fit(X_train, y_train)
            
            # Predict on training and testing sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate accuracies
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            training_accuracies.append(train_acc)
            testing_accuracies.append(test_acc)
            
            print(f'Iteration: {n_estimators}, Training Accuracy: {train_acc:.4f}, Testing Accuracy: {test_acc:.4f}')
        
        self.training_progress = training_accuracies
        self.testing_progress = testing_accuracies
        self.model = model

    def plot_progress(self):
        # Plot training progress
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_progress, label='Training Accuracy')
        plt.plot(self.testing_progress, label='Testing Accuracy')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Training Progress Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def predict_peak_probabilities(self, signal):
        # Generate features for prediction
        features = []
        for i in range(len(signal)):
            prev_val = signal[i - 1] if i > 0 else 0
            next_val = signal[i + 1] if i < len(signal) - 1 else 0
            features.append([signal[i], prev_val, next_val])
        
        # Predict probability of being a peak for each index in the signal
        self.peak_probabilities = self.model.predict_proba(features)[:, 1]

    def calculate_peaks_by_probability_threshold(self,threshold=0.5):
        # We use a threshold to determine if a peak is detected
        # There should be a better way of determining individual peaks
        self.detected_peaks = np.where(self.peak_probabilities > threshold)[0]