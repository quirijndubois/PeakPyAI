import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from .utils import plot_data, mapRange, normal_distribution
import matplotlib.pyplot as plt

class GaussianDataGenerator:
    @staticmethod
    def generate_random_data(resolution=256, peaks_range=[3, 3], peak_height_range=[1, 1],
                              peak_position_range=[32, 256-32], std_range=[10, 10], noise_strength=0.001,
                              peak_distance_treshold=30):
        n_peaks = int(mapRange(np.random.random(), 0, 1, peaks_range[0], peaks_range[1]))
        if peak_position_range[1] is None:
            peak_position_range[1] = resolution

        signal = np.zeros(resolution)
        peaks = []
        heights = []

        x_peaks = []
        for _ in range(n_peaks):
            x_peak = mapRange(np.random.random(), 0, 1, peak_position_range[0], peak_position_range[1])
            while True:
                stop = False
                for x in x_peaks:
                    if abs(x_peak - x) < peak_distance_treshold:
                        stop = True
                if not stop:
                    break
                x_peak = mapRange(np.random.random(), 0, 1, peak_position_range[0], peak_position_range[1])
            x_peaks.append(x_peak)

        x_peaks.sort()

        for x_peak in x_peaks:
            y_peak = mapRange(np.random.random(), 0, 1, peak_height_range[0], peak_height_range[1])
            std = mapRange(np.random.random(), 0, 1, std_range[0], std_range[1])
            peaks.append(round(x_peak))
            signal += normal_distribution(np.arange(resolution), x_peak, std) * y_peak
            heights.append(y_peak)
        signal += (np.random.randn(resolution) * 2 - 1) * noise_strength

        return [signal, peaks, heights]

class GaussianPeakDataset(Dataset):
    def __init__(self, n_samples=1000):
        self.data = [GaussianDataGenerator.generate_random_data() for _ in range(n_samples)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        signal, peaks, heights = self.data[idx]
        target = np.zeros(3)
        for i, (peak, height) in enumerate(zip(peaks, heights)):
            if i < 3:
                target[i] = peak / 256
        return torch.tensor(signal, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)

class GaussianPeakDetector(nn.Module):
    def __init__(self):
        super(GaussianPeakDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 peaks (locations only)
        )

    def forward(self, x):
        x = self.fc(x)
        return x.view(-1, 3)

class GaussianTrainingPipeline:
    def __init__(self, model, dataset, batch_size=1, lr=0.0001):
        self.model = model
        self.dataset = dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.losses = []

    def train(self, epochs=200):
        for epoch in range(epochs):
            for signals, targets in self.dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(signals)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
            self.losses.append(loss.item())

    def plot_losses(self):
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs')
        plt.yscale('log')
        plt.show()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

class GaussianTesting:
    @staticmethod
    def test_model(model, test_signal):
        model.eval()
        with torch.no_grad():
            test_signal_tensor = torch.tensor(test_signal, dtype=torch.float32).unsqueeze(0)
            prediction = model(test_signal_tensor).squeeze(0).numpy()
        return prediction * 256

    @staticmethod
    def visualize_results(test_signal, actual_peaks, predicted_peaks):
        print("Predicted Peaks:", predicted_peaks)
        print("Actual Peaks:", actual_peaks)
        plot_data(signal=test_signal, actual_peaks=actual_peaks, detected_peaks=predicted_peaks)

if __name__ == "__main__":
    # Training
    dataset = GaussianPeakDataset(50)
    model = GaussianPeakDetector()
    pipeline = GaussianTrainingPipeline(model, dataset)

    pipeline.train(epochs=200)
    pipeline.plot_losses()
    pipeline.save_model('PeakPyAI/trained_gaussian_model.pt')

    # Testing
    test_signal, test_peaks, test_heights = GaussianDataGenerator.generate_random_data()
    predicted_peaks = GaussianTesting.test_model(model, test_signal)

    # Visualization
    GaussianTesting.visualize_results(test_signal, test_peaks, predicted_peaks)
