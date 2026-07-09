import torch
import torch.nn as nn
import torch.optim as optim
from typing import List

class AutoencoderNet(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 2):
        super(AutoencoderNet, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Normalize inputs to [0, 1] range for reconstruction
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class PyTorchAnomalyDetector:
    def __init__(self, threshold: float = 0.05):
        self.model = AutoencoderNet(input_dim=3, hidden_dim=2)
        self.threshold = threshold
        self._train_baseline_model()

    def _normalize(self, cpu_pct: float, mem_mb: float, files_written: int) -> List[float]:
        # Normalize features to [0, 1] bounds
        norm_cpu = min(cpu_pct / 100.0, 1.0)
        norm_mem = min(mem_mb / 100.0, 1.0)  # Max memory expected is 100MB
        norm_files = min(files_written / 10.0, 1.0)  # Max files expected is 10
        return [norm_cpu, norm_mem, norm_files]

    def _train_baseline_model(self):
        """
        Trains the autoencoder on typical, non-malicious resource metrics (low CPU, low memory).
        """
        normal_metrics = [
            [0.05, 0.10, 0.0],  # 5% CPU, 10MB RAM, 0 files
            [0.10, 0.15, 0.1],  # 10% CPU, 15MB RAM, 1 file
            [0.02, 0.08, 0.0],  # 2% CPU, 8MB RAM, 0 files
            [0.15, 0.20, 0.2],  # 15% CPU, 20MB RAM, 2 files
            [0.08, 0.12, 0.1]   # 8% CPU, 12MB RAM, 1 file
        ]
        
        data_tensor = torch.tensor(normal_metrics, dtype=torch.float32)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        self.model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            outputs = self.model(data_tensor)
            loss = criterion(outputs, data_tensor)
            loss.backward()
            optimizer.step()

    def is_anomalous(self, cpu_pct: float, mem_mb: float, files_written: int) -> Tuple[bool, float]:
        """
        Calculates reconstruction loss. Returns (is_anomalous, reconstruction_error).
        """
        self.model.eval()
        normalized = self._normalize(cpu_pct, mem_mb, files_written)
        input_tensor = torch.tensor([normalized], dtype=torch.float32)
        
        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            # Calculate Mean Squared Error reconstruction loss
            loss = nn.functional.mse_loss(output_tensor, input_tensor).item()
            
        return loss > self.threshold, loss
