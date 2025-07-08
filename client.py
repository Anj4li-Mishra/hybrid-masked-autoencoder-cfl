"""
Federated Learning Client for Malicious Network Detection using MDAE
"""

import os
import sys
import numpy as np
import torch
import flwr as fl
from torch.utils.data import DataLoader, TensorDataset
from autoencoder import MaskedDenoisingAutoencoder, masked_mse_loss


class MDAEClient(fl.client.NumPyClient):
    """Federated Learning Client with Masked Denoising Autoencoder"""
    
    def __init__(self, client_dir):
        print(f"Starting client at: {client_dir}")
        self.client_dir = client_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load client data and masks
        self.data = torch.tensor(np.load(os.path.join(client_dir, "data.npy")), dtype=torch.float32)
        self.mask = torch.tensor(np.load(os.path.join(client_dir, "mask.npy")), dtype=torch.float32)
        
        # Setup dataset and dataloader
        self.dataset = TensorDataset(self.data, self.mask)
        self.trainloader = DataLoader(self.dataset, batch_size=32, shuffle=True)

        # Initialize model and optimizer
        self.input_dim = self.data.shape[1]
        self.model = MaskedDenoisingAutoencoder(self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def get_parameters(self, config=None):
        """Send model parameters to server"""
        print(f"[{self.client_dir}] Sending parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Update model with parameters from server"""
        print(f"[{self.client_dir}] Receiving new parameters")
        state_dict = self.model.state_dict()
        for k, v in zip(state_dict.keys(), parameters):
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        """Local training for 5 epochs"""
        print(f"[{self.client_dir}] fit() called")
        self.set_parameters(parameters)
        self.model.train()
        
        for epoch in range(5):
            print(f"[{self.client_dir}] Training epoch {epoch+1}/5")
            total_loss = 0.0
            
            for i, (x, m) in enumerate(self.trainloader):
                x, m = x.to(self.device), m.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(x, m)
                loss = masked_mse_loss(output, x, m)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                print(f"[{self.client_dir}]   Batch {i+1}: loss = {loss.item():.4f}")
            
            print(f"[{self.client_dir}] Epoch {epoch+1} total loss = {total_loss:.4f}")
        
        print(f"[{self.client_dir}] Training complete for this round.")
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate model on local data"""
        print(f"[{self.client_dir}] evaluate() called")
        self.set_parameters(parameters)
        self.model.eval()
        
        with torch.no_grad():
            output = self.model(self.data.to(self.device), self.mask.to(self.device))
            loss = masked_mse_loss(output, self.data.to(self.device), self.mask.to(self.device))
        
        print(f"[{self.client_dir}] Evaluation loss: {loss.item():.4f}")
        return float(loss.item()), len(self.dataset), {}


if __name__ == "__main__":
    """Usage: python client_manual.py <client_dir>"""
    if len(sys.argv) != 2:
        print("Usage: python client_manual.py <client_dir>")
        exit(1)
    
    client_dir = sys.argv[1]
    fl.client.start_numpy_client(
        server_address="localhost:8080", 
        client=MDAEClient(client_dir)
    )
