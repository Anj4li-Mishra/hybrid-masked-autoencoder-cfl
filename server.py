
"""
Federated Learning Server for Clustered Malicious Network Detection
"""

import os
import torch
import flwr as fl
import numpy as np
from autoencoder import MaskedDenoisingAutoencoder
from flwr.common import parameters_to_ndarrays

class SaveFinalModelStrategy(fl.server.strategy.FedAvg):
    """Custom strategy that saves the final aggregated model"""
    
    def __init__(self, protocol, input_dim):
        super().__init__(
            min_fit_clients=4,
            min_available_clients=4,
            on_fit_config_fn=None,
            fit_metrics_aggregation_fn=lambda metrics: {}
        )
        self.protocol = protocol
        self.input_dim = input_dim

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client models and save final model"""
        print(f"Aggregating {len(results)} client(s) weights at round {server_round}")
        aggregated = super().aggregate_fit(server_round, results, failures)
        
        if aggregated is not None:
            # Convert aggregated parameters to model
            weights = parameters_to_ndarrays(aggregated[0])
            model = MaskedDenoisingAutoencoder(self.input_dim)
            state_dict = model.state_dict()
            
            for k, v in zip(state_dict.keys(), weights):
                state_dict[k] = torch.tensor(v)
            model.load_state_dict(state_dict)
            
            # Save model to disk
            os.makedirs("cluster_models", exist_ok=True)
            torch.save(model.state_dict(), f"cluster_models/cluster_{self.protocol}.pth")
            print(f"Saved global model for cluster: {self.protocol}")
        
        return aggregated

if __name__ == "__main__":
    """Usage: python server_manual.py <protocol>"""
    import sys
    if len(sys.argv) != 2:
        print("Usage: python server_manual.py <protocol>")
        exit(1)

    protocol = sys.argv[1]

    # Find client directories for this protocol
    client_dirs = [d for d in os.listdir("preprocessed") if d.startswith("client") and d.endswith(protocol)]
    if not client_dirs:
        print(f"No client directories found for protocol '{protocol}'")
        exit(1)

    # Determine input dimension from example client
    example_client_dir = os.path.join("preprocessed", client_dirs[0])
    data_path = os.path.join(example_client_dir, "data.npy")
    if not os.path.exists(data_path):
        print(f"No data found at {data_path}. Cannot infer input dim.")
        exit(1)

    input_dim = torch.tensor(np.load(data_path)).shape[1]
    strategy = SaveFinalModelStrategy(protocol=protocol, input_dim=input_dim)

    print(f"Starting FLWR server for cluster: {protocol}")
    print(f"FL Server ready: waiting for clients on protocol '{protocol}'...")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy
    )
