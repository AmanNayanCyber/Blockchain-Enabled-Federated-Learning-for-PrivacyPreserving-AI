import copy
import torch
import torch.nn as nn


class Server:
    def __init__(self, model, device="cpu"):
        self.global_model = copy.deepcopy(model).to(device)
        self.device = device

    def fedavg(self, client_states, client_sizes):
        """
        Weighted Federated Averaging.
        client_states: list of state_dicts
        client_sizes: list of number of samples per client
        """
        global_state = self.global_model.state_dict()
        total_samples = sum(client_sizes)

        new_state = {}

        for key in global_state.keys():
            weighted_sum = 0
            for client_state, size in zip(client_states, client_sizes):
                weighted_sum += client_state[key].float() * (size / total_samples)
            new_state[key] = weighted_sum

        self.global_model.load_state_dict(new_state)
        return self.global_model

    def evaluate(self, test_loader):
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = test_loss / total if total > 0 else 0
        accuracy = 100.0 * correct / total if total > 0 else 0

        return avg_loss, accuracy
