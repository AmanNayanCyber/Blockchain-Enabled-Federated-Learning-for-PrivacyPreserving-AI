import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Client:
    def __init__(self, client_id, dataset, device="cpu", lr=0.01, batch_size=32):
        self.client_id = client_id
        self.dataset = dataset
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self, global_model, epochs=1):
        model = copy.deepcopy(global_model)
        model.to(self.device)
        model.train()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        total_loss = 0.0
        total_samples = 0

        for _ in range(epochs):
            for images, labels in self.loader:
                images, labels = images.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                total_samples += images.size(0)

        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        return model.state_dict(), avg_loss
