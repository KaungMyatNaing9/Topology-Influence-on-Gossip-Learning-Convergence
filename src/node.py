import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class Node:
    def __init__(self, node_id, model, dataset, neighbors, device='cpu'):
        self.id = node_id
        self.model = model
        self.device = device
        self.model.to(device)

        self.dataset = dataset
        num_workers = 0
        self.dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=(device == 'cuda'),
        )

        self.neighbors = neighbors

        self.bytes_sent = 0
        self.bytes_received = 0
        self.train_losses = []
        self.train_accuracies = []

    def local_train(self, epochs=1, lr=0.01):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        epoch_loss = 0.0
        correct = 0
        total = 0

        for _ in range(epochs):
            for batch_data, batch_labels in self.dataloader:
                batch_data, batch_labels = batch_data.to(
                    self.device), batch_labels.to(self.device)

                outputs = self.model(batch_data)
                loss = criterion(outputs, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

        avg_loss = epoch_loss / max(1, (len(self.dataloader) * epochs))
        accuracy = 100 * correct / total if total > 0 else 0.0

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(accuracy)

        return avg_loss, accuracy

    def get_model_params(self):
        params = [p.data.view(-1) for p in self.model.parameters()]
        return torch.cat(params)

    def set_model_params(self, flat_params):
        start = 0
        for param in self.model.parameters():
            numel = param.numel()
            param.data = flat_params[start:start +
                                     numel].view_as(param.data).clone()
            start += numel

    def evaluate(self, test_dataset=None):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        if test_dataset is None:
            dataloader = self.dataloader
        else:
            dataloader = DataLoader(
                test_dataset, batch_size=256, shuffle=False, num_workers=0)

        total = 0
        correct = 0
        total_loss = 0.0

        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / max(1, len(dataloader))
        accuracy = 100 * correct / total if total > 0 else 0.0

        return avg_loss, accuracy
