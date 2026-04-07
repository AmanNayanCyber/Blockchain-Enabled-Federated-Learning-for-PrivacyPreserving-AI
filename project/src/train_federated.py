import os
import csv
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

from model import CNNMnist
from data import get_mnist_data, split_dataset, get_test_loader
from client import Client
from server import Server
from blockchain import BlockchainLedger
from privacy import add_gaussian_noise, clip_state_dict


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def poison_update(state_dict, scale=0.2):
    poisoned = {}
    for k, v in state_dict.items():
        if torch.is_tensor(v):
            poisoned[k] = v + torch.randn_like(v) * scale
        else:
            poisoned[k] = v
    return poisoned


def main():
    set_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_clients = 7
    rounds = 5
    local_epochs = 1
    batch_size = 32
    lr = 0.01
    noise_scale = 0.01
    malicious_client_id = 0

    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    os.makedirs("outputs/blockchain", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    train_dataset, test_dataset = get_mnist_data()
    client_datasets = split_dataset(train_dataset, num_clients=num_clients)
    test_loader = get_test_loader(test_dataset, batch_size=64)

    global_model = CNNMnist().to(device)
    server = Server(global_model, device=device)
    ledger = BlockchainLedger()

    clients = [
        Client(client_id=i, dataset=client_datasets[i], device=device, lr=lr, batch_size=batch_size)
        for i in range(num_clients)
    ]

    round_losses = []
    round_accuracies = []

    metrics_path = "outputs/metrics/training_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "test_loss", "accuracy", "chain_valid"])

        for round_num in range(1, rounds + 1):
            print(f"\n--- Round {round_num} ---")

            client_states = []
            client_losses = []
            client_sizes = []

            for client in clients:
                state_dict, loss, size = client.train(server.global_model, epochs=local_epochs)

                state_dict = clip_state_dict(state_dict, max_norm=5.0)
                state_dict = add_gaussian_noise(state_dict, noise_scale=noise_scale)

                # Simulate one malicious client
                if client.client_id == malicious_client_id and round_num >= 2:
                    print(f"Client {client.client_id} is malicious this round.")
                    state_dict = poison_update(state_dict, scale=0.3)

                ledger.add_block(client_id=client.client_id, round_num=round_num, update_state_dict=state_dict)

                client_states.append(state_dict)
                client_losses.append(loss)
                client_sizes.append(size)

                print(f"Client {client.client_id} loss: {loss:.4f}")

            server.fedavg(client_states, client_sizes)

            test_loss, accuracy = server.evaluate(test_loader)

            avg_client_loss = sum(client_losses) / len(client_losses)
            round_losses.append(test_loss)
            round_accuracies.append(accuracy)

            chain_valid = ledger.is_chain_valid()

            print(f"Average client loss: {avg_client_loss:.4f}")
            print(f"Test loss: {test_loss:.4f}")
            print(f"Test accuracy: {accuracy:.2f}%")
            print(f"Blockchain valid: {chain_valid}")

            writer.writerow([round_num, test_loss, accuracy, chain_valid])

            torch.save(
                server.global_model.state_dict(),
                f"outputs/checkpoints/global_model_round_{round_num}.pth"
            )

    ledger.save_chain("outputs/blockchain/ledger.json")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rounds + 1), round_accuracies, marker="o", label="Test Accuracy")
    plt.plot(range(1, rounds + 1), round_losses, marker="s", label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("Federated Learning Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/plots/training_progress.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nBlockchain saved to outputs/blockchain/ledger.json")
    print("Metrics saved to outputs/metrics/training_metrics.csv")
    print("Training complete.")


if __name__ == "__main__":
    main()
