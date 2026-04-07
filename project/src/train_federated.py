import os
import copy
import torch
import matplotlib.pyplot as plt

from model import CNNMnist
from data import get_mnist_data, split_dataset, get_test_loader
from client import Client
from server import Server
from blockchain import BlockchainLedger
from privacy import add_gaussian_noise, clip_state_dict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_clients = 5
    rounds = 5
    local_epochs = 1
    batch_size = 32
    lr = 0.01
    noise_scale = 0.01

    os.makedirs("outputs/plots", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

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

    for round_num in range(1, rounds + 1):
        print(f"\n--- Round {round_num} ---")

        client_states = []
        client_losses = []

        for client in clients:
            state_dict, loss = client.train(server.global_model, epochs=local_epochs)

            # Privacy protection
            state_dict = clip_state_dict(state_dict, max_norm=5.0)
            state_dict = add_gaussian_noise(state_dict, noise_scale=noise_scale)

            # Log to blockchain ledger
            ledger.add_block(client_id=client.client_id, round_num=round_num, update_state_dict=state_dict)

            client_states.append(state_dict)
            client_losses.append(loss)

            print(f"Client {client.client_id} loss: {loss:.4f}")

        # Aggregate client updates
        server.fedavg(client_states)

        # Evaluate global model
        test_loss, accuracy = server.evaluate(test_loader)

        avg_client_loss = sum(client_losses) / len(client_losses)
        round_losses.append(test_loss)
        round_accuracies.append(accuracy)

        print(f"Average client loss: {avg_client_loss:.4f}")
        print(f"Test loss: {test_loss:.4f}")
        print(f"Test accuracy: {accuracy:.2f}%")

        # Save checkpoint
        torch.save(server.global_model.state_dict(), f"outputs/checkpoints/global_model_round_{round_num}.pth")

    # Save metrics plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, rounds + 1), round_accuracies, marker="o", label="Test Accuracy")
    plt.plot(range(1, rounds + 1), round_losses, marker="s", label="Test Loss")
    plt.xlabel("Round")
    plt.ylabel("Value")
    plt.title("Federated Learning Training Progress")
    plt.legend()
    plt.grid(True)
    plt.savefig("outputs/plots/training_progress.png")
    plt.show()

    print("\nBlockchain valid:", ledger.is_chain_valid())
    print("Training complete.")


if __name__ == "__main__":
    main()
