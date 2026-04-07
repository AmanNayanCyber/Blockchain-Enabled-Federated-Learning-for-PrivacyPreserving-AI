import torch
from model import CNNMnist
from data import get_mnist_data, get_test_loader
from server import Server


def evaluate_saved_model(model_path="outputs/checkpoints/global_model_round_5.pth"):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, test_dataset = get_mnist_data()
    test_loader = get_test_loader(test_dataset, batch_size=64)

    model = CNNMnist().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    server = Server(model, device=device)
    test_loss, accuracy = server.evaluate(test_loader)

    print(f"Loaded model test loss: {test_loss:.4f}")
    print(f"Loaded model test accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    evaluate_saved_model()
