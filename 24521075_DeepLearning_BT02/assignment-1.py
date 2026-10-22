from torch.utils.data import DataLoader
import torch
from torch import nn
from utils import mnist_dataset, tools
from models.LeNet import LeNet

if __name__ == "__main__":
    train_dataset = mnist_dataset.MnistDataset(
        image_path="./datasets/MNIST/raw/train-images-idx3-ubyte",
        label_path="./datasets/MNIST/raw/train-labels-idx1-ubyte"
    )

    test_dataset = mnist_dataset.MnistDataset(
        image_path="./datasets/MNIST/raw/t10k-images-idx3-ubyte",
        label_path="./datasets/MNIST/raw/t10k-labels-idx1-ubyte"
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=True
    )

    net1 = LeNet()
    optimizer = torch.optim.Adam(net1.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    EPOCHS = 10

    tools.train(net1, optimizer, loss_fn, train_dataloader, test_dataloader, 
                EPOCHS, save_dir=f"checkpoints/assignment-1/")

    # Evaluation
    net1 = LeNet()
    net1.load_state_dict(torch.load('checkpoints/assignment-1/LeNet.pth', weights_only=True))
    evaluation = tools.eval(net1, test_dataloader)
    print(evaluation)