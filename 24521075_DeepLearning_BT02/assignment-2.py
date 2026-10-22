from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn

from models.GoogleNet import GoogleNet
from utils import vinafood21_dataset, tools

if __name__ == "__main__":

    # Train set
    vinaTrain_data = vinafood21_dataset.VinaFoodDataset(is_train=True)
    vinaTrain_loader = DataLoader(vinaTrain_data, batch_size=32, shuffle=True)

    # Test set
    vinaTest_data = vinafood21_dataset.VinaFoodDataset(is_train=False)
    vinaTest_loader = DataLoader(vinaTest_data, batch_size=32, shuffle=True)

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    print(f"Current used device: {device}")

    net2 = GoogleNet(in_channels=3, out_channels=len(vinaTrain_data.label2idx))
    optimizer = torch.optim.Adam(net2.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 3    

    tools.train(net2, optimizer, loss_fn, vinaTrain_loader, vinaTest_loader, 
                epochs=epochs, device=device, save_dir=r"checkpoints/assignment-2/")

    # Evaluation
    net2 = GoogleNet(in_channels=3, out_channels=len(vinaTrain_data.label2idx))
    net2.load_state_dict(torch.load('checkpoints/assignment-2/GoogleNet.pth', weights_only=True))
    evaluation = tools.eval(net2,vinaTest_loader, device=device)
    print(evaluation)