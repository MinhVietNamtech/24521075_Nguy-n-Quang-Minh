from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn
from utils import vinafood21_dataset, tools
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch

if __name__=="__main__":
    # Train set
    vinaTrain_data = vinafood21_dataset.VinaFoodDataset(is_train=True)
    vinaTrain_loader = DataLoader(vinaTrain_data, batch_size=32, shuffle=True)

    # Test set
    vinaTest_data = vinafood21_dataset.VinaFoodDataset(is_train=False)
    vinaTest_loader = DataLoader(vinaTest_data, batch_size=32, shuffle=True)

    device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else "cpu"
    print(f"Current used device: {device}")

    # Pretrained model loading
    print("Loading pre-trained model...")
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    net4 = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
    net4.classifier = nn.Sequential(
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=2048, out_features=len(vinaTrain_data.label2idx), bias=True)
    )

    optimizer = torch.optim.Adam(net4.parameters(),lr=0.01)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 3

    # Fine-tuning
    tools.train(net4,optimizer,loss_fn,vinaTrain_loader,vinaTest_loader,epochs=epochs,
                    device=device, save_dir=r"checkpoints/assignment-4/")

    # Evaluation
    net4.load_state_dict(torch.load('checkpoints/assignment-4/ResNetForImageClassification.pth', weights_only=True))
    evaluation = tools.eval(net4, vinaTest_loader, device=device)
    print(evaluation)

