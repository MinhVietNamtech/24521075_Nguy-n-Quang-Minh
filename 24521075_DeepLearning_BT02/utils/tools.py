from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
import torch
from pathlib import Path
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

def eval(net, data_loader, device="cpu"):
    net.to(device)
    is_training = net.training
    try: 
        net.eval()
        all_predictions, all_truth = [], []
        with torch.no_grad():
            for (x, y) in tqdm(data_loader, desc=f"Evaluating in {len(data_loader.dataset)} samples", unit="batch"):
                x = x.to(device)
                y = y.to(device)
                y_hat = net(x)
                if type(y_hat) != torch.Tensor:
                    y_hat = y_hat.logits

                if y.ndim > 1 and y.shape == y_hat.shape:
                    y = y.argmax(dim=1)
                elif y.ndim > 1 and y.shape[-1] == 1:
                    y = y.squeeze(-1)

                y_hat = y_hat.argmax(dim=1)
                all_predictions.extend(y_hat.cpu().numpy())
                all_truth.extend(y.cpu().numpy())
        return {
                "accuracy": accuracy_score(all_truth, all_predictions),
                "precision": precision_score(all_truth, all_predictions, average="macro"),
                "recall": recall_score(all_truth, all_predictions, average="macro"),
                "f1": f1_score(all_truth, all_predictions, average="macro")
            }
    finally: 
        net.train(is_training)

def train(net, optimizer, loss_fn, train_loader, test_loader, epochs=3, device="cpu", need_checkpoint=True, save_dir=r"checkpoints"):
    net.to(device)
    best_score = -1
    print("train: ", len(train_loader))
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        for (x,y) in tqdm(train_loader, desc=f"Training at epoch {epoch+1}/{epochs}", unit='batch'):
            x,y = x.to(device), y.to(device)
            y_hat = net(x)
            if type(y_hat) != torch.Tensor:
                y_hat = y_hat.logits
            loss = loss_fn(y_hat, y)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"Epoch {epoch+1}| Loss: {epoch_loss/len(train_loader):.4f}")
        print('='*20)
        scores = eval(net, test_loader, device)
        current_score = scores["f1"]
        if current_score > best_score:
            best_score = current_score
            if need_checkpoint:
                save_dir = Path.cwd() / save_dir
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"{type(net).__name__}.pth"
                torch.save(net.state_dict(), checkpoint_path)

    print("Finish training!")

