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
            for batch in tqdm(data_loader, desc=f"Evaluating in {len(data_loader.dataset)} samples", unit="batch"):
                x = batch["input_ids"].to(device)
                y = batch["label"].to(device)
                y_hat = net(x, batch["lengths"].to(device))
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

def train(net, optimizer, criterion, train_loader, dev_loader, n_epoch, device="cpu", save_path=None):
    net.to(device)
    net.train()
    best_score = -1
    for epoch in range(n_epoch):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Training at epoch {epoch+1}/{n_epoch}", unit="batch"):
            optimizer.zero_grad()
            X = batch["input_ids"].to(device)
            y = batch["label"].to(device)
            logits = net(X, batch["lengths"].to(device))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss/len(train_loader):.4f}")
        scores = eval(net, dev_loader, device)
        if scores["f1"] > best_score:
            best_score = scores["f1"]
            if save_path is not None:
                save_dir = Path.cwd() / save_path
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"{type(net).__name__}.pth"
                torch.save(net.state_dict(), checkpoint_path)

    print("Finish training!")


