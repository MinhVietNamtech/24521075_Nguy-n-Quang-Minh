from utils.PhoNER import Vocab, PhoNER, collate_fn
import torch
from models.encoder import Encoder
from tqdm.auto import tqdm
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

g = torch.Generator()
g.manual_seed(42)

def eval(encoder, data_loader, device="cpu"):
    encoder.to(device)
    is_training = encoder.training
    try:
        encoder.eval()
        with torch.no_grad():
            all_predictions, all_truth = [], []
            for batch in tqdm(data_loader, desc=f"Evaluating in {len(data_loader.dataset)} samples"):
                X = batch["word_ids"].to(device)
                Y = batch["tags"].to(device)
                logits = encoder(X, batch['lengths'])
                labels = logits.argmax(dim=-1)

                mask = Y != -1
                labels = labels[mask] # Filter out padding value (-1)
                Y = Y[mask]
                
                all_predictions.extend(labels.cpu().tolist())
                all_truth.extend(Y.cpu().tolist())
        return {
            "accuracy": accuracy_score(all_truth, all_predictions),
            "precision": precision_score(all_truth, all_predictions, average="macro", zero_division=0),
            "recall": recall_score(all_truth, all_predictions, average="macro"),
            "f1": f1_score(all_truth, all_predictions, average="macro")
        }
    finally:
        encoder.train(is_training)

def train_epoch(encoder, train_loader, encoder_optimizer, criterion, epoch_id, device="cpu"):
    encoder.to(device)
    encoder.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Training at epoch {epoch_id}", unit="batch"):
        X = batch["word_ids"].to(device)
        Y = batch["tags"].to(device)
        logits = encoder(X, batch['lengths'])
        loss = criterion(logits.permute(0,2,1), Y)
        encoder_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        encoder_optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

def train(encoder, train_loader, dev_loader, encoder_optimizer, criterion, n_epoch=5, device="cpu", save_path=None):
    best_score = -1
    for epoch in range(n_epoch):
        epoch_loss = train_epoch(encoder, train_loader, encoder_optimizer, 
                                criterion, epoch_id=epoch+1, device="cpu")
        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss:.4f}")
        scores = eval(encoder, dev_loader)
        if scores["f1"] > best_score:
            best_score = scores['f1']
            if save_path is not None:
                save_dir = Path.cwd() / save_path
                save_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = save_dir / f"{type(encoder).__name__}.pth"
                torch.save(encoder.state_dict(), checkpoint_path)

    print("Finish training!")

if __name__ == "__main__":
    vocab = Vocab('data/PhoNER_COVID19/data/word/train_word.json')
    train_set = PhoNER('data/PhoNER_COVID19/data/word/train_word.json', vocab)
    dev_set = PhoNER('data/PhoNER_COVID19/data/word/dev_word.json', vocab)
    test_set = PhoNER('data/PhoNER_COVID19/data/word/test_word.json', vocab)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g
    )

    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        generator=g
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        generator=g
    )

    encoder = Encoder(vocab.len, 
                      hidden_size=256,
                      n_layer=5,
                      n_labels=vocab.n_labels,
                      padding_idx=0,
                      dropout=0.2)

    optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    n_epoch = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Currently used device: {device}")
    train(encoder, train_loader, dev_loader, optimizer, criterion,
          n_epoch, device=device, save_path="checkpoints/assignment_3")

    scores = eval(encoder, test_loader, device=device)
    print(f"F1-score on test set: {scores['f1']}")


        




    




