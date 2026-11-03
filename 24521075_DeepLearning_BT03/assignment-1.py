import torch
from utils.uit_vsfc import Vocab, UIT_VSFC, collate_fn
from models.lstm import LSTM
from utils.tools import train, eval

if __name__ == "__main__":
    # Extract data
    vocab = Vocab('data/UIT-VSFC-train.json')
    train_set = UIT_VSFC('data/UIT-VSFC-train.json', vocab)
    dev_set = UIT_VSFC('data/UIT-VSFC-dev.json', vocab)
    test_set = UIT_VSFC('data/UIT-VSFC-test.json', vocab)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net1 = LSTM(
        vocab_size=vocab.len,
        hidden_size=256,
        n_layer=5,
        n_labels=vocab.n_labels,
        padding_idx=vocab.w2i[vocab.pad],
        dropout=0.2
    )
    optimizer = torch.optim.Adam(net1.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    n_epoch = 10
    print(f"Current used device: {device}")
    # Training model
    train(net1, optimizer, criterion, train_loader, dev_loader, n_epoch=n_epoch, device=device, save_path="checkpoints/assignment_1")

    # Evaluating model
    scores = eval(net1, test_loader, device=device)
    print(f"F1-score on test set: {scores["f1"]:.4f}")


    
