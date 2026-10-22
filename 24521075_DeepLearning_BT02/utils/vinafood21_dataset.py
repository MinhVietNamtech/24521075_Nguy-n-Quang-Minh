from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from torchvision import transforms

class VinaFoodDataset(Dataset):
    def __init__(self, path=r"datasets/VinaFood21/", is_train=True, transform=None):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ]) if transform is None else transform

        root = Path(path)
        data_path = root/"train" if is_train else root/"test"

        # Convert label to index with order
        labels = sorted([p.name for p in data_path.iterdir() if p.is_dir()])
        self.label2idx = {label: idx for idx,label in enumerate(labels)}

        # Read all samples
        self.samples = []
        for folder in data_path.iterdir():
            if not folder.is_dir(): continue
            for file in folder.iterdir():
                if file.is_file():
                    self.samples.append((file, self.label2idx[folder.name]))

    def __getitem__(self, index):
        file, label = self.samples[index]
        img = Image.open(file).convert("RGB")
        img = self.transform(img)
        return (img, label)

    def __len__(self):
        return len(self.samples)