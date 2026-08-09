from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]

class RetinopathyDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = PROJECT_ROOT / row["file_path"]
        image = Image.open(image_path).convert("RGB")

        label = row["diagnosis"]

        if self.transform:
            image = self.transform(image)

        return image, label