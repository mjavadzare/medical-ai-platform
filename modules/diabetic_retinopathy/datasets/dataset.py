from torch.utils.data import Dataset
from PIL import Image


class RetinopathyDataset(Dataset):

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image = Image.open(
            row["file_path"]
        ).convert("RGB")

        label = row["diagnosis"]

        if self.transform:
            image = self.transform(image)

        return image, label