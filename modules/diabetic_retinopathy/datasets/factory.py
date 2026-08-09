import pandas as pd
from shared.datasets.split import stratified_split

from shared.preprocessing.statistics import get_mean_std_rgb
from modules.diabetic_retinopathy.datasets.dataset import (
    RetinopathyDataset
)
from modules.diabetic_retinopathy.transforms.transforms import (
    get_train_transforms,
    get_val_transforms
)


def create_datasets(
    csv_path: str,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 10
) -> tuple[RetinopathyDataset, RetinopathyDataset, RetinopathyDataset]:

    df = pd.read_csv(csv_path)

    train_df, val_df, test_df = stratified_split(
        df=df,
        label_column="diagnosis",
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )

    # Calculate statistics only on training data
    mean, std = get_mean_std_rgb(train_df)
    
    train_dataset = RetinopathyDataset(
        dataframe=train_df,
        transform=get_train_transforms(mean, std)
    )

    val_dataset = RetinopathyDataset(
        dataframe=val_df,
        transform=get_val_transforms(mean, std)
    )

    test_dataset = RetinopathyDataset(
        dataframe=test_df,
        transform=get_val_transforms(mean, std)
    )

    return train_dataset, val_dataset, test_dataset