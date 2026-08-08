import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_split(
    df: pd.DataFrame,
    label_column: str,
    test_size: float = 0.2,
    val_size: float = 0.25,
    random_state: int = 10
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_column],
        random_state=random_state
    )

    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        stratify=train_df[label_column],
        random_state=random_state
    )

    return train_df, val_df, test_df


