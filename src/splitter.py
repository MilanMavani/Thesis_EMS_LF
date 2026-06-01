import pandas as pd


def time_based_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    """
    Chronological split by row percentage.
    """
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    return train_df, val_df, test_df


def date_based_split(
    df: pd.DataFrame,
    *,
    train_end: str,
    val_end: str,
):
    """
    Chronological split by explicit dates.

    Split logic:
        train:      index <= train_end
        validation: train_end < index <= val_end
        test:       index > val_end
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")

    out = df.sort_index().copy()

    train_end = pd.to_datetime(train_end)
    val_end = pd.to_datetime(val_end)

    if train_end >= val_end:
        raise ValueError("train_end must be earlier than val_end")

    train_df = out.loc[out.index <= train_end].copy()
    val_df = out.loc[(out.index > train_end) & (out.index <= val_end)].copy()
    test_df = out.loc[out.index > val_end].copy()

    if train_df.empty:
        raise ValueError("Training set is empty. Check train_end.")
    if val_df.empty:
        raise ValueError("Validation set is empty. Check train_end and val_end.")
    if test_df.empty:
        raise ValueError("Test set is empty. Check val_end.")

    return train_df, val_df, test_df