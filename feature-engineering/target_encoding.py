from category_encoders import TargetEncoder
import pandas as pd


def discretize_feature(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Discretizes a numerical feature in the DataFrame by rounding it to the nearest integer.

    Args:
        df (pd.DataFrame): The input DataFrame containing the feature to be discretized.
        feature (str): The name of the feature to be discretized.

    Returns:
        pd.DataFrame: The DataFrame with the specified feature discretized to integers.
    """

    df[feature] = df[feature].apply(round)
    return df


def target_encode(
    df: pd.DataFrame, features_to_encode: list[str], target_col: str
) -> pd.DataFrame:
    """
    Encodes categorical features in the DataFrame using target encoding based on the specified target column.

    Args:
        df (pd.DataFrame): The input DataFrame containing the features to be encoded.
        features_to_encode (list(str)): List of feature names to be target encoded.
        target_col (str): The name of the target column used for encoding.

    Returns:
        pd.DataFrame: The DataFrame with specified features target encoded.
    """

    encoder = TargetEncoder()
    df[features_to_encode] = encoder.fit_transform(df[features_to_encode], df[target_col])
    return df
