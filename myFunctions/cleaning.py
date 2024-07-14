import pandas as pd
import numpy as np
from typing import Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

def handle_missing_values(df: pd.DataFrame, 
                          strategy: str = 'mean', 
                          fill_value: Any = None
) -> pd.DataFrame:
    """
    This function handles missing values in a pandas DataFrame based on a specified strategy.

    Parameters:
    df (pd.DataFrame): The input DataFrame with missing values.
    strategy (str): The strategy to handle missing values. It can be one of the following:
        - 'mean': Replace missing values with the mean of the respective column.
        - 'median': Replace missing values with the median of the respective column.
        - 'mode': Replace missing values with the mode of the respective column.
        - 'ffill': Replace missing values with the previous valid observation in the DataFrame.
        - 'bfill': Replace missing values with the next valid observation in the DataFrame.
        - 'constant': Replace missing values with a specified fill_value.
    fill_value (Any): The value to use when strategy is 'constant'. Default is None.

    Returns:
    pd.DataFrame: The DataFrame with missing values handled according to the specified strategy.

    Example:
    >>> import pandas as pd
    >>> data = {'A': [1, 2, 2, np.nan, 5],
    ...         'B': ['a', 'b', 'b', 'a', np.nan],
    ...         'C': [1.1, 2.2, 2.2, np.nan, 5.5],
    ...         'D': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01']}
    >>> df = pd.DataFrame(data)
    >>> df_filled = handle_missing_values(df, strategy='mean')
    >>> print(df_filled)
    """
    if strategy == 'mean':
        return df.fillna(df.mean())
    elif strategy == 'median':
        return df.fillna(df.median())
    elif strategy == 'mode':
        return df.fillna(df.mode().iloc[0])
    elif strategy == 'ffill':
        return df.fillna(method='ffill')
    elif strategy == 'bfill':
        return df.fillna(method='bfill')
    elif strategy == 'constant':
        return df.fillna(fill_value)
    else:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', 'mode', 'ffill', 'bfill', or 'constant'.")

# Function to encode categorical variables
def encode_categorical(df: pd.DataFrame, 
                       column: list[str], 
                       encoding: str = 'onehot'
) -> pd.DataFrame:
    """
    Encode categorical variables in a DataFrame.

    This function transforms categorical variables into numerical representations 
    using the specified encoding technique.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.

    column : list[str]
        A list of column names that contain categorical variables to encode.

    encoding : str, optional
        The encoding method to use. Options are:
        - 'onehot': Perform one-hot encoding (default).
        - 'label': Perform label encoding.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the specified categorical columns encoded.

    Raises:
    ------
    ValueError
        If an invalid encoding method is specified.
    """
    if encoding == 'onehot':
        return pd.get_dummies(df, columns=[column])
    elif encoding == 'label':
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        return df
    else:
        raise ValueError("Invalid encoding. Choose from 'onehot' or 'label'.")

def scale_data(df: pd.DataFrame, 
               columns: list[str], 
               scaler: str = 'standard'
) -> pd.DataFrame:
    """
    Scale or normalize the specified columns in a DataFrame.

    This function transforms the specified columns using the chosen scaling method 
    to ensure that the data is in a suitable range for modeling.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data to scale.

    columns : list[str]
        A list of column names to scale or normalize.

    scaler : str, optional
        The scaling method to use. Options are:
        - 'standard': Standardize features by removing the mean and scaling to unit variance (default).
        - 'minmax': Scale features to a given range (default 0 to 1).
        - 'robust': Use the median and interquartile range for scaling, which is robust to outliers.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with the specified columns scaled or normalized.

    Raises:
    ------
    ValueError
        If an invalid scaler method is specified.
    """
    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Invalid scaler. Choose from 'standard', 'minmax', or 'robust'.")
    
    df[columns] = scaler.fit_transform(df[columns])
    return df

def handle_outliers(df: pd.DataFrame, 
                    columns: list[str], 
                    method: str = 'zscore', 
                    threshold: float = 3.0
) -> pd.DataFrame:
    """
    Handle outliers in the specified columns of a DataFrame using the chosen method.

    This function removes rows containing outliers based on either the Z-score or 
    the Interquartile Range (IQR) method.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.

    columns : list[str]
        A list of column names in which to detect and handle outliers.

    method : str, optional
        The method to use for outlier detection. Options are:
        - 'zscore': Use Z-score to identify outliers (default).
        - 'iqr': Use Interquartile Range (IQR) to identify outliers.

    threshold : float, optional
        The threshold value for outlier detection when using the Z-score method (default is 3.0).

    Returns:
    -------
    pd.DataFrame
        A DataFrame with outliers removed from the specified columns.

    Raises:
    ------
    ValueError
        If an invalid method is specified.
    """
    if method == 'zscore':
        from scipy.stats import zscore
        z_scores = np.abs(zscore(df[columns]))
        return df[(z_scores < threshold).all(axis=1)]
    elif method == 'iqr':
        Q1 = df[columns].quantile(0.25)
        Q3 = df[columns].quantile(0.75)
        IQR = Q3 - Q1
        return df[~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    else:
        raise ValueError("Invalid method. Choose from 'zscore' or 'iqr'.")

def extract_datetime_features(df: pd.DataFrame, 
                              column: str) -> pd.DataFrame:
    """
    Extract various datetime features from a specified column in a DataFrame.

    This function converts the specified column to datetime format and creates
    new columns representing the year, month, day, hour, minute, and second.

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.

    column : str
        The name of the column from which to extract datetime features.

    Returns:
    -------
    pd.DataFrame
        A DataFrame with additional columns for year, month, day, hour, 
        minute, and second extracted from the specified datetime column.
    """
    df[column] = pd.to_datetime(df[column])
    df[column + '_year'] = df[column].dt.year
    df[column + '_month'] = df[column].dt.month
    df[column + '_day'] = df[column].dt.day
    df[column + '_hour'] = df[column].dt.hour
    df[column + '_minute'] = df[column].dt.minute
    df[column + '_second'] = df[column].dt.second
    return df