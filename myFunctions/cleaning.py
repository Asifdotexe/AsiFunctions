import pandas as pd
import numpy as np
from typing import Any
import scipy.stats as stats
from myFunctions.model import detect_distribution
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function standardizes the column names of a pandas DataFrame by converting them to lowercase and replacing spaces with underscores.

    Parameters:
    df (pd.DataFrame): The input pandas DataFrame with column names to be standardized.

    Returns:
    pd.DataFrame: The modified pandas DataFrame with standardized column names.

    Example usage:
    >>> # Standardize the column names
    >>> df_standardized = standardize_column_names(df)

    >>> # Print the standardized DataFrame
    >>> print(df_standardized)
    """
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df 

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
                       columns: list[str] = None, 
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

    columns : list[str], optional
        A list of column names that contain categorical variables to encode. 
        If None, columns with object data types will be selected, excluding those with more than 5 unique values.

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
    if columns is None:
        columns = df.select_dtypes(include=['object']).columns.tolist()
        columns = [col for col in columns if df[col].nunique() <= 5]

    if encoding == 'onehot':
        return pd.get_dummies(df, columns=columns, drop_first=True)
    elif encoding == 'label':
        le = LabelEncoder()
        for column in columns:
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
    
    for column in columns:
        df[f'scaled_{column}'] = scaler.fit_transform(df[[column]])
    return df

def remove_outliers(df: pd.DataFrame,
                    columns: list[str],
                    zscore_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Removes outliers from the given pandas DataFrame columns based on their distribution.
    
    Parameters:
    df (pd.DataFrame): Input DataFrame with numerical columns to clean.
        The DataFrame should contain numerical columns specified in the 'columns' parameter.
    
    columns (list[str]): List of column names to consider for outlier removal.
        If None, all numeric columns in the DataFrame will be used.
    
    zscore_threshold (float): Z-score threshold for identifying outliers.
        Default is 3.0, which means data points with a Z-score greater than 3.0 will be considered outliers.
    
    Returns:
    pd.DataFrame: DataFrame with outliers removed.
        The returned DataFrame will have the same columns as the input DataFrame, but without the outliers.
    
    Examples:
    >>> remove_outliers(df)
    """
    if columns is None:
        columns = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
    
    distribution_results = detect_distribution(df[columns])
    clean_data = df.copy()
    
    for column in columns:
        if column in distribution_results:
            col_data = df[column].dropna()
            distribution = distribution_results[column]
            
            if distribution == "Normally distributed":
                # Z-score method
                z_scores = np.abs(stats.zscore(col_data))
                clean_data = clean_data[z_scores < zscore_threshold]
            else:
                # IQR method
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                clean_data = clean_data[(col_data >= lower_bound) & (col_data <= upper_bound)]
    return clean_data

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