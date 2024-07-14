import pandas as pd
import numpy as np

def get_feature_importance(x_train: pd.DataFrame,
                           model: object,
                           top_n: str = 5
) -> pd.DataFrame:
    """
    This function calculates and returns the feature importances of a trained machine learning model.

    Parameters:
    x_train (pd.DataFrame): Input training data as a pandas DataFrame.
    model (object): A trained machine learning model with a 'feature_importances_' attribute.
    top_n (str, optional): Number of top features to return. Defaults to 5.

    Returns:
    pd.DataFrame: A pandas DataFrame containing the top feature importances, sorted in descending order.

    Example usage:

    >>> import pandas as pd
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>>
    >>> # Create a sample training dataset
    >>> x_train = pd.DataFrame({
    ...     'feature1': np.random.randn(100),
    ...     'feature2': np.random.randn(100),
    ...     'feature3': np.random.randn(100),
    ...     'target': np.random.choice(['class1', 'class2'], 100)
    ... })
    >>>
    >>> # Train a RandomForestClassifier model
    >>> model = RandomForestClassifier(n_estimators=100, random_state=42)
    >>> model.fit(x_train.drop('target', axis=1), x_train['target'])
    >>>
    >>> # Get the top 5 feature importances
    >>> importance_df = get_feature_importance(x_train, model)
    >>> print(importance_df)
    """
    feature_importances = model.feature_importances_

    feature_list = []
    for feature_name, importance in zip(x_train.columns, feature_importances):
        feature_list.append([feature_name, importance])

    importance_df = pd.DataFrame(feature_list, columns=['feature_name', 'importance'])
    importance_df = importance_df.sort_values(by=['importance'], ascending=False).head(top_n)

    return importance_df

def check_imbalance(df: pd.DataFrame, class_column: str) -> pd.DataFrame:
    """
    Check the distribution of classes in a binary classification column and return as a DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - class_column (str): The name of the column containing the class labels.

    Returns:
    - pd.DataFrame: A DataFrame containing counts and ratios of each class.
    """
    class_counts = df[class_column].value_counts()
    total = class_counts.sum()

    imbalance_df = pd.DataFrame({
        'Class': class_counts.index,
        'Count': class_counts.values,
        'Ratio': class_counts.values / total
    })

    imbalance_df['Percentage'] = imbalance_df['Ratio'] * 100
    return imbalance_df
