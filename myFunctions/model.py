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