import pandas as pd
import numpy as np
from typing import Any
import scipy.stats as stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

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

def detect_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detects the distribution of the given pandas DataFrame columns.
    
    Parameters:
    data (pd.DataFrame): Input DataFrame with numerical columns to analyze.
    
    Returns:
    dict: A dictionary with the column names as keys and distribution descriptions as values.
    
    Examples:
    >>> detect_distribution(data)
    {'normal': 'Normally distributed', 'uniform': 'Uniformly distributed', 
    'right_skewed': 'Right skewed', 'left_skewed': 'Left skewed'}
    
    Notes:
    - Shapiro-Wilk Test: A statistical test for normality. A high p-value (>0.05) suggests normal distribution.
    - Anderson-Darling Test: Another test for normality with critical values to compare against the test statistic.
    - Skewness: Measures the asymmetry of the data distribution. Positive skew indicates right skew, and negative skew indicates left skew.
    """
    results = {}
    
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            col_data = df[column].dropna()
            
            # Shapiro-Wilk Test for Normality
            # Shapiro-Wilk Test: A statistical test for normality. 
            # A high p-value (>0.05) suggests normal distribution.
            shapiro_test = stats.shapiro(col_data)
            
            # Anderson-Darling Test for Normality
            # Anderson-Darling Test: Another test for normality with critical values 
            # to compare against the test statistic.
            ad_test = stats.anderson(col_data, dist='norm')
            
            # Skewness
            # Skewness: Measures the asymmetry of the data distribution. 
            # Positive skew indicates right skew, and negative skew indicates left skew.
            skewness = stats.skew(col_data)
            
            distribution = "Unknown distribution"
            
            if shapiro_test.pvalue > 0.05 and ad_test.statistic < ad_test.critical_values[2]:
                distribution = "Normally distributed"
            elif skewness > 0.5:
                distribution = "Right skewed"
            elif skewness < -0.5:
                distribution = "Left skewed"
            else:
                distribution = "Uniformly distributed"
            
            results[column] = distribution
    
    return results

def assess_performance(score: float, metric_type: str) -> str:
    """
    Determine the inference based on the score and metric type.

    Parameters
    ----------
    score : float
        The score to be evaluated.
    metric_type : str
        The type of metric used to evaluate the score.

    Returns
    -------
    str
        A string indicating the inference based on the score and metric type.

    The function takes a score and a metric type as input and returns a string indicating the inference based on the score and metric type. The inference is categorized as 'Good', 'Decent', or 'Bad' depending on the score and the metric type.

    The function first checks if the metric type is one of ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']. If it is, the function checks if the score is greater than or equal to 0.8, in which case it returns 'Good'. If the score is greater than or equal to 0.6 but less than 0.8, it returns 'Decent'. Otherwise, it returns 'Bad'.

    If the metric type is one of ['Mean Absolute Error', 'Mean Squared Error'], the function checks if the score is less than 0.5. If it is, it returns 'Good'. If the score is less than 1.0 but greater than or equal to 0.5, it returns 'Decent'. Otherwise, it returns 'Bad'.

    If the metric type is 'R-squared', the function checks if the score is greater than or equal to 0.8. If it is, it returns 'Good'. If the score is greater than or equal to 0.6 but less than 0.8, it returns 'Decent'. Otherwise, it returns 'Bad'.

    If the metric type does not match any of the specified types, the function returns 'Unknown'.
    """
    if metric_type in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']:
        if score >= 0.8:
            return 'Good'
        elif score >= 0.6:
            return 'Decent'
        else:
            return 'Bad'
    elif metric_type in ['Mean Absolute Error', 'Mean Squared Error']:
        if score <= 0.5:
            return 'Good'
        elif score <= 1.0:
            return 'Decent'
        else:
            return 'Bad'
    elif metric_type == 'R-squared':
        if score >= 0.8:
            return 'Good'
        elif score >= 0.6:
            return 'Decent'
        else:
            return 'Bad'
    return 'Unknown'

def train_model(data: pd.DataFrame, 
                target_column: str, 
                model: Any, 
                test_size: float = 0.2, 
                random_state: int = None) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """
    Splits the data into training and testing sets, trains the specified model, 
    and returns the trained model along with the train/test data and performance metrics.

    Parameters:
    ----------
    data : pd.DataFrame
        The input DataFrame containing features and the target variable.

    target_column : str
        The name of the column containing the target variable.

    model : Any
        An instantiated machine learning model object to be trained.

    test_size : float, optional
        Proportion of the dataset to include in the test split (default is 0.2).

    random_state : int, optional
        Random seed for reproducibility (default is None).

    Returns:
    -------
    Tuple[Any, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]
        A tuple containing the trained model, training features, 
        testing features, training target, testing target, and performance metrics as a DataFrame.
    """
    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state)

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Create an empty DataFrame for performance metrics
    performance_metrics = pd.DataFrame(columns=['Metric', 'Score', 'Inference'])

    # Choose performance metrics based on the type of model
    if hasattr(model, "predict_proba"):  # Likely a classification model
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        }
    else:  # Likely a regression model
        metrics = {
            'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
            'Mean Squared Error': mean_squared_error(y_test, y_pred),
            'R-squared': r2_score(y_test, y_pred)
        }

    # Populate the performance metrics DataFrame
    for metric, score in metrics.items():
        performance_metrics = pd.concat([
            performance_metrics, 
            pd.DataFrame({
                'Metric': [metric], 
                'Score': [score], 
                'Inference': [assess_performance(score, metric)]
            })
        ], ignore_index=True)

    return model, X_train, X_test, y_train, y_test, performance_metrics