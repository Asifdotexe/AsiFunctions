import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score, ShuffleSplit

def plot_histogram_with_stats(df: pd.DataFrame, 
                              columns: list[str] = None,
                              scale: str = 'linear'
) -> None:
    """
    This function plots a histogram with mean and median lines for the specified data series in the input DataFrame.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data series for which to generate the histogram. This DataFrame should have numerical variables for the histogram to be effective.
    - columns (list[str], optional): List of column names in the input DataFrame for which to generate the histogram. If not provided, all numerical columns will be used. Default is None.
    - scale (str, optional): String representing the scale for the y-axis of the histogram. Default is 'linear'.

    Returns:
    - None: This function does not return any value. It only plots the histogram with mean and median lines.

    Example usage:
    >>> # Plot a histogram with mean and median lines for the 'number_of_appliances' column
    >>> plot_histogram_with_stats(df, columns=['number_of_appliances'])
    """
    if columns is None:
        columns = df.describe().columns.to_list()
    
    for column in columns:
        sns.histplot(df[column])
        plt.axvline(df[column].mean(), color='r', linestyle='--')
        plt.axvline(df[column].median(), color='g', linestyle='-')
        
        plt.legend({'Mean': df[column].mean(), 'Median': df[column].median()})
        plt.title(f'Histogram of {column}')
        plt.yscale(scale)
        plt.show()

def plot_bar(df: pd.DataFrame, 
             x: str, 
             y: str, 
             hue: str = None, 
             palette: str = 'Set1'
) -> None:
    """
    This function plots a bar plot of the specified x and y variables from the input DataFrame. A bar plot is a type of plot that displays the frequency or count of data points in different categories.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data for the bar plot. This DataFrame should have numerical variables for the bar plot to be effective.
    - x (str): String representing the x-axis column name. This column will be used as the categories for the bar plot.
    - y (str): String representing the y-axis column name. This column will be used as the values for the bar plot.
    - hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None. If provided, the bar plot will be colored based on the values in this column.
    - palette (str, optional): String representing the color palette for the plot. Default is 'Set1'. This parameter specifies the colors that will be used for the bars in the plot.

    Returns:
    - None: This function does not return any value. It only plots the bar plot.

    Example usage:
    >>> # Plot a bar plot of 'number_of_appliances' vs 'room_area'
    >>> plot_bar(df, x='number_of_appliances', y='building_type')
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette)
    plt.title(f'Bar plot of {y} by {x}')
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame,
                             annot: bool = True,
                             fmt: str = '.2f',
                             cmap: str = 'RdBu_r',
                             line_width: float = .5,
                             square: bool = True,
                             annot_size: int = 12,
                             title: str = 'Correlation Heatmap' 
) -> None:
    """
    This function plots a correlation heatmap of the numerical columns in the input DataFrame.

    Parameters:
        - df (pd.DataFrame): Input DataFrame containing numerical data.
        - annot (bool, optional): If True, annotate the heatmap with correlation values. Defaults to True.
        - fmt (str, optional): Format of the correlation values in the heatmap. Defaults to '.2f'.
        - cmap (str, optional): Color map for the heatmap. Defaults to 'RdBu_r'.
        - line_width (float, optional): Width of the lines separating the cells in the heatmap. Defaults to .5.
        - square (bool, optional): If True, make the heatmap square by filling in the lower triangle. Defaults to True.
        - annot_size (int, optional): Size of the annotations in the heatmap. Defaults to 12.
        - title (str, optional): Title of the heatmap. Defaults to 'Correlation Heatmap'.

    Returns:
        - None: This function does not return any value, it only plots the correlation heatmap.
    """
    numerical_columns = df.describe().columns.to_list()
    correlation_matrix = df[numerical_columns].corr()

    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(correlation_matrix, mask=mask, annot=annot, fmt=fmt, cmap=cmap, center=0, linewidths=line_width, square=square, annot_kws={'size': annot_size})
    plt.title(title)
    plt.show()
    
def plot_heatmap(df: pd.DataFrame,
                 annot: bool = True,
                 fmt: str = '.2f',
                 cmap: str = 'coolwarm',
                 columns: list[str] = None,
                 title: str = 'Heatmap'
) -> None:
    """
    This function plots a heatmap with annotations for the specified data series in the input DataFrame. A heatmap is a type of plot that displays the correlation between pairs of variables in a dataset.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data series for which to generate the heatmap. This DataFrame should have numerical variables for the heatmap to be effective.
    - annot (bool, optional): Boolean representing whether to display annotations on the heatmap. Default is True.
    - fmt (str, optional): String representing the format for the values displayed on the heatmap. Default is '.2f'.
    - cmap (str, optional): String representing the color map for the heatmap. Default is 'coolwarm'.
    - columns (list[str], optional): List of column names in the input DataFrame for which to generate the heatmap. If not provided, all numerical columns will be used. Default is None.
    - title (str, optional): String representing the title for the heatmap. Default is 'Heatmap'.

    Returns:
    - None: This function does not return any value. It only plots the heatmap with annotations.

    Example usage:
    >>> # Plot a heatmap with annotations for the 'number_of_appliances' and 'room_area' columns
    >>> plot_heatmap(df, annot=True, columns=['number_of_appliances', 'room_area'])
    """
    if columns is None:
        columns = df.describe().columns.to_list()
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[columns], annot=annot, fmt=fmt, cmap=cmap)
    plt.title(title)
    plt.show()
    
def plot_count(df: pd.DataFrame, 
               x: str, 
               hue: str = None, 
               palette: str = 'Set1'
) -> None:
    """
    This function plots a count plot of the specified x variable from the input DataFrame. A count plot is a type of plot that displays the frequency or count of data points in different categories.
   
    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data for the count plot. This DataFrame should have numerical variables for the count plot to be effective.
    - x (str): String representing the x-axis column name. This column will be used as the categories for the count plot.
    - hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None. If provided, the count plot will be colored based on the values in this column.
    - palette (str, optional): String representing the color palette for the plot. Default is 'Set1'. This parameter specifies the colors that will be used for the bars in the plot.
    
    Returns:
    - None: This function does not return any value. It only plots the count plot.
    """
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x=x, hue=hue, palette=palette)
    plt.title(f'Count plot of {x}')
    plt.show()
    
def plot_scatter(df: pd.DataFrame,
                 x: str,
                 y: str,
                 hue: str = None, 
                 style: str = None,
                 size: int = None,
                 palette: str = 'Set1',
                 alpha: float = 0.7
) -> None:
    """
    This function plots a scatter plot of the specified x and y variables from the input DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data for the scatter plot.
    x (str): String representing the x-axis column name.
    y (str): String representing the y-axis column name.
    hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None.
    style (str, optional): Optional string representing the categorical variable for styling the plot. Default is None.
    size (int, optional): Optional integer representing the size of the markers. Default is None.
    palette (str, optional): String representing the color palette for the plot. Default is 'Set1'.
    alpha (float, optional): Float representing the transparency of the markers. Default is 0.7.

    Returns:
    None: This function does not return any value. It only plots the scatter plot.

    Example usage:
    >>> # Plot a scatter plot of 'number_of_appliances' vs 'room_area'
    >>> plot_scatter(df, x='number_of_appliances', y='room_area')
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, style=style, size=size, palette=palette, alpha=alpha)
    plt.title(f'Scatter plot of {y} vs {x}')
    plt.show()
    
def plot_box(df: pd.DataFrame,
             x: str,
             y: str,
             hue: str = None,
             palette: str = 'Set1'
) -> None:
    """
    This function plots a box plot of the specified x and y variables from the input DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data for the box plot.
    x (str): String representing the x-axis column name.
    y (str): String representing the y-axis column name.
    hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None.
    palette (str, optional): String representing the color palette for the plot. Default is 'Set1'.

    Returns:
    None: This function does not return any value. It only plots the box plot.

    Example usage:
    >>> # Plot a box plot of 'number_of_appliances' vs 'room_area'
    >>> plot_box(df, x='number_of_appliances', y='room_area')
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue, palette=palette)
    plt.title(f'Box plot of {y} by {x}')
    plt.show()
    
def plot_pair(df: pd.DataFrame,
              hue: str = None,
              palette: str = 'Set1'
) -> None:
    """
    This function plots a pairplot of the input DataFrame. A pairplot is a type of plot that displays the relationships between pairs of variables in a dataset.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data for the pairplot. This DataFrame should have numerical variables for the pairplot to be effective.
    hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None.
    palette (str, optional): String representing the color palette for the plot. Default is 'Set1'.

    Returns:
    None: This function does not return any value. It only plots the pairplot.

    Example usage:
    >>> # Plot a pairplot of the input DataFrame
    >>> plot_pair(df)
    """
    sns.pairplot(df, hue=hue, palette=palette)
    plt.show()
    
def plot_violin(df: pd.DataFrame, 
                x: str, 
                y: str, 
                hue: str = None, 
                split: bool = False, 
                palette: str = 'Set1'
) -> pd.DataFrame:
    """
    This function plots a violin plot of the specified x and y variables from the input DataFrame. A violin plot is a type of plot that displays the distribution of a variable and its underlying density.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data for the violin plot. This DataFrame should have numerical variables for the violin plot to be effective.
    - x (str): String representing the x-axis column name.
    - y (str): String representing the y-axis column name.
    - hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None.
    - split (bool, optional): Boolean representing whether to split the violin plot by the specified hue variable. Default is False.
    - palette (str, optional): String representing the color palette for the plot. Default is 'Set1'.

    Returns:
    - pd.DataFrame: The modified DataFrame with the violin plot added. This function does not return any other value.

    Example usage:
    >>> # Plot a violin plot of 'number_of_appliances' vs 'room_area'
    >>> plot_violin(df, x='number_of_appliances', y='room_area')
    """
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x=x, y=y, hue=hue, split=split, palette=palette)
    plt.title(f'Violin plot of {y} by {x}')
    plt.show()
    
def plot_line(df: pd.DataFrame,
              x: str, 
              y: str, 
              hue: str = None, 
              style: str = None, 
              markers: bool = False, 
              dashes: bool = True, 
              palette: str = 'Set1'
) -> None:
    """
    This function plots a line plot of the specified x and y variables from the input DataFrame. A line plot is a type of plot that displays the relationship between two variables over time or across different categories.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data for the line plot. This DataFrame should have numerical variables for the line plot to be effective.
    - x (str): String representing the x-axis column name.
    - y (str): String representing the y-axis column name.
    - hue (str, optional): Optional string representing the categorical variable for coloring the plot. Default is None.
    - style (str, optional): Optional string representing the categorical variable for styling the plot. Default is None.
    - markers (bool, optional): Boolean representing whether to display markers at each data point. Default is False.
    - dashes (bool, optional): Boolean representing whether to display dashed lines between data points. Default is True.
    - palette (str, optional): String representing the color palette for the plot. Default is 'Set1'.

    Returns:
    - None: This function does not return any value. It only plots the line plot.

    Example usage:
    >>> # Plot a line plot of 'number_of_appliances' vs 'room_area'
    >>> plot_line(df, x='number_of_appliances', y='room_area')
    """
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x, y=y, hue=hue, style=style, markers=markers, dashes=dashes, palette=palette)
    plt.title(f'Line plot of {y} over {x}')
    plt.show()
    
import matplotlib.pyplot as plt

def plot_roc_curve(fpr_list: list[float], 
                   tpr_list: list[float], 
                   roc_auc_list: list[float], 
                   labels: list[str]
) -> None:
    """
    Plot ROC curves for multiple models on the same graph.

    Parameters:
    - fpr_list (list): List of false positive rates for each model.
    - tpr_list (list): List of true positive rates for each model.
    - roc_auc_list (list): List of ROC AUC scores for each model.
    - labels (list): List of labels for each model.

    Returns:
    - None: This function does not return any value. It only plots the ROC curves for the given models.

    Example usage:
    >>> # Plot ROC curves for multiple models on the same graph
    >>> fpr_list = [0.1, 0.2, 0.3]
    >>> tpr_list = [0.8, 0.7, 0.6]
    >>> roc_auc_list = [0.9, 0.8, 0.7]
    >>> labels = ['Model 1', 'Model 2', 'Model 3']
    >>> plot_roc_curve(fpr_list, tpr_list, roc_auc_list, labels)
    """
    plt.figure(figsize=(10, 6))
    for fpr, tpr, roc_auc, label in zip(fpr_list, tpr_list, roc_auc_list, labels):
        plt.plot(fpr, tpr, label=f'{label} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_confusion_matrix(y_true: pd.DataFrame, 
                          y_pred: pd.DataFrame, 
                          labels: list[str] = None
)-> None:
    """
    Plot a confusion matrix.

    A confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are also known.
    
    Parameters:
    - y_true: True labels. This is the actual class labels of the test data.
    - y_pred: Predicted labels. This is the predicted class labels of the test data by the model.
    - labels: List of class labels (default is None). If provided, the confusion matrix will be labeled with these class names. If not provided, the confusion matrix will be labeled with 'True' and 'False' by default.

    Returns:
    - None: This function does not return any value. It only plots the confusion matrix.
    """
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    display_labels = labels if labels is not None else ['True', 'False']

    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    cm_display.plot()
    
def plot_feature_importance(importance_df: pd.DataFrame) -> None:
    """
    Plot feature importance for a model.

    Parameters:
    - importance_df (pd.DataFrame): DataFrame with the top features and their importances that you get after running the `get_feature_importance` function

    Returns:
    - None: This function does not return any value. It only plots the feature importances.

    Example usage:
    >>> # Plot feature importances for a model
    >>> import pandas as pd
    >>> import matplotlib.pyplot as plt
    >>> plot_feature_importance(importance_df)
    """
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances')
    plt.bar(importance_df['name'], importance_df['importance'], align='center')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    
def plot_precision_recall(y_true: pd.DataFrame,
                          y_scores: pd.DataFrame
) -> None:
    """
    Plot a precision-recall curve.

    Parameters:
    - y_true: True binary labels
    - y_scores: Estimated probabilities or decision function

    Returns:
    - None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    disp = PrecisionRecallDisplay(precision=precision, recall=recall)
    disp.plot()
    plt.title('Precision-Recall Curve')
    plt.show()
    
def visualize_cross_validation(model: object, 
                               X: pd.DataFrame, 
                               y: pd.Series, 
                               n_splits: int = 10, 
                               test_size: float = 0.2, 
                               random_state: int = 0
) -> None:
    """
    Visualize cross-validation scores for a given model using a boxplot.

    Parameters:
    - model (object): The machine learning model to evaluate.
    - X (pd.DataFrame): The feature matrix.
    - y (pd.Series): The target variable.
    - n_splits (int): Number of shuffle splits in cross-validation (default is 10).
    - test_size (float): Proportion of the dataset to include in the test split (default is 0.2).
    - random_state (int): Seed for reproducibility (default is 0).

    Returns:
    - None: This function does not return any value. It only displays a boxplot and prints summary statistics.

    Example usage:
    >>> from sklearn.linear_model import LinearRegression
    >>> visualize_cross_validation(LinearRegression(), X, y)
    """
    # Creating a ShuffleSplit object
    cross_validation = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)

    # Performing cross-validation and getting the scores
    cv_scores = cross_val_score(model, X, y, cv=cross_validation)

    # Displaying summary statistics
    print(f'We ran {n_splits} shuffle splits')
    print('Minimum cross-validation score:', min(cv_scores))
    print('Mean cross-validation score:', cv_scores.mean())
    print('Maximum cross-validation score:', max(cv_scores))

    # Creating a boxplot to visualize the cross-validation scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=cv_scores)
    plt.title(f'{model.__class__.__name__} Cross-Validation Scores', fontsize=14)
    plt.xlabel('Cross-Validation Splits', fontsize=12)
    plt.ylabel('Score', fontsize=12)  # Adjust the label based on your evaluation metric
    plt.grid(True)
    plt.tight_layout()
    plt.show()