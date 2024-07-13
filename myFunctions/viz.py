import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay

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