import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_heatmap(df: pd.DataFrame,  # Input DataFrame containing numerical data
                             annot: bool = True,  # If True, annotate the heatmap with correlation values
                             fmt: str = '.2f',  # Format of the correlation values in the heatmap
                             cmap: str = 'RdBu_r',  # Color map for the heatmap
                             line_width: float = .5,  # Width of the lines separating the cells in the heatmap
                             square: bool = True,  # If True, make the heatmap square by filling in the lower triangle
                             annot_size: int = 12,  # Size of the annotations in the heatmap
                             title: str = 'Correlation Heatmap'  # Title of the heatmap
) -> None:  # This function plots a correlation heatmap of the numerical columns in the input DataFrame
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