"""
Visualization functions for creating statistical plots in AzzamAnalyst.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from typing import Dict, List, Union, Any, Tuple, Optional
import warnings
import json
import matplotlib.dates as mdates

# Configure matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
warnings.filterwarnings('ignore')  # Ignore warnings for cleaner output


def fig_to_base64(fig, format='png', dpi=300) -> str:
    """
    Convert a matplotlib figure to base64 encoded string.
    
    Args:
        fig: Matplotlib figure
        format: Image format (png, jpg, svg, etc.)
        dpi: Resolution in dots per inch
        
    Returns:
        Base64 encoded HTML img tag
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/{format};base64,{img_str}" style="max-width:100%;height:auto;" />'


def plot_histogram(df: pd.DataFrame, column: str, bins: int = 20, kde: bool = True, 
                  title: str = None, color: str = '#1976D2', show_stats: bool = True) -> str:
    """
    Create a histogram for a numeric column.
    
    Args:
        df: Input DataFrame
        column: Column to plot
        bins: Number of bins
        kde: Whether to include kernel density estimate
        title: Plot title
        color: Bar color
        show_stats: Whether to show statistics on plot
        
    Returns:
        Base64 encoded HTML img tag
    """
    if column not in df.columns:
        return f"<p>Error: Column '{column}' not found in data</p>"
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"<p>Error: Column '{column}' is not numeric</p>"
    
    data = df[column].dropna()
    
    if len(data) == 0:
        return "<p>Error: No non-missing values in column</p>"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    sns.histplot(data, bins=bins, kde=kde, color=color, ax=ax)
    
    # Set title and labels
    ax.set_title(title or f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    
    # Add statistics if requested
    if show_stats:
        stats_text = (
            f"Mean: {data.mean():.2f}\n"
            f"Median: {data.median():.2f}\n"
            f"Std Dev: {data.std():.2f}\n"
            f"Min: {data.min():.2f}\n"
            f"Max: {data.max():.2f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_boxplot(df: pd.DataFrame, y_column: str, x_column: str = None, 
               title: str = None, palette: str = 'viridis', orient: str = 'vertical') -> str:
    """
    Create a box plot for a numeric column, optionally grouped by a categorical column.
    
    Args:
        df: Input DataFrame
        y_column: Numeric column for box plot
        x_column: Optional categorical column for grouping
        title: Plot title
        palette: Color palette
        orient: Orientation ('vertical' or 'horizontal')
        
    Returns:
        Base64 encoded HTML img tag
    """
    if y_column not in df.columns:
        return f"<p>Error: Column '{y_column}' not found in data</p>"
    
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"<p>Error: Column '{y_column}' is not numeric</p>"
    
    if x_column is not None and x_column not in df.columns:
        return f"<p>Error: Column '{x_column}' not found in data</p>"
    
    data = df[[y_column]] if x_column is None else df[[y_column, x_column]]
    data = data.dropna()
    
    if len(data) == 0:
        return "<p>Error: No non-missing values in column(s)</p>"
    
    # Determine figure size based on the number of groups
    if x_column is not None:
        n_groups = data[x_column].nunique()
        figsize = (max(8, n_groups * 0.8), 6) if orient == 'vertical' else (8, max(6, n_groups * 0.5))
    else:
        figsize = (8, 6)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create boxplot
    if orient == 'vertical':
        if x_column is not None:
            sns.boxplot(x=x_column, y=y_column, data=data, palette=palette, ax=ax)
            plt.xticks(rotation=45, ha='right')
        else:
            sns.boxplot(y=y_column, data=data, palette=palette, ax=ax)
    else:  # horizontal
        if x_column is not None:
            sns.boxplot(y=x_column, x=y_column, data=data, palette=palette, ax=ax)
        else:
            sns.boxplot(x=y_column, data=data, palette=palette, ax=ax)
    
    # Set title
    ax.set_title(title or (f'Box Plot of {y_column} by {x_column}' if x_column else f'Box Plot of {y_column}'))
    
    # Add statistics
    if x_column is None:
        stats_text = (
            f"Mean: {data[y_column].mean():.2f}\n"
            f"Median: {data[y_column].median():.2f}\n"
            f"IQR: {data[y_column].quantile(0.75) - data[y_column].quantile(0.25):.2f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_scatter(df: pd.DataFrame, x_column: str, y_column: str, color_column: str = None,
              title: str = None, alpha: float = 0.7, size: int = 60,
              regression: bool = False) -> str:
    """
    Create a scatter plot between two numeric columns.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis
        y_column: Column for y-axis
        color_column: Optional column for color coding points
        title: Plot title
        alpha: Transparency of points
        size: Size of points
        regression: Whether to add regression line
        
    Returns:
        Base64 encoded HTML img tag
    """
    if x_column not in df.columns:
        return f"<p>Error: Column '{x_column}' not found in data</p>"
    
    if y_column not in df.columns:
        return f"<p>Error: Column '{y_column}' not found in data</p>"
    
    if not pd.api.types.is_numeric_dtype(df[x_column]):
        return f"<p>Error: Column '{x_column}' is not numeric</p>"
    
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"<p>Error: Column '{y_column}' is not numeric</p>"
    
    if color_column is not None and color_column not in df.columns:
        return f"<p>Error: Column '{color_column}' not found in data</p>"
    
    # Select relevant columns and drop missing values
    columns = [x_column, y_column]
    if color_column:
        columns.append(color_column)
    
    data = df[columns].dropna()
    
    if len(data) == 0:
        return "<p>Error: No non-missing values in selected columns</p>"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create scatter plot
    if color_column:
        # Check if color column is categorical or numeric
        if pd.api.types.is_numeric_dtype(data[color_column]):
            scatter = ax.scatter(data[x_column], data[y_column], 
                              c=data[color_column], alpha=alpha, s=size, cmap='viridis')
            cbar = plt.colorbar(scatter)
            cbar.set_label(color_column)
        else:
            # For categorical color column, use seaborn
            sns.scatterplot(x=x_column, y=y_column, hue=color_column, data=data, 
                          alpha=alpha, s=size, ax=ax)
            plt.legend(title=color_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax.scatter(data[x_column], data[y_column], alpha=alpha, s=size)
    
    # Add regression line if requested
    if regression:
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(data[x_column], data[y_column])
        x_range = np.linspace(data[x_column].min(), data[x_column].max(), 100)
        ax.plot(x_range, intercept + slope * x_range, 'r--', 
              label=f'y = {slope:.3f}x + {intercept:.3f} (r²={r_value**2:.3f})')
        ax.legend()
        
        # Add correlation information
        corr_text = (
            f"Correlation: {r_value:.3f}\n"
            f"R²: {r_value**2:.3f}\n"
            f"p-value: {p_value:.3g}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, 
              fontsize=10, verticalalignment='top', bbox=props)
    
    # Set title and labels
    ax.set_title(title or f'{y_column} vs {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_correlation_heatmap(df: pd.DataFrame, columns: List[str] = None, method: str = 'pearson',
                          title: str = None, cmap: str = 'coolwarm', annot: bool = True) -> str:
    """
    Create a correlation heatmap for numeric columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to include (None for all numeric)
        method: Correlation method ('pearson', 'spearman', or 'kendall')
        title: Plot title
        cmap: Color map
        annot: Whether to annotate heatmap with values
        
    Returns:
        Base64 encoded HTML img tag
    """
    # Get numeric columns if not specified
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter to ensure columns exist and are numeric
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_cols) < 2:
        return "<p>Error: Need at least two numeric columns for a correlation heatmap</p>"
    
    # Calculate correlation matrix
    corr_matrix = df[numeric_cols].corr(method=method)
    
    # Determine figure size based on the number of variables
    figsize = (max(8, len(numeric_cols) * 0.7), max(6, len(numeric_cols) * 0.7))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask for upper triangle
    sns.heatmap(corr_matrix, mask=mask, cmap=cmap, annot=annot, fmt='.2f', 
              square=True, linewidths=.5, cbar_kws={'shrink': .8}, ax=ax)
    
    # Set title
    ax.set_title(title or f'Correlation Matrix ({method.capitalize()})')
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_bar(df: pd.DataFrame, x_column: str, y_column: str = None, 
           title: str = None, color: str = '#1976D2', orientation: str = 'vertical',
           top_n: int = 15, agg_func: str = 'mean') -> str:
    """
    Create a bar chart for categorical data.
    
    Args:
        df: Input DataFrame
        x_column: Categorical column
        y_column: Optional numeric column for values (None for counts)
        title: Plot title
        color: Bar color
        orientation: Bar orientation ('vertical' or 'horizontal')
        top_n: Show only top N categories
        agg_func: Aggregation function when y_column is provided ('mean', 'sum', etc.)
        
    Returns:
        Base64 encoded HTML img tag
    """
    if x_column not in df.columns:
        return f"<p>Error: Column '{x_column}' not found in data</p>"
    
    if y_column is not None and y_column not in df.columns:
        return f"<p>Error: Column '{y_column}' not found in data</p>"
    
    if y_column is not None and not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"<p>Error: Column '{y_column}' is not numeric</p>"
    
    # Prepare data
    if y_column is None:
        # Value counts for single column
        values = df[x_column].value_counts().nlargest(top_n)
        x_label = x_column
        y_label = 'Count'
    else:
        # Aggregated values
        values = df.groupby(x_column)[y_column].agg(agg_func).nlargest(top_n)
        x_label = x_column
        y_label = f'{agg_func.capitalize()} of {y_column}'
    
    # Determine figure size based on the number of categories
    n_categories = len(values)
    if orientation == 'vertical':
        figsize = (max(8, n_categories * 0.5), 6)
    else:  # horizontal
        figsize = (8, max(6, n_categories * 0.4))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    if orientation == 'vertical':
        values.plot.bar(ax=ax, color=color)
        plt.xticks(rotation=45, ha='right')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    else:  # horizontal
        values.plot.barh(ax=ax, color=color)
        ax.set_xlabel(y_label)
        ax.set_ylabel(x_label)
    
    # Set title
    if title:
        ax.set_title(title)
    elif y_column:
        ax.set_title(f'{agg_func.capitalize()} {y_column} by {x_column}')
    else:
        ax.set_title(f'Counts of {x_column}')
    
    # Add value labels on the bars
    for i, v in enumerate(values):
        if orientation == 'vertical':
            ax.text(i, v * 1.01, f'{v:.0f}' if v.is_integer() else f'{v:.2f}', 
                  ha='center', va='bottom', fontsize=8)
        else:  # horizontal
            ax.text(v * 1.01, i, f'{v:.0f}' if v.is_integer() else f'{v:.2f}', 
                  ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_pie(df: pd.DataFrame, column: str, title: str = None, 
           colors: str = 'viridis', top_n: int = 8) -> str:
    """
    Create a pie chart for categorical data.
    
    Args:
        df: Input DataFrame
        column: Categorical column
        title: Plot title
        colors: Color palette
        top_n: Show only top N categories (others grouped as 'Other')
        
    Returns:
        Base64 encoded HTML img tag
    """
    if column not in df.columns:
        return f"<p>Error: Column '{column}' not found in data</p>"
    
    # Get value counts
    value_counts = df[column].value_counts()
    
    # Handle too many categories
    if len(value_counts) > top_n:
        top_values = value_counts.nlargest(top_n - 1)
        others = pd.Series({'Other': value_counts[~value_counts.index.isin(top_values.index)].sum()})
        values = pd.concat([top_values, others])
    else:
        values = value_counts
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create pie chart
    explode = [0.05] * len(values)  # Explode all slices slightly
    wedges, texts, autotexts = ax.pie(
        values, 
        labels=values.index, 
        autopct='%1.1f%%', 
        explode=explode,
        shadow=True, 
        startangle=90,
        textprops={'fontsize': 9}
    )
    
    # Customize appearance
    plt.setp(autotexts, fontsize=9, weight='bold')
    plt.setp(texts, fontsize=9)
    
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax.axis('equal')
    
    # Set title
    ax.set_title(title or f'Distribution of {column}')
    
    # Add legend if there are many categories
    if len(values) > 5:
        ax.legend(
            wedges, 
            values.index,
            title=column,
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1)
        )
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_line(df: pd.DataFrame, x_column: str, y_column: str, group_column: str = None,
           title: str = None, color: str = '#1976D2', marker: str = 'o') -> str:
    """
    Create a line plot, optionally grouped by a categorical column.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis (numeric or datetime)
        y_column: Column for y-axis (numeric)
        group_column: Optional column for grouping lines
        title: Plot title
        color: Line color (when not grouped)
        marker: Marker style
        
    Returns:
        Base64 encoded HTML img tag
    """
    if x_column not in df.columns:
        return f"<p>Error: Column '{x_column}' not found in data</p>"
    
    if y_column not in df.columns:
        return f"<p>Error: Column '{y_column}' not found in data</p>"
    
    if not pd.api.types.is_numeric_dtype(df[y_column]):
        return f"<p>Error: Column '{y_column}' is not numeric</p>"
    
    if group_column is not None and group_column not in df.columns:
        return f"<p>Error: Column '{group_column}' not found in data</p>"
    
    # Prepare data
    columns = [x_column, y_column]
    if group_column:
        columns.append(group_column)
    
    data = df[columns].dropna()
    
    if len(data) == 0:
        return "<p>Error: No non-missing values in selected columns</p>"
    
    # Sort by x_column if it's datetime or numeric
    if pd.api.types.is_datetime64_dtype(data[x_column]) or pd.api.types.is_numeric_dtype(data[x_column]):
        data = data.sort_values(by=x_column)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create line plot
    if group_column:
        # Group by the categorical column
        groups = data[group_column].unique()
        
        # Limit to top N groups if there are too many
        if len(groups) > 10:
            # Count occurrences of each group
            group_counts = data[group_column].value_counts()
            groups = group_counts.nlargest(10).index.tolist()
            
            # Filter data to selected groups
            data = data[data[group_column].isin(groups)]
        
        # Plot each group
        for group in groups:
            group_data = data[data[group_column] == group]
            ax.plot(group_data[x_column], group_data[y_column], marker=marker, label=str(group))
        
        ax.legend(title=group_column)
    else:
        # Simple line plot
        ax.plot(data[x_column], data[y_column], color=color, marker=marker)
    
    # Format x-axis for datetime
    if pd.api.types.is_datetime64_dtype(data[x_column]):
        # Determine appropriate date formatting based on range
        date_range = (data[x_column].max() - data[x_column].min()).days
        
        if date_range > 365 * 3:  # More than 3 years
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        elif date_range > 365:  # More than 1 year
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        elif date_range > 30:  # More than 1 month
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        
        fig.autofmt_xdate()
    
    # Set title and labels
    ax.set_title(title or f'{y_column} over {x_column}')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_heatmap(df: pd.DataFrame, x_column: str, y_column: str, value_column: str = None,
              title: str = None, cmap: str = 'viridis', annot: bool = True) -> str:
    """
    Create a heatmap from categorical x and y columns.
    
    Args:
        df: Input DataFrame
        x_column: Column for x-axis (categorical)
        y_column: Column for y-axis (categorical)
        value_column: Optional column for cell values (numeric, None for counts)
        title: Plot title
        cmap: Color map
        annot: Whether to annotate heatmap with values
        
    Returns:
        Base64 encoded HTML img tag
    """
    if x_column not in df.columns:
        return f"<p>Error: Column '{x_column}' not found in data</p>"
    
    if y_column not in df.columns:
        return f"<p>Error: Column '{y_column}' not found in data</p>"
    
    if value_column is not None and value_column not in df.columns:
        return f"<p>Error: Column '{value_column}' not found in data</p>"
    
    if value_column is not None and not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"<p>Error: Column '{value_column}' is not numeric</p>"
    
    # Prepare data for heatmap
    if value_column is None:
        # Contingency table (counts)
        heatmap_data = pd.crosstab(df[y_column], df[x_column])
    else:
        # Pivot table with values
        heatmap_data = df.pivot_table(index=y_column, columns=x_column, values=value_column, aggfunc='mean')
    
    # Check dimensions - limit if too large
    max_categories = 20
    if heatmap_data.shape[0] > max_categories or heatmap_data.shape[1] > max_categories:
        return f"<p>Error: Too many categories for heatmap (max {max_categories} in each dimension)</p>"
    
    # Determine figure size based on categories
    figsize = (max(8, heatmap_data.shape[1] * 0.5), max(6, heatmap_data.shape[0] * 0.5))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap=cmap, annot=annot, fmt='.1f' if value_column else 'd', 
              linewidths=0.5, ax=ax)
    
    # Set title
    if title:
        ax.set_title(title)
    elif value_column:
        ax.set_title(f'Heatmap of {value_column} by {x_column} and {y_column}')
    else:
        ax.set_title(f'Frequency of {x_column} and {y_column}')
    
    # Rotate x-axis labels if many categories
    if heatmap_data.shape[1] > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_pca(df: pd.DataFrame, columns: List[str] = None, n_components: int = 2,
           title: str = None, point_labels: str = None) -> str:
    """
    Create a PCA (Principal Component Analysis) plot.
    
    Args:
        df: Input DataFrame
        columns: Numeric columns to include in PCA (None for all numeric)
        n_components: Number of components to compute (2 or 3)
        title: Plot title
        point_labels: Optional column to use for labeling points
        
    Returns:
        Base64 encoded HTML img tag
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    # Get numeric columns if not specified
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter to ensure columns exist and are numeric
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_cols) < 2:
        return "<p>Error: Need at least two numeric columns for PCA</p>"
    
    # Check point labels column
    if point_labels is not None and point_labels not in df.columns:
        return f"<p>Error: Column '{point_labels}' not found in data</p>"
    
    # Prepare data (drop missing values)
    columns_to_use = numeric_cols.copy()
    if point_labels:
        columns_to_use.append(point_labels)
    
    data = df[columns_to_use].dropna()
    
    if len(data) < 3:
        return "<p>Error: Not enough complete observations for PCA</p>"
    
    # Standardize the data
    X = data[numeric_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    n_components = min(n_components, len(numeric_cols))
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    
    # Create plot based on number of components
    if n_components >= 3:
        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        if point_labels:
            # Color by category
            for label in data[point_labels].unique():
                mask = (data[point_labels] == label)
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    pca_result[mask, 2],
                    label=str(label),
                    alpha=0.7
                )
            ax.legend(title=point_labels)
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)')
    else:
        # 2D plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if point_labels:
            # Color by category
            for label in data[point_labels].unique():
                mask = (data[point_labels] == label)
                ax.scatter(
                    pca_result[mask, 0],
                    pca_result[mask, 1],
                    label=str(label),
                    alpha=0.7
                )
            ax.legend(title=point_labels)
        else:
            ax.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
        
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # Set title
    ax.set_title(title or 'PCA Plot')
    
    # Add grid
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    # Also create loadings plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get loadings
    loadings = pca.components_.T
    
    # Plot loadings for first two components
    for i, feature in enumerate(numeric_cols):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], head_width=0.05, head_length=0.05, fc='black', ec='black')
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='blue', ha='center', va='center')
    
    # Draw a unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax.add_artist(circle)
    
    # Set limits and labels
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA Loading Plot')
    ax.grid(True)
    
    plt.tight_layout()
    loadings_result = fig_to_base64(fig)
    plt.close(fig)
    
    # Combine both plots
    return f"""
    <div>
        <h3>PCA Score Plot</h3>
        {result}
        <h3>PCA Loading Plot</h3>
        {loadings_result}
    </div>
    """


def plot_cluster(df: pd.DataFrame, columns: List[str] = None, n_clusters: int = 3,
              method: str = 'kmeans', title: str = None) -> str:
    """
    Create a cluster plot using K-means or hierarchical clustering.
    
    Args:
        df: Input DataFrame
        columns: Numeric columns to include in clustering (None for all numeric)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'hierarchical')
        title: Plot title
        
    Returns:
        Base64 encoded HTML img tag
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Get numeric columns if not specified
    if columns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        # Filter to ensure columns exist and are numeric
        numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if len(numeric_cols) < 2:
        return "<p>Error: Need at least two numeric columns for clustering</p>"
    
    # Prepare data (drop missing values)
    data = df[numeric_cols].dropna()
    
    if len(data) < n_clusters + 1:
        return f"<p>Error: Not enough complete observations for clustering with {n_clusters} clusters</p>"
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)
    
    # Perform clustering
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = model.fit_predict(X_scaled)
        centers = model.cluster_centers_
    else:  # hierarchical
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Perform hierarchical clustering
        Z = linkage(X_scaled, method='ward')
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust') - 1  # 0-based indexing
        
        # Calculate cluster centers
        centers = np.array([X_scaled[cluster_labels == i].mean(axis=0) for i in range(n_clusters)])
    
    # Reduce to 2D for visualization if more than 2 dimensions
    if len(numeric_cols) > 2:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        centers_pca = pca.transform(centers)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6)
        ax.scatter(centers_pca[:, 0], centers_pca[:, 1], s=200, marker='X', c='red', label='Cluster Centers')
        
        ax.set_title(title or f'{method.capitalize()} Clustering (PCA)')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    else:
        # Plot directly if 2D
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(data[numeric_cols[0]], data[numeric_cols[1]], c=cluster_labels, cmap='viridis', alpha=0.6)
        
        if method == 'kmeans':
            # Add cluster centers (only for k-means)
            ax.scatter(
                scaler.inverse_transform(centers)[:, 0], 
                scaler.inverse_transform(centers)[:, 1], 
                s=200, marker='X', c='red', label='Cluster Centers'
            )
        
        ax.set_title(title or f'{method.capitalize()} Clustering')
        ax.set_xlabel(numeric_cols[0])
        ax.set_ylabel(numeric_cols[1])
    
    # Add legend
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)
    
    if method == 'kmeans':
        ax.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    # Get cluster plot
    cluster_plot = fig_to_base64(fig)
    plt.close(fig)
    
    # For hierarchical clustering, also create dendrogram
    if method == 'hierarchical':
        from scipy.cluster.hierarchy import dendrogram
        
        fig, ax = plt.subplots(figsize=(12, 6))
        dendrogram(Z, ax=ax, truncate_mode='lastp', p=30, leaf_rotation=90)
        ax.set_title('Hierarchical Clustering Dendrogram')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Distance')
        plt.axhline(y=Z[-(n_clusters-1), 2], color='r', linestyle='--')
        plt.tight_layout()
        
        dendrogram_plot = fig_to_base64(fig)
        plt.close(fig)
        
        # Combine both plots
        return f"""
        <div>
            <h3>Cluster Plot</h3>
            {cluster_plot}
            <h3>Dendrogram</h3>
            {dendrogram_plot}
        </div>
        """
    
    return cluster_plot


def plot_missing_values(df: pd.DataFrame, title: str = None) -> str:
    """
    Create a visualization of missing values pattern.
    
    Args:
        df: Input DataFrame
        title: Plot title
        
    Returns:
        Base64 encoded HTML img tag
    """
    # Create a mask of missing values
    missing_mask = df.isna()
    
    # Calculate missing percentages by column
    missing_percent = missing_mask.sum() / len(df) * 100
    
    # Sort columns by missing percentage
    sorted_columns = missing_percent.sort_values(ascending=False).index
    
    # Use only columns with missing values
    missing_columns = sorted_columns[missing_percent[sorted_columns] > 0]
    
    if len(missing_columns) == 0:
        return "<p>No missing values found in the dataset</p>"
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # Heatmap of missing values
    sns.heatmap(
        missing_mask[missing_columns],
        cbar=False,
        yticklabels=False,
        cmap='binary',
        ax=ax1
    )
    
    ax1.set_title('Missing Values Pattern')
    ax1.set_xlabel('Variables')
    ax1.set_ylabel('Observations')
    
    # Bar chart of missing percentages
    missing_percent[missing_columns].plot.barh(ax=ax2)
    ax2.set_title('Missing Values (%)')
    ax2.set_xlabel('Percentage')
    ax2.set_xlim(0, 100)
    
    # Add percentages to the bars
    for i, v in enumerate(missing_percent[missing_columns]):
        ax2.text(v + 1, i, f'{v:.1f}%', va='center')
    
    # Overall title
    if title:
        fig.suptitle(title, fontsize=16, y=1.05)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_time_series(df: pd.DataFrame, date_column: str, value_column: str, 
                   group_column: str = None, ma_window: int = None,
                   title: str = None, color: str = '#1976D2') -> str:
    """
    Create a time series plot.
    
    Args:
        df: Input DataFrame
        date_column: Column with dates
        value_column: Column with values to plot
        group_column: Optional column for grouping
        ma_window: Window size for moving average (None for no smoothing)
        title: Plot title
        color: Line color (when not grouped)
        
    Returns:
        Base64 encoded HTML img tag
    """
    if date_column not in df.columns:
        return f"<p>Error: Column '{date_column}' not found in data</p>"
    
    if value_column not in df.columns:
        return f"<p>Error: Column '{value_column}' not found in data</p>"
    
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"<p>Error: Column '{value_column}' is not numeric</p>"
    
    if group_column is not None and group_column not in df.columns:
        return f"<p>Error: Column '{group_column}' not found in data</p>"
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_dtype(df[date_column]):
        try:
            df = df.copy()
            df[date_column] = pd.to_datetime(df[date_column])
        except:
            return f"<p>Error: Column '{date_column}' cannot be converted to datetime</p>"
    
    # Prepare data
    columns = [date_column, value_column]
    if group_column:
        columns.append(group_column)
    
    data = df[columns].dropna()
    
    if len(data) == 0:
        return "<p>Error: No non-missing values in selected columns</p>"
    
    # Sort by date
    data = data.sort_values(by=date_column)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot time series
    if group_column:
        # Group by the categorical column
        groups = data[group_column].unique()
        
        # Limit to top N groups if there are too many
        if len(groups) > 7:
            # Count occurrences of each group
            group_counts = data[group_column].value_counts()
            groups = group_counts.nlargest(7).index.tolist()
            
            # Filter data to selected groups
            data = data[data[group_column].isin(groups)]
        
        # Plot each group
        for group in groups:
            group_data = data[data[group_column] == group]
            
            # Apply moving average if requested
            if ma_window and len(group_data) > ma_window:
                # Calculate moving average
                values = group_data[value_column].rolling(window=ma_window).mean()
                
                # Plot both raw data and moving average
                ax.plot(group_data[date_column], group_data[value_column], alpha=0.3)
                ax.plot(group_data[date_column], values, label=f"{group} (MA{ma_window})")
            else:
                ax.plot(group_data[date_column], group_data[value_column], label=str(group))
        
        ax.legend(title=group_column)
    else:
        # Apply moving average if requested
        if ma_window and len(data) > ma_window:
            # Calculate moving average
            values = data[value_column].rolling(window=ma_window).mean()
            
            # Plot both raw data and moving average
            ax.plot(data[date_column], data[value_column], color=color, alpha=0.3, label='Raw data')
            ax.plot(data[date_column], values, color='red', label=f'Moving average (window={ma_window})')
            ax.legend()
        else:
            # Simple line plot
            ax.plot(data[date_column], data[value_column], color=color)
    
    # Format x-axis for datetime
    date_range = (data[date_column].max() - data[date_column].min()).days
    
    if date_range > 365 * 3:  # More than 3 years
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
    elif date_range > 365:  # More than 1 year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    elif date_range > 30:  # More than 1 month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    
    fig.autofmt_xdate()
    
    # Set title and labels
    ax.set_title(title or f'Time Series of {value_column}')
    ax.set_xlabel(date_column)
    ax.set_ylabel(value_column)
    
    # Add grid
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_roc_curve(y_true: List[Any], y_score: List[float], title: str = None) -> str:
    """
    Create a ROC curve for binary classification.
    
    Args:
        y_true: True binary labels
        y_score: Target scores (probabilities or decision function)
        title: Plot title
        
    Returns:
        Base64 encoded HTML img tag
    """
    from sklearn.metrics import roc_curve, auc
    
    if len(y_true) != len(y_score):
        return "<p>Error: y_true and y_score must have the same length</p>"
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot ROC curve
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title or 'Receiver Operating Characteristic (ROC) Curve')
    
    # Add grid and legend
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right')
    
    # Set axis limits
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def plot_survival_curve(time: List[float], event: List[int], group: List[Any] = None, title: str = None) -> str:
    """
    Create a Kaplan-Meier survival curve.
    
    Args:
        time: Array of times to event or censoring
        event: Array of event indicators (1=event occurred, 0=censored)
        group: Optional array of group labels
        title: Plot title
        
    Returns:
        Base64 encoded HTML img tag
    """
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        return "<p>Error: lifelines package is required for survival analysis</p>"
    
    if len(time) != len(event):
        return "<p>Error: time and event must have the same length</p>"
    
    if group is not None and len(group) != len(time):
        return "<p>Error: group must have the same length as time and event</p>"
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if group is None:
        # Single survival curve
        kmf = KaplanMeierFitter()
        kmf.fit(time, event, label='Overall')
        kmf.plot_survival_function(ax=ax, ci_show=True)
        
        # Add at risk counts
        add_at_risk_counts(kmf, ax)
    else:
        # Survival curves by group
        unique_groups = set(group)
        
        if len(unique_groups) < 2:
            return "<p>Error: At least two groups are needed for group comparison</p>"
        
        # Fit KM for each group
        results = {}
        for g in unique_groups:
            mask = [x == g for x in group]
            kmf = KaplanMeierFitter()
            kmf.fit(
                durations=[time[i] for i in range(len(time)) if mask[i]], 
                event_observed=[event[i] for i in range(len(event)) if mask[i]], 
                label=str(g)
            )
            results[g] = kmf
            kmf.plot_survival_function(ax=ax, ci_show=True)
        
        # Perform log-rank test if there are exactly two groups
        if len(unique_groups) == 2:
            g1, g2 = unique_groups
            mask1 = [x == g1 for x in group]
            mask2 = [x == g2 for x in group]
            
            t1 = [time[i] for i in range(len(time)) if mask1[i]]
            e1 = [event[i] for i in range(len(event)) if mask1[i]]
            t2 = [time[i] for i in range(len(time)) if mask2[i]]
            e2 = [event[i] for i in range(len(event)) if mask2[i]]
            
            results = logrank_test(t1, t2, e1, e2)
            
            # Add log-rank test p-value
            ax.text(
                0.05, 0.05, 
                f'Log-rank test p-value: {results.p_value:.4f}', 
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    # Set labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel('Survival Probability')
    ax.set_title(title or 'Kaplan-Meier Survival Curve')
    
    # Add grid
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def add_at_risk_counts(kmf, ax, rows_to_show=None):
    """Helper function to add at-risk counts to Kaplan-Meier plot."""
    try:
        # Get survival function
        survival_df = kmf.survival_function_
        
        # Choose time points
        if rows_to_show is None:
            # Determine number of time points to show based on x-axis range
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            if x_range > 100:
                rows_to_show = np.linspace(0, survival_df.index.max(), 10)
            else:
                rows_to_show = np.arange(0, int(survival_df.index.max()) + 1, max(1, int(x_range / 10)))
        
        # Add table with at-risk counts
        ax_table = ax.twinx()
        ax_table.set_ylim(ax.get_ylim())
        ax_table.set_yticks([])
        ax_table.set_ylabel('At risk')
        
        # Calculate at-risk counts for time points
        at_risk_counts = []
        for t in rows_to_show:
            count = (kmf.durations >= t).sum()
            at_risk_counts.append(count)
        
        # Format table data
        table_data = [['At risk:'] + at_risk_counts]
        
        # Add table
        from matplotlib.table import Table
        table = Table(ax_table, bbox=[0, -0.15, 1, 0.1], axes=ax_table)
        
        # Add cells
        for (i, j), val in np.ndenumerate(table_data):
            table.add_cell(i, j, 1/len(table_data[0]), 1/len(table_data), text=val, loc='center')
        
        # Add row with time points
        for i, t in enumerate(rows_to_show):
            table.add_cell(1, i+1, 1/len(table_data[0]), 1/len(table_data), text=f'{t:.0f}', loc='center')
        
        ax_table.add_table(table)
        
    except Exception as e:
        # If anything goes wrong, just skip adding at-risk counts
        print(f"Error adding at-risk counts: {e}")


def plot_multiple(plots: List[Dict[str, Any]], title: str = None, figsize: Tuple[int, int] = None) -> str:
    """
    Create a figure with multiple plots.
    
    Args:
        plots: List of plot configurations
        title: Overall figure title
        figsize: Figure size (width, height)
        
    Returns:
        Base64 encoded HTML img tag
    """
    n_plots = len(plots)
    
    if n_plots == 0:
        return "<p>Error: No plots specified</p>"
    
    # Determine layout based on number of plots
    if n_plots == 1:
        nrows, ncols = 1, 1
    elif n_plots == 2:
        nrows, ncols = 1, 2
    elif n_plots <= 4:
        nrows, ncols = 2, 2
    elif n_plots <= 6:
        nrows, ncols = 2, 3
    elif n_plots <= 9:
        nrows, ncols = 3, 3
    else:
        nrows = (n_plots + 2) // 3
        ncols = 3
    
    # Create figure
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Make axes accessible for 1D or 0D cases
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Hide extra subplots
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    # Create each subplot
    for i, plot_config in enumerate(plots):
        if i >= len(axes):
            break
        
        ax = axes[i]
        plot_type = plot_config.get('type', 'histogram')
        
        try:
            # Extract DataFrame or arrays from plot config
            if 'df' in plot_config:
                df = plot_config['df']
            elif 'data' in plot_config:
                df = pd.DataFrame(plot_config['data'])
            else:
                # Try to extract arrays for direct plotting
                x = plot_config.get('x', [])
                y = plot_config.get('y', [])
                
                if len(x) != len(y):
                    ax.text(0.5, 0.5, "Error: x and y must have the same length", 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                plot_config['df'] = pd.DataFrame({'x': x, 'y': y})
                plot_config['x_column'] = 'x'
                plot_config['y_column'] = 'y'
                
                df = plot_config['df']
            
            # Extract common parameters
            title = plot_config.get('title', '')
            
            # Create the appropriate plot type
            if plot_type == 'histogram':
                column = plot_config.get('column')
                if column and column in df.columns:
                    if pd.api.types.is_numeric_dtype(df[column]):
                        sns.histplot(
                            df[column].dropna(), 
                            bins=plot_config.get('bins', 20),
                            kde=plot_config.get('kde', True),
                            color=plot_config.get('color', '#1976D2'),
                            ax=ax
                        )
                        ax.set_title(title or f'Distribution of {column}')
                        ax.set_xlabel(column)
                        ax.set_ylabel('Frequency')
            
            elif plot_type == 'scatter':
                x_col = plot_config.get('x_column')
                y_col = plot_config.get('y_column')
                
                if x_col in df.columns and y_col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        ax.scatter(
                            df[x_col], 
                            df[y_col], 
                            alpha=plot_config.get('alpha', 0.7),
                            color=plot_config.get('color', '#1976D2'),
                            s=plot_config.get('size', 60)
                        )
                        
                        # Add regression line if requested
                        if plot_config.get('regression', False):
                            from scipy import stats
                            slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col], df[y_col])
                            x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
                            ax.plot(x_range, intercept + slope * x_range, 'r--', 
                                  label=f'y = {slope:.3f}x + {intercept:.3f} (r²={r_value**2:.3f})')
                            ax.legend()
                        
                        ax.set_title(title or f'{y_col} vs {x_col}')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
            
            elif plot_type == 'bar':
                x_col = plot_config.get('x_column')
                y_col = plot_config.get('y_column')
                
                if x_col in df.columns:
                    if y_col is None:
                        # Value counts
                        value_counts = df[x_col].value_counts().nlargest(plot_config.get('top_n', 10))
                        value_counts.plot.bar(ax=ax, color=plot_config.get('color', '#1976D2'))
                        ax.set_title(title or f'Counts of {x_col}')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel('Count')
                    elif y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                        # Aggregated bar chart
                        agg_func = plot_config.get('agg_func', 'mean')
                        df.groupby(x_col)[y_col].agg(agg_func).nlargest(plot_config.get('top_n', 10)).plot.bar(
                            ax=ax, 
                            color=plot_config.get('color', '#1976D2')
                        )
                        ax.set_title(title or f'{agg_func.capitalize()} {y_col} by {x_col}')
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(f'{agg_func.capitalize()} of {y_col}')
                    
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            elif plot_type == 'line':
                x_col = plot_config.get('x_column')
                y_col = plot_config.get('y_column')
                
                if x_col in df.columns and y_col in df.columns:
                    if pd.api.types.is_datetime64_dtype(df[x_col]):
                        # Time series
                        sorted_df = df.sort_values(by=x_col)
                        ax.plot(
                            sorted_df[x_col], 
                            sorted_df[y_col], 
                            color=plot_config.get('color', '#1976D2'),
                            marker=plot_config.get('marker', 'o'),
                            markersize=plot_config.get('markersize', 4)
                        )
                        
                        # Format x-axis for datetime
                        date_range = (sorted_df[x_col].max() - sorted_df[x_col].min()).days
                        if date_range > 365:
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                        else:
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                            
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                    else:
                        # Regular line plot
                        ax.plot(
                            df[x_col], 
                            df[y_col], 
                            color=plot_config.get('color', '#1976D2'),
                            marker=plot_config.get('marker', 'o'),
                            markersize=plot_config.get('markersize', 4)
                        )
                    
                    ax.set_title(title or f'{y_col} vs {x_col}')
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(y_col)
            
            elif plot_type == 'boxplot':
                y_col = plot_config.get('y_column')
                x_col = plot_config.get('x_column')
                
                if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                    if x_col is None:
                        # Simple boxplot
                        sns.boxplot(y=df[y_col], ax=ax, color=plot_config.get('color', '#1976D2'))
                        ax.set_title(title or f'Box Plot of {y_col}')
                    elif x_col in df.columns:
                        # Grouped boxplot
                        sns.boxplot(x=x_col, y=y_col, data=df, ax=ax)
                        ax.set_title(title or f'Box Plot of {y_col} by {x_col}')
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add grid for all plots
            ax.grid(alpha=0.3)
            
        except Exception as e:
            # If plot creation fails, show error message in the subplot
            ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center', transform=ax.transAxes)
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16)
        plt.subplots_adjust(top=0.9)
    
    plt.tight_layout()
    result = fig_to_base64(fig)
    plt.close(fig)
    
    return result


def create_visualization(viz_type: str, data: Dict[str, Any], params: Dict[str, Any] = None) -> str:
    """
    Create a visualization based on type and parameters.
    
    Args:
        viz_type: Type of visualization
        data: Data dictionary or DataFrame
        params: Parameters for the visualization
        
    Returns:
        Base64 encoded HTML img tag
    """
    if params is None:
        params = {}
    
    # Convert data to DataFrame if needed
    if isinstance(data, dict) and any(isinstance(data[key], list) for key in data):
        df = pd.DataFrame(data)
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        return f"<p>Error: Invalid data format for visualization</p>"
    
    # Choose the appropriate visualization function
    if viz_type == 'histogram':
        return plot_histogram(df, 
                           column=params.get('column'), 
                           bins=params.get('bins', 20), 
                           kde=params.get('kde', True),
                           title=params.get('title'),
                           color=params.get('color', '#1976D2'),
                           show_stats=params.get('show_stats', True))
    
    elif viz_type == 'scatter':
        return plot_scatter(df,
                         x_column=params.get('x'), 
                         y_column=params.get('y'),
                         color_column=params.get('color_by'),
                         title=params.get('title'),
                         alpha=params.get('alpha', 0.7),
                         size=params.get('size', 60),
                         regression=params.get('regression', False))
    
    elif viz_type == 'boxplot':
        return plot_boxplot(df,
                         y_column=params.get('y'),
                         x_column=params.get('x'),
                         title=params.get('title'),
                         palette=params.get('palette', 'viridis'),
                         orient=params.get('orient', 'vertical'))
    
    elif viz_type == 'correlation':
        return plot_correlation_heatmap(df,
                                     columns=params.get('columns'),
                                     method=params.get('method', 'pearson'),
                                     title=params.get('title'),
                                     cmap=params.get('cmap', 'coolwarm'),
                                     annot=params.get('annot', True))
    
    elif viz_type == 'bar':
        return plot_bar(df,
                     x_column=params.get('x'),
                     y_column=params.get('y'),
                     title=params.get('title'),
                     color=params.get('color', '#1976D2'),
                     orientation=params.get('orientation', 'vertical'),
                     top_n=params.get('top_n', 15),
                     agg_func=params.get('agg_func', 'mean'))
    
    elif viz_type == 'pie':
        return plot_pie(df,
                     column=params.get('column'),
                     title=params.get('title'),
                     colors=params.get('colors', 'viridis'),
                     top_n=params.get('top_n', 8))
    
    elif viz_type == 'line':
        return plot_line(df,
                      x_column=params.get('x'),
                      y_column=params.get('y'),
                      group_column=params.get('group_by'),
                      title=params.get('title'),
                      color=params.get('color', '#1976D2'),
                      marker=params.get('marker', 'o'))
    
    elif viz_type == 'heatmap':
        return plot_heatmap(df,
                         x_column=params.get('x'),
                         y_column=params.get('y'),
                         value_column=params.get('value'),
                         title=params.get('title'),
                         cmap=params.get('cmap', 'viridis'),
                         annot=params.get('annot', True))
    
    elif viz_type == 'pca':
        return plot_pca(df,
                     columns=params.get('columns'),
                     n_components=params.get('n_components', 2),
                     title=params.get('title'),
                     point_labels=params.get('point_labels'))
    
    elif viz_type == 'cluster':
        return plot_cluster(df,
                         columns=params.get('columns'),
                         n_clusters=params.get('n_clusters', 3),
                         method=params.get('method', 'kmeans'),
                         title=params.get('title'))
    
    elif viz_type == 'missing':
        return plot_missing_values(df,
                                title=params.get('title'))
    
    elif viz_type == 'time_series':
        return plot_time_series(df,
                             date_column=params.get('date'),
                             value_column=params.get('value'),
                             group_column=params.get('group_by'),
                             ma_window=params.get('ma_window'),
                             title=params.get('title'),
                             color=params.get('color', '#1976D2'))
    
    elif viz_type == 'multiple':
        return plot_multiple(plots=params.get('plots', []),
                          title=params.get('title'),
                          figsize=params.get('figsize'))
    
    else:
        return f"<p>Error: Unknown visualization type '{viz_type}'</p>"
