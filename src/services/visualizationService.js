import { runPython } from './pyodideService';

/**
 * Generate visualization using Python
 * @param {string} vizType - Type of visualization to generate
 * @param {Object} data - Data to visualize
 * @param {Object} params - Parameters for the visualization
 * @returns {Promise<string>} - HTML representation of the visualization
 */
export const generateVisualization = async (vizType, data, params = {}) => {
  try {
    // Convert data and params to Python-friendly format
    const dataJSON = JSON.stringify(data);
    const paramsJSON = JSON.stringify(params);
    
    // Generate the visualization
    const pythonCode = `
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Set up matplotlib
plt.switch_backend('agg')
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# Parse inputs
viz_type = ${JSON.stringify(vizType)}
data = json.loads('''${dataJSON}''')
params = json.loads('''${paramsJSON}''')

# Helper function for visualization
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;height:auto;" />'

# Convert to DataFrame if needed
if isinstance(data, list) and all(isinstance(item, dict) for item in data):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and all(isinstance(data[key], list) for key in data):
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

# Generate the requested visualization
html_output = ""

# Histogram
if viz_type == 'histogram':
    column = params.get('column')
    if column and column in df.columns:
        try:
            # Check if column is numeric
            if pd.api.types.is_numeric_dtype(df[column]):
                # Create figure
                fig, ax = plt.subplots()
                
                # Get histogram parameters
                bins = params.get('bins', 20)
                kde = params.get('kde', True)
                color = params.get('color', '#1976D2')
                title = params.get('title', f'Distribution of {column}')
                
                # Plot histogram
                sns.histplot(df[column].dropna(), bins=bins, kde=kde, color=color, ax=ax)
                
                # Customize plot
                ax.set_title(title)
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                ax.grid(alpha=0.3)
                
                # Add statistics if requested
                if params.get('show_stats', True):
                    mean = df[column].mean()
                    median = df[column].median()
                    std = df[column].std()
                    stats_text = f"Mean: {mean:.2f}\\nMedian: {median:.2f}\\nStd Dev: {std:.2f}"
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                            verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5, color='white'))
                
                plt.tight_layout()
                html_output = fig_to_base64(fig)
                plt.close(fig)
            else:
                html_output = "<p>Error: Column is not numeric</p>"
        except Exception as e:
            html_output = f"<p>Error generating histogram: {str(e)}</p>"
    else:
        html_output = "<p>Error: Column not specified or not found in data</p>"

# Scatter plot
elif viz_type == 'scatter':
    x_col = params.get('x')
    y_col = params.get('y')
    
    if x_col in df.columns and y_col in df.columns:
        try:
            # Check if columns are numeric
            if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                # Create figure
                fig, ax = plt.subplots()
                
                # Get scatter parameters
                color = params.get('color', '#1976D2')
                alpha = params.get('alpha', 0.7)
                title = params.get('title', f'{y_col} vs {x_col}')
                size = params.get('size', 60)
                
                # Color by group if specified
                if 'color_by' in params and params['color_by'] in df.columns:
                    group_col = params['color_by']
                    scatter = ax.scatter(df[x_col], df[y_col], c=df[group_col], alpha=alpha, s=size, cmap='viridis')
                    plt.colorbar(scatter, label=group_col)
                else:
                    ax.scatter(df[x_col], df[y_col], color=color, alpha=alpha, s=size)
                
                # Add regression line if requested
                if params.get('regression', False):
                    from scipy import stats
                    mask = ~(np.isnan(df[x_col]) | np.isnan(df[y_col]))
                    if np.sum(mask) > 1:  # Need at least 2 points for regression
                        slope, intercept, r_value, p_value, std_err = stats.linregress(df[x_col][mask], df[y_col][mask])
                        x_range = np.linspace(df[x_col].min(), df[x_col].max(), 100)
                        ax.plot(x_range, intercept + slope * x_range, 'r', 
                                label=f'y = {slope:.3f}x + {intercept:.3f} (r²={r_value**2:.3f})')
                        ax.legend()
                
                # Customize plot
                ax.set_title(title)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                ax.grid(alpha=0.3)
                
                plt.tight_layout()
                html_output = fig_to_base64(fig)
                plt.close(fig)
            else:
                html_output = "<p>Error: One or both columns are not numeric</p>"
        except Exception as e:
            html_output = f"<p>Error generating scatter plot: {str(e)}</p>"
    else:
        html_output = "<p>Error: X or Y column not specified or not found in data</p>"

# Bar chart
elif viz_type == 'bar':
    x_col = params.get('x')
    y_col = params.get('y')
    
    if x_col in df.columns:
        try:
            # Create figure
            fig, ax = plt.subplots()
            
            # Get parameters
            title = params.get('title', f'Bar Chart of {x_col}')
            color = params.get('color', '#1976D2')
            horizontal = params.get('horizontal', False)
            
            # Plot bar chart - single variable (counts)
            if y_col is None:
                # Get value counts
                value_counts = df[x_col].value_counts().sort_values(ascending=False)
                # Limit to top N categories if needed
                max_categories = params.get('max_categories', 20)
                if len(value_counts) > max_categories:
                    value_counts = value_counts[:max_categories]
                    
                if horizontal:
                    value_counts.plot.barh(ax=ax, color=color)
                    ax.set_xlabel('Count')
                    ax.set_ylabel(x_col)
                else:
                    value_counts.plot.bar(ax=ax, color=color)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel('Count')
            
            # Plot bar chart - two variables (y values by x category)
            elif y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                # Aggregate by group
                agg_func = params.get('agg_func', 'mean')
                grouped = df.groupby(x_col)[y_col].agg(agg_func).sort_values(ascending=False)
                # Limit to top N categories if needed
                max_categories = params.get('max_categories', 20)
                if len(grouped) > max_categories:
                    grouped = grouped[:max_categories]
                    
                if horizontal:
                    grouped.plot.barh(ax=ax, color=color)
                    ax.set_xlabel(f'{agg_func.capitalize()} of {y_col}')
                    ax.set_ylabel(x_col)
                else:
                    grouped.plot.bar(ax=ax, color=color)
                    ax.set_xlabel(x_col)
                    ax.set_ylabel(f'{agg_func.capitalize()} of {y_col}')
            
            # Customize plot
            ax.set_title(title)
            plt.xticks(rotation=45, ha='right') if not horizontal else None
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            html_output = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            html_output = f"<p>Error generating bar chart: {str(e)}</p>"
    else:
        html_output = "<p>Error: X column not specified or not found in data</p>"

# Box plot
elif viz_type == 'boxplot':
    y_col = params.get('y')
    x_col = params.get('x')
    
    if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
        try:
            # Create figure
            fig, ax = plt.subplots()
            
            # Get parameters
            title = params.get('title', f'Box Plot of {y_col}')
            color = params.get('color', '#1976D2')
            
            # Simple box plot for a single variable
            if x_col is None:
                sns.boxplot(y=df[y_col], ax=ax, color=color)
                ax.set_ylabel(y_col)
            
            # Box plot grouped by a categorical variable
            elif x_col in df.columns:
                # Limit to top N categories if needed
                if pd.api.types.is_numeric_dtype(df[x_col]) and df[x_col].nunique() > 10:
                    html_output = "<p>Error: X column has too many unique values for a box plot</p>"
                    plt.close(fig)
                    return html_output
                    
                max_categories = params.get('max_categories', 15)
                if df[x_col].nunique() > max_categories:
                    # Get top categories by frequency
                    top_cats = df[x_col].value_counts().nlargest(max_categories).index.tolist()
                    plot_df = df[df[x_col].isin(top_cats)]
                else:
                    plot_df = df
                
                sns.boxplot(x=x_col, y=y_col, data=plot_df, ax=ax)
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                plt.xticks(rotation=45, ha='right')
            
            # Customize plot
            ax.set_title(title)
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            html_output = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            html_output = f"<p>Error generating box plot: {str(e)}</p>"
    else:
        html_output = "<p>Error: Y column not specified, not found in data, or not numeric</p>"

# Line plot
elif viz_type == 'line':
    x_col = params.get('x')
    y_col = params.get('y')
    
    if x_col in df.columns and y_col in df.columns:
        try:
            # Create figure
            fig, ax = plt.subplots()
            
            # Get parameters
            title = params.get('title', f'{y_col} over {x_col}')
            color = params.get('color', '#1976D2')
            marker = params.get('marker', 'o')
            
            # Ensure x is sorted for line plot
            if pd.api.types.is_numeric_dtype(df[x_col]) or pd.api.types.is_datetime64_dtype(df[x_col]):
                plot_df = df.sort_values(by=x_col)
            else:
                plot_df = df
            
            # Line plot by group if specified
            if 'group_by' in params and params['group_by'] in df.columns:
                group_col = params['group_by']
                groups = plot_df[group_col].unique()
                for group in groups:
                    group_df = plot_df[plot_df[group_col] == group]
                    ax.plot(group_df[x_col], group_df[y_col], marker=marker, label=str(group))
                ax.legend(title=group_col)
            else:
                ax.plot(plot_df[x_col], plot_df[y_col], color=color, marker=marker)
            
            # Customize plot
            ax.set_title(title)
            ax.set_xlabel(x_col)
            ax.set_ylabel(y_col)
            ax.grid(alpha=0.3)
            
            # Rotate x labels if non-numeric
            if not pd.api.types.is_numeric_dtype(df[x_col]):
                plt.xticks(rotation=45, ha='right')
            
            plt.tight_layout()
            html_output = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            html_output = f"<p>Error generating line plot: {str(e)}</p>"
    else:
        html_output = "<p>Error: X or Y column not specified or not found in data</p>"

# Heatmap / Correlation Matrix
elif viz_type == 'heatmap':
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            html_output = "<p>Error: Not enough numeric columns for a heatmap</p>"
        else:
            # Get parameters
            title = params.get('title', 'Correlation Heatmap')
            columns = params.get('columns', numeric_cols)
            
            # Filter columns that exist in the dataframe
            columns = [col for col in columns if col in numeric_cols]
            
            # Limit number of columns to avoid huge heatmap
            max_columns = params.get('max_columns', 20)
            if len(columns) > max_columns:
                columns = columns[:max_columns]
            
            # Get correlation method
            method = params.get('method', 'pearson')
            
            # Create correlation matrix
            corr = df[columns].corr(method=method)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(max(8, len(columns) * 0.5), max(6, len(columns) * 0.5)))
            
            # Plot heatmap
            mask = params.get('mask_upper', True)
            if mask:
                mask_array = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask_array, cmap='coolwarm', annot=True, 
                            fmt='.2f', square=True, linewidths=.5, cbar_kws={'shrink': .8}, ax=ax)
            else:
                sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f', 
                            square=True, linewidths=.5, cbar_kws={'shrink': .8}, ax=ax)
            
            # Customize plot
            ax.set_title(title)
            
            plt.tight_layout()
            html_output = fig_to_base64(fig)
            plt.close(fig)
    except Exception as e:
        html_output = f"<p>Error generating heatmap: {str(e)}</p>"

# Pie chart
elif viz_type == 'pie':
    column = params.get('column')
    
    if column and column in df.columns:
        try:
            # Get parameters
            title = params.get('title', f'Pie Chart of {column}')
            max_categories = params.get('max_categories', 8)
            
            # Get value counts
            value_counts = df[column].value_counts()
            
            # Handle too many categories
            if len(value_counts) > max_categories:
                # Keep top categories and group others
                top_counts = value_counts.iloc[:max_categories-1]
                others_sum = value_counts.iloc[max_categories-1:].sum()
                
                # Create new series with 'Others' category
                if others_sum > 0:
                    top_counts['Others'] = others_sum
                value_counts = top_counts
            
            # Create figure
            fig, ax = plt.subplots()
            
            # Plot pie chart
            explode = [0.05] * len(value_counts)  # Explode all slices slightly
            value_counts.plot.pie(autopct='%1.1f%%', ax=ax, explode=explode, 
                                shadow=params.get('shadow', True),
                                startangle=params.get('startangle', 90))
            
            # Customize plot
            ax.set_title(title)
            ax.set_ylabel('')  # Hide the label
            
            plt.tight_layout()
            html_output = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            html_output = f"<p>Error generating pie chart: {str(e)}</p>"
    else:
        html_output = "<p>Error: Column not specified or not found in data</p>"

# Pair plot (scatter matrix)
elif viz_type == 'pairplot':
    try:
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            html_output = "<p>Error: Not enough numeric columns for a pair plot</p>"
        else:
            # Get parameters
            columns = params.get('columns', numeric_cols[:min(5, len(numeric_cols))])
            hue = params.get('hue')
            
            # Filter columns that exist in the dataframe
            columns = [col for col in columns if col in numeric_cols]
            
            # Check hue column
            if hue and hue not in df.columns:
                hue = None
            
            # Create pair plot
            if hue:
                g = sns.pairplot(df, vars=columns, hue=hue, diag_kind='kde')
            else:
                g = sns.pairplot(df, vars=columns, diag_kind='kde')
            
            # Customize plot
            plt.tight_layout()
            html_output = fig_to_base64(g.fig)
            plt.close(g.fig)
    except Exception as e:
        html_output = f"<p>Error generating pair plot: {str(e)}</p>"

# Count plot
elif viz_type == 'countplot':
    column = params.get('column')
    
    if column and column in df.columns:
        try:
            # Create figure
            fig, ax = plt.subplots()
            
            # Get parameters
            title = params.get('title', f'Count Plot of {column}')
            color = params.get('color', '#1976D2')
            max_categories = params.get('max_categories', 15)
            
            # Handle too many categories
            if df[column].nunique() > max_categories:
                # Get top categories by frequency
                top_cats = df[column].value_counts().nlargest(max_categories).index.tolist()
                plot_df = df[df[column].isin(top_cats)].copy()
            else:
                plot_df = df.copy()
            
            # Plot count plot
            if 'hue' in params and params['hue'] in df.columns:
                sns.countplot(x=column, hue=params['hue'], data=plot_df, ax=ax)
            else:
                sns.countplot(x=column, data=plot_df, ax=ax, color=color)
            
            # Customize plot
            ax.set_title(title)
            ax.set_xlabel(column)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            ax.grid(alpha=0.3)
            
            plt.tight_layout()
            html_output = fig_to_base64(fig)
            plt.close(fig)
        except Exception as e:
            html_output = f"<p>Error generating count plot: {str(e)}</p>"
    else:
        html_output = "<p>Error: Column not specified or not found in data</p>"

# Fallback for unknown visualization type
else:
    html_output = f"<p>Error: Unknown visualization type '{viz_type}'</p>"

html_output
    `;
    
    const result = await runPython(pythonCode);
    return result;
  } catch (error) {
    console.error('Error generating visualization:', error);
    return `<p>Error generating visualization: ${error.message}</p>`;
  }
};

/**
 * Generate a combined visualization (multiple plots)
 * @param {Array} visualizations - Array of visualization configs
 * @param {Object} data - Data to visualize
 * @returns {Promise<string>} - HTML representation of combined visualization
 */
export const generateCombinedVisualization = async (visualizations, data) => {
  try {
    // Convert data and visualizations to Python-friendly format
    const dataJSON = JSON.stringify(data);
    const visualizationsJSON = JSON.stringify(visualizations);
    
    // Generate the combined visualization
    const pythonCode = `
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from matplotlib.gridspec import GridSpec

# Set up matplotlib
plt.switch_backend('agg')
plt.style.use('seaborn-v0_8-whitegrid')

# Parse inputs
visualizations = json.loads('''${visualizationsJSON}''')
data = json.loads('''${dataJSON}''')

# Helper function for visualization
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;height:auto;" />'

# Convert to DataFrame if needed
if isinstance(data, list) and all(isinstance(item, dict) for item in data):
    df = pd.DataFrame(data)
elif isinstance(data, dict) and all(isinstance(data[key], list) for key in data):
    df = pd.DataFrame(data)
else:
    df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)

# Create figure with subplots
n_plots = len(visualizations)
if n_plots <= 0:
    html_output = "<p>No visualizations specified</p>"
else:
    # Determine grid layout based on number of plots
    if n_plots == 1:
        rows, cols = 1, 1
    elif n_plots == 2:
        rows, cols = 1, 2
    elif n_plots <= 4:
        rows, cols = 2, 2
    elif n_plots <= 6:
        rows, cols = 2, 3
    elif n_plots <= 9:
        rows, cols = 3, 3
    else:
        rows, cols = (n_plots + 2) // 3, 3  # Approximate for many plots
    
    # Create figure
    fig = plt.figure(figsize=(5*cols, 4*rows))
    gs = GridSpec(rows, cols, figure=fig)
    
    # Create each subplot
    for i, viz in enumerate(visualizations):
        if i >= rows * cols:  # Skip if we run out of grid space
            break
            
        # Get subplot position
        row = i // cols
        col = i % cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get visualization parameters
        viz_type = viz.get('type', 'histogram')
        
        # Generate the appropriate visualization
        try:
            # histogram
            if viz_type == 'histogram':
                column = viz.get('column')
                if column and column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
                    sns.histplot(df[column].dropna(), bins=viz.get('bins', 20), 
                                kde=viz.get('kde', True), ax=ax)
                    ax.set_title(viz.get('title', f'Distribution of {column}'))
                    ax.set_xlabel(column)
                    ax.set_ylabel('Frequency')
            
            # scatter
            elif viz_type == 'scatter':
                x_col = viz.get('x')
                y_col = viz.get('y')
                if x_col in df.columns and y_col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        ax.scatter(df[x_col], df[y_col], alpha=0.7)
                        ax.set_title(viz.get('title', f'{y_col} vs {x_col}'))
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(y_col)
            
            # bar
            elif viz_type == 'bar':
                x_col = viz.get('x')
                y_col = viz.get('y')
                if x_col in df.columns:
                    if y_col is None:
                        # Value counts
                        value_counts = df[x_col].value_counts().nlargest(10)
                        value_counts.plot.bar(ax=ax)
                        ax.set_title(viz.get('title', f'Bar Chart of {x_col}'))
                        ax.set_xlabel(x_col)
                        ax.set_ylabel('Count')
                    elif y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                        # Aggregated bar chart
                        agg_func = viz.get('agg_func', 'mean')
                        df.groupby(x_col)[y_col].agg(agg_func).nlargest(10).plot.bar(ax=ax)
                        ax.set_title(viz.get('title', f'{agg_func.capitalize()} {y_col} by {x_col}'))
                        ax.set_xlabel(x_col)
                        ax.set_ylabel(f'{agg_func.capitalize()} of {y_col}')
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # boxplot
            elif viz_type == 'boxplot':
                y_col = viz.get('y')
                x_col = viz.get('x')
                if y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                    if x_col is None:
                        # Simple boxplot
                        sns.boxplot(y=df[y_col], ax=ax)
                        ax.set_title(viz.get('title', f'Box Plot of {y_col}'))
                    elif x_col in df.columns:
                        # Grouped boxplot
                        sns.boxplot(x=x_col, y=y_col, data=df, ax=ax)
                        ax.set_title(viz.get('title', f'Box Plot of {y_col} by {x_col}'))
                        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add grid to all plots
            ax.grid(alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", horizontalalignment='center',
                    verticalalignment='center', transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='red', alpha=0.2))
    
    plt.tight_layout()
    html_output = fig_to_base64(fig)
    plt.close(fig)

html_output
    `;
    
    const result = await runPython(pythonCode);
    return result;
  } catch (error) {
    console.error('Error generating combined visualization:', error);
    return `<p>Error generating combined visualization: ${error.message}</p>`;
  }
};

/**
 * Export visualization to different formats
 * @param {string} htmlContent - HTML content of the visualization
 * @param {string} format - Export format (png, pdf, svg)
 * @param {Object} options - Export options
 * @returns {Promise<Blob>} - Exported blob
 */
export const exportVisualization = async (htmlContent, format = 'png', options = {}) => {
  // This would be a real implementation using libraries like html2canvas, jsPDF, etc.
  // For now, we'll just return a placeholder
  
  console.log(`Exporting visualization as ${format} with options:`, options);
  
  // Mock implementation
  if (format === 'png' || format === 'jpg') {
    // Extract base64 image data if it exists
    const imgMatch = htmlContent.match(/src="data:image\/[^;]+;base64,([^"]+)"/);
    if (imgMatch && imgMatch[1]) {
      const base64Data = imgMatch[1];
      const binary = atob(base64Data);
      const array = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) {
        array[i] = binary.charCodeAt(i);
      }
      return new Blob([array], { type: `image/${format}` });
    }
  }
  
  // Return placeholder blob for other formats
  return new Blob(['Exported visualization content'], { type: 'text/plain' });
};
