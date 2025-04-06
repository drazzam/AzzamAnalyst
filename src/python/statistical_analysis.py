import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io
import base64
import json
from typing import Dict, List, Tuple, Union, Any

# Configure Matplotlib for non-interactive use
plt.switch_backend('agg')
plt.style.use('seaborn-v0_8-whitegrid')

class BiostatisticalAnalyzer:
    """
    Comprehensive biostatistical analysis toolkit for various statistical methods.
    """
    
    def __init__(self, data):
        """
        Initialize the analyzer with data.
        
        Args:
            data: pandas DataFrame or list of dictionaries
        """
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            self.df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data.copy()
        else:
            raise ValueError("Data must be a pandas DataFrame or a list of dictionaries")
        
        # Results container
        self.results = {
            'tables': [],
            'visualizations': [],
            'statistics': [],
            'summary': ''
        }
        
        # Analyze dataset structure
        self._analyze_dataset_structure()
    
    def _analyze_dataset_structure(self):
        """Analyze the dataset structure and classify variables."""
        self.variables = {}
        
        for col in self.df.columns:
            var_info = {
                'name': col,
                'missing': self.df[col].isna().sum(),
                'missing_pct': self.df[col].isna().mean() * 100,
                'unique_values': self.df[col].nunique()
            }
            
            # Determine variable type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                unique_count = self.df[col].nunique()
                if unique_count <= 5:
                    var_info['type'] = 'categorical'
                    var_info['subtype'] = 'ordinal' if unique_count > 2 else 'binary'
                else:
                    var_info['type'] = 'continuous'
                    
                    # Check for normality using Shapiro-Wilk test if enough data
                    if 3 <= len(self.df[col].dropna()) <= 5000:
                        try:
                            shapiro_test = stats.shapiro(self.df[col].dropna())
                            var_info['normality'] = {
                                'test': 'shapiro',
                                'statistic': shapiro_test.statistic,
                                'p_value': shapiro_test.pvalue,
                                'is_normal': shapiro_test.pvalue > 0.05
                            }
                        except:
                            var_info['normality'] = {'test': 'failed'}
                    
            elif pd.api.types.is_datetime64_dtype(self.df[col]):
                var_info['type'] = 'datetime'
            else:
                if self.df[col].nunique() <= 10:
                    var_info['type'] = 'categorical'
                    var_info['subtype'] = 'nominal'
                else:
                    var_info['type'] = 'text'
            
            self.variables[col] = var_info
    
    def _fig_to_base64(self, fig):
        """Convert matplotlib figure to base64 string for embedding in HTML."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;height:auto;" />'
    
    def descriptive_statistics(self, variables=None):
        """
        Generate descriptive statistics for the specified variables.
        
        Args:
            variables: List of variables to analyze (None for all)
        
        Returns:
            self
        """
        if variables is None:
            variables = self.df.columns.tolist()
        
        # Filter to existing columns
        variables = [var for var in variables if var in self.df.columns]
        
        if not variables:
            self.results['statistics'].append({
                'title': 'Descriptive Statistics Error',
                'text': "No valid variables found for descriptive statistics."
            })
            return self
        
        # Split variables by type
        continuous_vars = [var for var in variables 
                        if var in self.variables and self.variables[var]['type'] == 'continuous']
        categorical_vars = [var for var in variables 
                         if var in self.variables and self.variables[var]['type'] == 'categorical']
        
        # Descriptive statistics for continuous variables
        if continuous_vars:
            # Calculate statistics
            desc_stats = self.df[continuous_vars].describe().transpose()
            desc_stats['missing'] = self.df[continuous_vars].isnull().sum()
            desc_stats['missing_pct'] = (self.df[continuous_vars].isnull().sum() / len(self.df) * 100).round(2)
            
            # Add skewness and kurtosis
            desc_stats['skewness'] = self.df[continuous_vars].skew()
            desc_stats['kurtosis'] = self.df[continuous_vars].kurtosis()
            
            # Create table for results
            desc_table = desc_stats.reset_index()
            desc_table.columns = ['Variable'] + list(desc_table.columns[1:])
            
            # Format for display
            headers = ['Variable', 'Count', 'Mean', 'Std Dev', 'Min', '25%', 'Median', '75%', 'Max', 'Missing', 'Missing %', 'Skewness', 'Kurtosis']
            rows = []
            for _, row in desc_table.iterrows():
                formatted_row = [row['Variable']]
                for header in headers[1:]:
                    header_key = header.lower().replace(' ', '_').replace('%', 'pct')
                    value = row.get(header_key, '')
                    
                    if isinstance(value, (int, float)) and not pd.isna(value):
                        if header in ['Count', 'Missing']:
                            formatted_row.append(f"{int(value)}")
                        elif header == 'Missing %':
                            formatted_row.append(f"{value:.2f}%")
                        else:
                            formatted_row.append(f"{value:.4f}")
                    else:
                        formatted_row.append('N/A')
                
                rows.append(formatted_row)
            
            # Add to results
            self.results['tables'].append({
                'title': 'Descriptive Statistics - Continuous Variables',
                'description': 'Summary statistics for continuous variables',
                'headers': headers,
                'rows': rows,
                'csvContent': desc_stats.to_csv()
            })
            
            # Create histograms for continuous variables
            for var in continuous_vars:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(self.df[var].dropna(), kde=True, ax=ax)
                ax.set_title(f'Distribution of {var}')
                ax.set_xlabel(var)
                ax.set_ylabel('Frequency')
                plt.tight_layout()
                
                self.results['visualizations'].append({
                    'title': f'Distribution of {var}',
                    'description': f'Histogram showing the distribution of {var}',
                    'content': self._fig_to_base64(fig)
                })
                plt.close(fig)
                
                # QQ plot to check normality
                fig, ax = plt.subplots(figsize=(10, 6))
                qqplot(self.df[var].dropna(), line='s', ax=ax)
                ax.set_title(f'Q-Q Plot for {var}')
                plt.tight_layout()
                
                self.results['visualizations'].append({
                    'title': f'Q-Q Plot for {var}',
                    'description': f'Quantile-Quantile plot to check normality of {var}',
                    'content': self._fig_to_base64(fig)
                })
                plt.close(fig)
                
                # Box plot
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.boxplot(y=self.df[var].dropna(), ax=ax)
                ax.set_title(f'Box Plot of {var}')
                ax.set_ylabel(var)
                plt.tight_layout()
                
                self.results['visualizations'].append({
                    'title': f'Box Plot of {var}',
                    'description': f'Box plot showing the distribution and potential outliers of {var}',
                    'content': self._fig_to_base64(fig)
                })
                plt.close(fig)
        
        # Descriptive statistics for categorical variables
        if categorical_vars:
            cat_stats = []
            
            for var in categorical_vars:
                # Calculate frequency statistics
                value_counts = self.df[var].value_counts()
                value_counts_pct = self.df[var].value_counts(normalize=True) * 100
                
                # Create row for each category
                for category, count in value_counts.items():
                    cat_stats.append({
                        'Variable': var,
                        'Category': category,
                        'Count': count,
                        'Percentage': value_counts_pct[category]
                    })
            
            if cat_stats:
                # Convert to DataFrame
                cat_stats_df = pd.DataFrame(cat_stats)
                
                # Format for display
                headers = ['Variable', 'Category', 'Count', 'Percentage']
                rows = []
                for _, row in cat_stats_df.iterrows():
                    rows.append([
                        row['Variable'],
                        str(row['Category']),
                        f"{int(row['Count'])}",
                        f"{row['Percentage']:.2f}%"
                    ])
                
                # Add to results
                self.results['tables'].append({
                    'title': 'Descriptive Statistics - Categorical Variables',
                    'description': 'Frequency table for categorical variables',
                    'headers': headers,
                    'rows': rows,
                    'csvContent': cat_stats_df.to_csv(index=False)
                })
                
                # Create bar charts for categorical variables
                for var in categorical_vars:
                    # Count plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(x=var, data=self.df, ax=ax)
                    ax.set_title(f'Counts of {var} Categories')
                    ax.set_xlabel(var)
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    self.results['visualizations'].append({
                        'title': f'Counts of {var} Categories',
                        'description': f'Bar chart showing the frequency of each category in {var}',
                        'content': self._fig_to_base64(fig)
                    })
                    plt.close(fig)
                    
                    # Percentage bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    (self.df[var].value_counts(normalize=True) * 100).plot(kind='bar', ax=ax)
                    ax.set_title(f'Percentage of {var} Categories')
                    ax.set_xlabel(var)
                    ax.set_ylabel('Percentage (%)')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    self.results['visualizations'].append({
                        'title': f'Percentage of {var} Categories',
                        'description': f'Bar chart showing the percentage of each category in {var}',
                        'content': self._fig_to_base64(fig)
                    })
                    plt.close(fig)
                    
                    # Pie chart if fewer than 8 categories
                    if self.df[var].nunique() < 8:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        self.df[var].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
                        ax.set_title(f'Distribution of {var}')
                        ax.set_ylabel('')  # Hide the label
                        plt.tight_layout()
                        
                        self.results['visualizations'].append({
                            'title': f'Pie Chart of {var}',
                            'description': f'Pie chart showing the distribution of {var} categories',
                            'content': self._fig_to_base64(fig)
                        })
                        plt.close(fig)
        
        # Add summary text
        summary_text = f"""
## Descriptive Statistics Summary

The analysis includes {len(continuous_vars)} continuous variables and {len(categorical_vars)} categorical variables.

### Continuous Variables
{"  \\n".join([f"- **{var}**: Mean = {self.df[var].mean():.2f}, Median = {self.df[var].median():.2f}, SD = {self.df[var].std():.2f}, Range = {self.df[var].min():.2f} to {self.df[var].max():.2f}" for var in continuous_vars]) if continuous_vars else "No continuous variables analyzed."}

### Categorical Variables
{"  \\n".join([f"- **{var}**: {self.df[var].nunique()} unique categories, Most common: {self.df[var].value_counts().index[0]} ({self.df[var].value_counts().iloc[0]} occurrences, {self.df[var].value_counts(normalize=True).iloc[0]*100:.1f}%)" for var in categorical_vars]) if categorical_vars else "No categorical variables analyzed."}

### Missing Data
{"  \\n".join([f"- **{var}**: {self.df[var].isna().sum()} missing values ({self.df[var].isna().mean()*100:.1f}%)" for var in variables if self.df[var].isna().sum() > 0]) if any(self.df[var].isna().sum() > 0 for var in variables) else "No missing values in the analyzed variables."}
"""
        
        self.results['statistics'].append({
            'title': 'Descriptive Statistics Overview',
            'text': summary_text
        })
        
        return self
    
    def correlation_analysis(self, variables=None, method='pearson'):
        """
        Perform correlation analysis on numeric variables.
        
        Args:
            variables: List of variables to include (None for all numeric)
            method: Correlation method ('pearson', 'spearman', or 'kendall')
            
        Returns:
            self
        """
        # Get numeric columns
        if variables is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        else:
            numeric_cols = [var for var in variables 
                           if var in self.df.columns and pd.api.types.is_numeric_dtype(self.df[var])]
        
        if len(numeric_cols) < 2:
            self.results['statistics'].append({
                'title': 'Correlation Analysis',
                'text': "Correlation analysis requires at least 2 numeric variables."
            })
            return self
        
        # Calculate correlation matrix
        corr_matrix = self.df[numeric_cols].corr(method=method)
        
        # Format for table display
        corr_df = corr_matrix.reset_index()
        corr_df.columns = ['Variable'] + list(corr_matrix.columns)
        
        headers = ['Variable'] + list(corr_matrix.columns)
        rows = []
        
        for _, row in corr_df.iterrows():
            formatted_row = [row['Variable']]
            for col in corr_matrix.columns:
                val = row[col]
                formatted_row.append(f"{val:.3f}")
            rows.append(formatted_row)
        
        # Add to results
        self.results['tables'].append({
            'title': f'Correlation Matrix ({method.capitalize()})',
            'description': f'{method.capitalize()} correlation coefficients between variables',
            'headers': headers,
            'rows': rows,
            'csvContent': corr_matrix.to_csv()
        })
        
        # Create heatmap visualization
        fig, ax = plt.subplots(figsize=(max(8, len(numeric_cols) * 0.8), max(6, len(numeric_cols) * 0.7)))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = 'coolwarm'
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, vmin=-1, vmax=1, 
                   annot=True, fmt='.2f', square=True, linewidths=.5, 
                   cbar_kws={'shrink': .8}, ax=ax)
        
        ax.set_title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Correlation Heatmap ({method.capitalize()})',
            'description': f'Heatmap showing {method} correlation coefficients between variables',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Create scatter plots for variables with strong correlations
        strong_correlations = []
        
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                corr_val = abs(corr_matrix.iloc[i, j])
                
                if corr_val > 0.5:  # Only plot strong correlations
                    strong_correlations.append((col1, col2, corr_matrix.iloc[i, j]))
        
        # Sort by absolute correlation strength
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Plot top 5 strongest correlations
        for col1, col2, corr_val in strong_correlations[:5]:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=col1, y=col2, data=self.df, scatter_kws={'alpha': 0.5}, ax=ax)
            ax.set_title(f'Scatter Plot: {col2} vs {col1} (r = {corr_val:.3f})')
            ax.set_xlabel(col1)
            ax.set_ylabel(col2)
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Scatter Plot: {col2} vs {col1}',
                'description': f'Scatter plot with regression line, correlation = {corr_val:.3f}',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
        
        # Add summary text
        if strong_correlations:
            strongest_pos = max(strong_correlations, key=lambda x: x[2])
            strongest_neg = min(strong_correlations, key=lambda x: x[2])
            
            summary_text = f"""
## Correlation Analysis Summary ({method.capitalize()})

{len(numeric_cols)} variables were included in the correlation analysis.

### Strongest Positive Correlation
- **{strongest_pos[0]} and {strongest_pos[1]}**: r = {strongest_pos[2]:.3f}

### Strongest Negative Correlation
- **{strongest_neg[0]} and {strongest_neg[1]}**: r = {strongest_neg[2]:.3f}

### Strong Correlations (|r| > 0.7)
{"  \\n".join([f"- **{col1} and {col2}**: r = {corr:.3f} ({('positive' if corr > 0 else 'negative')})" for col1, col2, corr in strong_correlations if abs(corr) > 0.7]) if any(abs(corr) > 0.7 for _, _, corr in strong_correlations) else "No strong correlations (|r| > 0.7) found."}

### Moderate Correlations (0.5 < |r| ≤ 0.7)
{"  \\n".join([f"- **{col1} and {col2}**: r = {corr:.3f}" for col1, col2, corr in strong_correlations if 0.5 < abs(corr) <= 0.7]) if any(0.5 < abs(corr) <= 0.7 for _, _, corr in strong_correlations) else "No moderate correlations found in this range."}
"""
        else:
            summary_text = f"""
## Correlation Analysis Summary ({method.capitalize()})

{len(numeric_cols)} variables were included in the correlation analysis.

No strong correlations (|r| > 0.5) were found between the variables.
"""
        
        self.results['statistics'].append({
            'title': 'Correlation Analysis Summary',
            'text': summary_text
        })
        
        return self
    
    def t_test(self, var1, var2=None, grouping_var=None, paired=False, equal_var=True):
        """
        Perform t-test: one-sample, independent samples, or paired samples.
        
        Args:
            var1: Numeric variable
            var2: Second numeric variable (for paired t-test) or None
            grouping_var: Categorical variable for grouping (for independent t-test)
            paired: Whether to perform a paired t-test
            equal_var: Whether to assume equal variances (for independent t-test)
            
        Returns:
            self
        """
        # Check if variables exist in the dataset
        if var1 not in self.df.columns:
            self.results['statistics'].append({
                'title': 'T-Test Error',
                'text': f"Variable '{var1}' not found in the dataset."
            })
            return self
        
        # Check if numeric
        if not pd.api.types.is_numeric_dtype(self.df[var1]):
            self.results['statistics'].append({
                'title': 'T-Test Error',
                'text': f"Variable '{var1}' is not numeric."
            })
            return self
        
        # One-sample t-test
        if var2 is None and grouping_var is None:
            # Perform one-sample t-test against mean=0
            sample = self.df[var1].dropna()
            t_stat, p_value = stats.ttest_1samp(sample, 0)
            
            # Create results table
            self.results['tables'].append({
                'title': f'One-Sample T-Test for {var1}',
                'description': f'Testing if the mean of {var1} differs from 0',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['Sample size', f"{len(sample)}"],
                    ['Mean', f"{sample.mean():.4f}"],
                    ['Std. deviation', f"{sample.std():.4f}"],
                    ['t-statistic', f"{t_stat:.4f}"],
                    ['p-value', f"{p_value:.4f}"]
                ],
                'csvContent': f"Statistic,Value\nSample size,{len(sample)}\nMean,{sample.mean()}\nStd. deviation,{sample.std()}\nt-statistic,{t_stat}\np-value,{p_value}"
            })
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(sample, kde=True, ax=ax)
            ax.axvline(x=0, color='red', linestyle='--', label='Test Value (0)')
            ax.axvline(x=sample.mean(), color='blue', linestyle='-', label=f'Sample Mean ({sample.mean():.2f})')
            ax.set_title(f'Distribution of {var1} (One-Sample T-Test)')
            ax.legend()
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Distribution of {var1}',
                'description': 'Histogram showing the distribution with reference lines for test value and sample mean',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add interpretation
            significance_level = 0.05
            interpretation = f"The one-sample t-test was conducted to determine if the mean of {var1} ({sample.mean():.4f}) differs significantly from 0. "
            
            if p_value < significance_level:
                interpretation += f"With a p-value of {p_value:.4f}, which is less than the significance level of {significance_level}, we reject the null hypothesis and conclude that the mean of {var1} is significantly different from 0."
            else:
                interpretation += f"With a p-value of {p_value:.4f}, which is greater than the significance level of {significance_level}, we fail to reject the null hypothesis and cannot conclude that the mean of {var1} is significantly different from 0."
            
            self.results['statistics'].append({
                'title': 'One-Sample T-Test Analysis',
                'text': f"""
## One-Sample T-Test: {var1}

- **Sample Size**: {len(sample)}
- **Sample Mean**: {sample.mean():.4f}
- **Standard Deviation**: {sample.std():.4f}
- **Test Value**: 0
- **t-statistic**: {t_stat:.4f}
- **p-value**: {p_value:.4f}

### Interpretation
{interpretation}
"""
            })
            
        # Independent samples t-test (with grouping variable)
        elif grouping_var is not None and var2 is None:
            if grouping_var not in self.df.columns:
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': f"Grouping variable '{grouping_var}' not found in the dataset."
                })
                return self
            
            # Check if grouping variable has exactly 2 unique values
            unique_groups = self.df[grouping_var].dropna().unique()
            if len(unique_groups) != 2:
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': f"Grouping variable '{grouping_var}' must have exactly 2 unique values for an independent samples t-test. It has {len(unique_groups)} unique values."
                })
                return self
            
            # Extract the two groups
            group1 = self.df[self.df[grouping_var] == unique_groups[0]][var1].dropna()
            group2 = self.df[self.df[grouping_var] == unique_groups[1]][var1].dropna()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
            
            # Create results table
            self.results['tables'].append({
                'title': f'Independent Samples T-Test: {var1} by {grouping_var}',
                'description': f'Comparing {var1} between {unique_groups[0]} and {unique_groups[1]}',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['t-statistic', f"{t_stat:.4f}"],
                    ['p-value', f"{p_value:.4f}"],
                    [f'Mean of {unique_groups[0]}', f"{group1.mean():.4f}"],
                    [f'Mean of {unique_groups[1]}', f"{group2.mean():.4f}"],
                    [f'Std Dev of {unique_groups[0]}', f"{group1.std():.4f}"],
                    [f'Std Dev of {unique_groups[1]}', f"{group2.std():.4f}"],
                    [f'N of {unique_groups[0]}', f"{len(group1)}"],
                    [f'N of {unique_groups[1]}', f"{len(group2)}"],
                    ['Equal variances assumed', f"{equal_var}"]
                ],
                'csvContent': f"Statistic,Value\nt-statistic,{t_stat}\np-value,{p_value}\nMean of {unique_groups[0]},{group1.mean()}\nMean of {unique_groups[1]},{group2.mean()}"
            })
            
            # Create visualization - Box plot comparing groups
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=grouping_var, y=var1, data=self.df, ax=ax)
            ax.set_title(f'Comparison of {var1} by {grouping_var}')
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Box Plot: {var1} by {grouping_var}',
                'description': 'Box plot comparing the distribution of the variable between groups',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Create visualization - Violin plot for more detailed comparison
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.violinplot(x=grouping_var, y=var1, data=self.df, inner='box', ax=ax)
            ax.set_title(f'Distribution Comparison of {var1} by {grouping_var}')
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Violin Plot: {var1} by {grouping_var}',
                'description': 'Violin plot showing the distribution density of the variable between groups',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add interpretation
            significance_level = 0.05
            mean_diff = group1.mean() - group2.mean()
            
            interpretation = f"An independent samples t-test was conducted to compare {var1} between {unique_groups[0]} (n={len(group1)}) and {unique_groups[1]} (n={len(group2)}). "
            
            if p_value < significance_level:
                interpretation += f"With a p-value of {p_value:.4f}, which is less than the significance level of {significance_level}, we reject the null hypothesis and conclude that there is a statistically significant difference in the means of {var1} between the two groups. The mean difference is {mean_diff:.4f} ({group1.mean():.4f} vs {group2.mean():.4f})."
            else:
                interpretation += f"With a p-value of {p_value:.4f}, which is greater than the significance level of {significance_level}, we fail to reject the null hypothesis and cannot conclude that there is a statistically significant difference in the means of {var1} between the two groups. The mean difference is {mean_diff:.4f} ({group1.mean():.4f} vs {group2.mean():.4f})."
            
            self.results['statistics'].append({
                'title': 'Independent Samples T-Test Analysis',
                'text': f"""
## Independent Samples T-Test: {var1} by {grouping_var}

- **t-statistic**: {t_stat:.4f}
- **p-value**: {p_value:.4f}
- **Mean of {unique_groups[0]}** (n={len(group1)}): {group1.mean():.4f}
- **Mean of {unique_groups[1]}** (n={len(group2)}): {group2.mean():.4f}
- **Mean Difference**: {mean_diff:.4f}
- **Equal variances assumed**: {equal_var}

### Interpretation
{interpretation}
"""
            })
            
        # Paired samples t-test
        elif var2 is not None and paired:
            if var2 not in self.df.columns:
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': f"Variable '{var2}' not found in the dataset."
                })
                return self
                
            if not pd.api.types.is_numeric_dtype(self.df[var2]):
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': f"Variable '{var2}' is not numeric."
                })
                return self
            
            # Get paired samples (drop rows with NaN in either variable)
            paired_data = self.df[[var1, var2]].dropna()
            
            if len(paired_data) < 2:
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': "Not enough paired observations for analysis."
                })
                return self
            
            # Perform paired t-test
            t_stat, p_value = stats.ttest_rel(paired_data[var1], paired_data[var2])
            
            # Calculate difference
            paired_data['difference'] = paired_data[var1] - paired_data[var2]
            
            # Create results table
            self.results['tables'].append({
                'title': f'Paired Samples T-Test: {var1} vs {var2}',
                'description': 'Comparing paired measurements',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['t-statistic', f"{t_stat:.4f}"],
                    ['p-value', f"{p_value:.4f}"],
                    [f'Mean of {var1}', f"{paired_data[var1].mean():.4f}"],
                    [f'Mean of {var2}', f"{paired_data[var2].mean():.4f}"],
                    ['Mean difference', f"{paired_data['difference'].mean():.4f}"],
                    ['Std Dev of difference', f"{paired_data['difference'].std():.4f}"],
                    ['N (pairs)', f"{len(paired_data)}"]
                ],
                'csvContent': f"Statistic,Value\nt-statistic,{t_stat}\np-value,{p_value}\nMean of {var1},{paired_data[var1].mean()}\nMean of {var2},{paired_data[var2].mean()}\nMean difference,{paired_data['difference'].mean()}"
            })
            
            # Create visualization - Box plot of both variables
            fig, ax = plt.subplots(figsize=(10, 6))
            paired_data_long = pd.melt(paired_data[[var1, var2]], value_vars=[var1, var2])
            sns.boxplot(x='variable', y='value', data=paired_data_long, ax=ax)
            ax.set_title(f'Comparison of {var1} and {var2} (Paired)')
            ax.set_xlabel('')
            ax.set_ylabel('Value')
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Box Plot: {var1} vs {var2}',
                'description': 'Box plot comparing the paired variables',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Create visualization - Histogram of differences
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(paired_data['difference'], kde=True, ax=ax)
            ax.axvline(x=0, color='red', linestyle='--', label='Zero Difference')
            ax.axvline(x=paired_data['difference'].mean(), color='blue', linestyle='-', label=f'Mean Difference ({paired_data["difference"].mean():.2f})')
            ax.set_title(f'Distribution of Differences ({var1} - {var2})')
            ax.set_xlabel('Difference')
            ax.legend()
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Distribution of Differences',
                'description': f'Histogram showing the distribution of differences between {var1} and {var2}',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add interpretation
            significance_level = 0.05
            mean_diff = paired_data['difference'].mean()
            
            interpretation = f"A paired samples t-test was conducted to compare {var1} and {var2} across {len(paired_data)} paired observations. "
            
            if p_value < significance_level:
                interpretation += f"With a p-value of {p_value:.4f}, which is less than the significance level of {significance_level}, we reject the null hypothesis and conclude that there is a statistically significant difference between {var1} and {var2}. The mean difference is {mean_diff:.4f} ({paired_data[var1].mean():.4f} vs {paired_data[var2].mean():.4f})."
            else:
                interpretation += f"With a p-value of {p_value:.4f}, which is greater than the significance level of {significance_level}, we fail to reject the null hypothesis and cannot conclude that there is a statistically significant difference between {var1} and {var2}. The mean difference is {mean_diff:.4f} ({paired_data[var1].mean():.4f} vs {paired_data[var2].mean():.4f})."
            
            self.results['statistics'].append({
                'title': 'Paired Samples T-Test Analysis',
                'text': f"""
## Paired Samples T-Test: {var1} vs {var2}

- **t-statistic**: {t_stat:.4f}
- **p-value**: {p_value:.4f}
- **Mean of {var1}**: {paired_data[var1].mean():.4f}
- **Mean of {var2}**: {paired_data[var2].mean():.4f}
- **Mean Difference**: {mean_diff:.4f}
- **Number of Pairs**: {len(paired_data)}

### Interpretation
{interpretation}
"""
            })
            
        # Independent samples t-test (with two separate variables)
        elif var2 is not None and not paired:
            if var2 not in self.df.columns:
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': f"Variable '{var2}' not found in the dataset."
                })
                return self
                
            if not pd.api.types.is_numeric_dtype(self.df[var2]):
                self.results['statistics'].append({
                    'title': 'T-Test Error',
                    'text': f"Variable '{var2}' is not numeric."
                })
                return self
            
            # Get samples (drop rows with NaN)
            sample1 = self.df[var1].dropna()
            sample2 = self.df[var2].dropna()
            
            # Perform independent t-test
            t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
            
            # Create results table
            self.results['tables'].append({
                'title': f'Independent Samples T-Test: {var1} vs {var2}',
                'description': 'Comparing two different variables',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['t-statistic', f"{t_stat:.4f}"],
                    ['p-value', f"{p_value:.4f}"],
                    [f'Mean of {var1}', f"{sample1.mean():.4f}"],
                    [f'Mean of {var2}', f"{sample2.mean():.4f}"],
                    [f'Std Dev of {var1}', f"{sample1.std():.4f}"],
                    [f'Std Dev of {var2}', f"{sample2.std():.4f}"],
                    [f'N of {var1}', f"{len(sample1)}"],
                    [f'N of {var2}', f"{len(sample2)}"],
                    ['Equal variances assumed', f"{equal_var}"]
                ],
                'csvContent': f"Statistic,Value\nt-statistic,{t_stat}\np-value,{p_value}\nMean of {var1},{sample1.mean()}\nMean of {var2},{sample2.mean()}"
            })
            
            # Create visualization - Box plot comparing variables
            fig, ax = plt.subplots(figsize=(10, 6))
            data_to_plot = pd.DataFrame({
                var1: sample1.reset_index(drop=True),
                var2: sample2.reset_index(drop=True)
            })
            data_long = pd.melt(data_to_plot)
            sns.boxplot(x='variable', y='value', data=data_long, ax=ax)
            ax.set_title(f'Comparison of {var1} and {var2}')
            ax.set_xlabel('')
            ax.set_ylabel('Value')
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Box Plot: {var1} vs {var2}',
                'description': 'Box plot comparing the two variables',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add interpretation
            significance_level = 0.05
            mean_diff = sample1.mean() - sample2.mean()
            
            interpretation = f"An independent samples t-test was conducted to compare {var1} (n={len(sample1)}) and {var2} (n={len(sample2)}). "
            
            if p_value < significance_level:
                interpretation += f"With a p-value of {p_value:.4f}, which is less than the significance level of {significance_level}, we reject the null hypothesis and conclude that there is a statistically significant difference between {var1} and {var2}. The mean difference is {mean_diff:.4f} ({sample1.mean():.4f} vs {sample2.mean():.4f})."
            else:
                interpretation += f"With a p-value of {p_value:.4f}, which is greater than the significance level of {significance_level}, we fail to reject the null hypothesis and cannot conclude that there is a statistically significant difference between {var1} and {var2}. The mean difference is {mean_diff:.4f} ({sample1.mean():.4f} vs {sample2.mean():.4f})."
            
            self.results['statistics'].append({
                'title': 'Independent Samples T-Test Analysis',
                'text': f"""
## Independent Samples T-Test: {var1} vs {var2}

- **t-statistic**: {t_stat:.4f}
- **p-value**: {p_value:.4f}
- **Mean of {var1}** (n={len(sample1)}): {sample1.mean():.4f}
- **Mean of {var2}** (n={len(sample2)}): {sample2.mean():.4f}
- **Mean Difference**: {mean_diff:.4f}
- **Equal variances assumed**: {equal_var}

### Interpretation
{interpretation}
"""
            })
        
        return self
    
    def anova(self, dependent_var, grouping_var):
        """
        Perform one-way ANOVA.
        
        Args:
            dependent_var: Numeric dependent variable
            grouping_var: Categorical grouping variable
            
        Returns:
            self
        """
        # Check if variables exist
        if dependent_var not in self.df.columns:
            self.results['statistics'].append({
                'title': 'ANOVA Error',
                'text': f"Dependent variable '{dependent_var}' not found in the dataset."
            })
            return self
            
        if grouping_var not in self.df.columns:
            self.results['statistics'].append({
                'title': 'ANOVA Error',
                'text': f"Grouping variable '{grouping_var}' not found in the dataset."
            })
            return self
        
        # Check if dependent variable is numeric
        if not pd.api.types.is_numeric_dtype(self.df[dependent_var]):
            self.results['statistics'].append({
                'title': 'ANOVA Error',
                'text': f"Dependent variable '{dependent_var}' must be numeric."
            })
            return self
        
        # Get unique groups
        groups = self.df[grouping_var].dropna().unique()
        
        if len(groups) < 2:
            self.results['statistics'].append({
                'title': 'ANOVA Error',
                'text': f"Grouping variable '{grouping_var}' must have at least 2 unique values."
            })
            return self
        
        # Get data for each group
        group_data = []
        group_labels = []
        
        for group in groups:
            data = self.df[self.df[grouping_var] == group][dependent_var].dropna()
            if len(data) > 0:
                group_data.append(data)
                group_labels.append(str(group))
        
        if len(group_data) < 2:
            self.results['statistics'].append({
                'title': 'ANOVA Error',
                'text': "Not enough groups with data for ANOVA."
            })
            return self
        
        # Perform one-way ANOVA
        f_stat, p_value = stats.f_oneway(*group_data)
        
        # Calculate summary statistics for each group
        group_stats = []
        
        for i, group in enumerate(group_labels):
            data = group_data[i]
            group_stats.append({
                'Group': group,
                'N': len(data),
                'Mean': data.mean(),
                'Std Dev': data.std(),
                'Min': data.min(),
                'Max': data.max()
            })
        
        # Create group statistics table
        group_stats_df = pd.DataFrame(group_stats)
        
        headers = ['Group', 'N', 'Mean', 'Std Dev', 'Min', 'Max']
        rows = []
        
        for _, row in group_stats_df.iterrows():
            rows.append([
                row['Group'],
                f"{int(row['N'])}",
                f"{row['Mean']:.4f}",
                f"{row['Std Dev']:.4f}" if not pd.isna(row['Std Dev']) else "N/A",
                f"{row['Min']:.4f}",
                f"{row['Max']:.4f}"
            ])
        
        self.results['tables'].append({
            'title': f'Group Statistics: {dependent_var} by {grouping_var}',
            'description': 'Descriptive statistics for each group',
            'headers': headers,
            'rows': rows,
            'csvContent': group_stats_df.to_csv(index=False)
        })
        
        # Create ANOVA results table
        self.results['tables'].append({
            'title': f'One-way ANOVA: {dependent_var} by {grouping_var}',
            'description': f'Testing for differences in {dependent_var} across {grouping_var} groups',
            'headers': ['Statistic', 'Value'],
            'rows': [
                ['F-statistic', f"{f_stat:.4f}"],
                ['p-value', f"{p_value:.4f}"],
                ['Number of groups', f"{len(group_data)}"]
            ],
            'csvContent': f"Statistic,Value\nF-statistic,{f_stat}\np-value,{p_value}\nNumber of groups,{len(group_data)}"
        })
        
        # Create visualizations
        # Box plot
        fig, ax = plt.subplots(figsize=(max(8, len(group_labels) * 1.2), 6))
        sns.boxplot(x=grouping_var, y=dependent_var, data=self.df, ax=ax)
        ax.set_title(f'Box Plot of {dependent_var} by {grouping_var}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Box Plot: {dependent_var} by {grouping_var}',
            'description': 'Box plot comparing the distribution across groups',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Violin plot
        fig, ax = plt.subplots(figsize=(max(8, len(group_labels) * 1.2), 6))
        sns.violinplot(x=grouping_var, y=dependent_var, data=self.df, inner='box', ax=ax)
        ax.set_title(f'Violin Plot of {dependent_var} by {grouping_var}')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Violin Plot: {dependent_var} by {grouping_var}',
            'description': 'Violin plot showing the distribution density across groups',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Bar plot with error bars
        fig, ax = plt.subplots(figsize=(max(8, len(group_labels) * 1.2), 6))
        sns.barplot(x=grouping_var, y=dependent_var, data=self.df, ax=ax)
        ax.set_title(f'Mean {dependent_var} by {grouping_var} (with 95% CI)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Bar Plot: Mean {dependent_var} by {grouping_var}',
            'description': 'Bar plot showing the mean values with 95% confidence intervals',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Perform post-hoc test (Tukey's HSD)
        posthoc_results = None
        if len(group_data) > 2:
            try:
                # Prepare data for Tukey's test
                tukey_data = self.df[[dependent_var, grouping_var]].dropna()
                tukey_result = pairwise_tukeyhsd(tukey_data[dependent_var], tukey_data[grouping_var], alpha=0.05)
                
                # Format results for display
                posthoc_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
                                        columns=tukey_result._results_table.data[0])
                
                headers = list(posthoc_df.columns)
                rows = []
                
                for _, row in posthoc_df.iterrows():
                    formatted_row = []
                    for col in headers:
                        if col in ['p-adj', 'meandiff']:
                            formatted_row.append(f"{float(row[col]):.4f}")
                        else:
                            formatted_row.append(str(row[col]))
                    rows.append(formatted_row)
                
                self.results['tables'].append({
                    'title': "Tukey's HSD Post-hoc Test Results",
                    'description': 'Pairwise comparisons between groups',
                    'headers': headers,
                    'rows': rows,
                    'csvContent': posthoc_df.to_csv(index=False)
                })
                
                posthoc_results = posthoc_df
            except:
                # If Tukey's test fails, continue without post-hoc results
                pass
        
        # Add interpretation
        significance_level = 0.05
        
        interpretation = f"A one-way ANOVA was conducted to compare the effect of {grouping_var} on {dependent_var} for {len(group_data)} groups. "
        
        if p_value < significance_level:
            interpretation += f"With a p-value of {p_value:.4f}, which is less than the significance level of {significance_level}, we reject the null hypothesis and conclude that there are statistically significant differences in {dependent_var} across the {grouping_var} groups."
            
            # Add post-hoc results if available
            if posthoc_results is not None:
                significant_pairs = posthoc_results[posthoc_results['reject']]
                if len(significant_pairs) > 0:
                    interpretation += f"\n\nTukey's HSD post-hoc test revealed the following significant pairwise differences:"
                    for _, row in significant_pairs.iterrows():
                        interpretation += f"\n- {row['group1']} vs {row['group2']}: Mean difference = {float(row['meandiff']):.4f}, p-adj = {float(row['p-adj']):.4f}"
        else:
            interpretation += f"With a p-value of {p_value:.4f}, which is greater than the significance level of {significance_level}, we fail to reject the null hypothesis and cannot conclude that there are statistically significant differences in {dependent_var} across the {grouping_var} groups."
        
        self.results['statistics'].append({
            'title': 'One-way ANOVA Analysis',
            'text': f"""
## One-way ANOVA: {dependent_var} by {grouping_var}

- **F-statistic**: {f_stat:.4f}
- **p-value**: {p_value:.4f}
- **Number of groups**: {len(group_data)}

### Group Statistics
| Group | N | Mean | Std Dev |
|-------|---|------|---------|
{chr(10).join([f"| {row['Group']} | {int(row['N'])} | {row['Mean']:.4f} | {row['Std Dev']:.4f} |" for _, row in group_stats_df.iterrows()])}

### Interpretation
{interpretation}
"""
        })
        
        return self
    
    def chi_square(self, var1, var2):
        """
        Perform chi-square test of independence.
        
        Args:
            var1: First categorical variable
            var2: Second categorical variable
            
        Returns:
            self
        """
        # Check if variables exist
        if var1 not in self.df.columns:
            self.results['statistics'].append({
                'title': 'Chi-Square Test Error',
                'text': f"Variable '{var1}' not found in the dataset."
            })
            return self
            
        if var2 not in self.df.columns:
            self.results['statistics'].append({
                'title': 'Chi-Square Test Error',
                'text': f"Variable '{var2}' not found in the dataset."
            })
            return self
        
        # Create contingency table
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        
        # Check if contingency table is valid for chi-square
        if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
            self.results['statistics'].append({
                'title': 'Chi-Square Test Error',
                'text': "Both variables must have at least 2 categories for a chi-square test."
            })
            return self
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Create chi-square results table
        self.results['tables'].append({
            'title': f'Chi-Square Test: {var1} vs {var2}',
            'description': 'Test of independence between categorical variables',
            'headers': ['Statistic', 'Value'],
            'rows': [
                ['Chi-square statistic', f"{chi2:.4f}"],
                ['p-value', f"{p_value:.4f}"],
                ['Degrees of freedom', f"{dof}"]
            ],
            'csvContent': f"Statistic,Value\nChi-square statistic,{chi2}\np-value,{p_value}\nDegrees of freedom,{dof}"
        })
        
        # Create contingency table for display
        # Add row and column totals
        contingency_with_totals = contingency_table.copy()
        contingency_with_totals.loc['Total'] = contingency_with_totals.sum()
        contingency_with_totals['Total'] = contingency_with_totals.sum(axis=1)
        
        # Format for display
        headers = [var1] + list(contingency_with_totals.columns)
        rows = []
        
        for idx, row in contingency_with_totals.iterrows():
            formatted_row = [str(idx)]
            for col in contingency_with_totals.columns:
                formatted_row.append(f"{row[col]}")
            rows.append(formatted_row)
        
        self.results['tables'].append({
            'title': 'Contingency Table',
            'description': f'Cross-tabulation of {var1} and {var2}',
            'headers': headers,
            'rows': rows,
            'csvContent': contingency_with_totals.to_csv()
        })
        
        # Create visualizations
        # Stacked bar chart
        fig, ax = plt.subplots(figsize=(max(8, contingency_table.shape[1] * 1.5), 6))
        contingency_table.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Stacked Bar Chart: {var1} vs {var2}')
        ax.set_xlabel(var1)
        ax.set_ylabel('Count')
        plt.legend(title=var2)
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Stacked Bar Chart: {var1} vs {var2}',
            'description': 'Stacked bar chart showing the relationship between variables',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Grouped bar chart (unstacked)
        fig, ax = plt.subplots(figsize=(max(8, contingency_table.shape[1] * 1.5), 6))
        contingency_table.plot(kind='bar', ax=ax)
        ax.set_title(f'Grouped Bar Chart: {var1} vs {var2}')
        ax.set_xlabel(var1)
        ax.set_ylabel('Count')
        plt.legend(title=var2)
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Grouped Bar Chart: {var1} vs {var2}',
            'description': 'Grouped bar chart showing counts for each combination',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Mosaic plot
        from statsmodels.graphics.mosaicplot import mosaic
        
        # Prepare data for mosaic plot
        mosaic_data = self.df[[var1, var2]].dropna()
        
        # Only create mosaic plot if we have reasonable number of categories
        if contingency_table.shape[0] <= 10 and contingency_table.shape[1] <= 10:
            fig, ax = plt.subplots(figsize=(8, 8))
            mosaic(mosaic_data, [var1, var2], ax=ax, title=f'Mosaic Plot: {var1} vs {var2}')
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Mosaic Plot: {var1} vs {var2}',
                'description': 'Mosaic plot showing the relationship between variables with proportional area',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
        
        # Heatmap of the contingency table
        fig, ax = plt.subplots(figsize=(max(8, contingency_table.shape[1] * 1.2), max(6, contingency_table.shape[0] * 1.2)))
        sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
        ax.set_title(f'Heatmap: {var1} vs {var2}')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': f'Heatmap: {var1} vs {var2}',
            'description': 'Heatmap visualization of the contingency table',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Add interpretation
        significance_level = 0.05
        
        # Calculate Cramer's V for effect size
        n = contingency_table.sum().sum()
        phi2 = chi2 / n
        r, k = contingency_table.shape
        cramers_v = np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0
        
        interpretation = f"A chi-square test of independence was performed to examine the relationship between {var1} and {var2}. "
        
        if p_value < significance_level:
            interpretation += f"With a p-value of {p_value:.4f}, which is less than the significance level of {significance_level}, we reject the null hypothesis and conclude that there is a statistically significant association between {var1} and {var2}."
            
            # Add effect size interpretation
            interpretation += f"\n\nCramer's V, which measures the strength of association, is {cramers_v:.4f}. "
            if cramers_v < 0.1:
                interpretation += "This indicates a negligible association."
            elif cramers_v < 0.3:
                interpretation += "This indicates a weak association."
            elif cramers_v < 0.5:
                interpretation += "This indicates a moderate association."
            else:
                interpretation += "This indicates a strong association."
        else:
            interpretation += f"With a p-value of {p_value:.4f}, which is greater than the significance level of {significance_level}, we fail to reject the null hypothesis and cannot conclude that there is a statistically significant association between {var1} and {var2}."
        
        self.results['statistics'].append({
            'title': 'Chi-Square Test Analysis',
            'text': f"""
## Chi-Square Test of Independence: {var1} vs {var2}

- **Chi-square statistic**: {chi2:.4f}
- **p-value**: {p_value:.4f}
- **Degrees of freedom**: {dof}
- **Cramer's V (effect size)**: {cramers_v:.4f}

### Interpretation
{interpretation}

### Contingency Table Summary
The contingency table shows the frequency distribution of {var1} and {var2}. Each cell contains the count of observations with that specific combination of categories.
"""
        })
        
        return self
    
    def regression(self, dependent_var, independent_vars):
        """
        Perform multiple linear regression.
        
        Args:
            dependent_var: Dependent variable
            independent_vars: List of independent variables
            
        Returns:
            self
        """
        # Check if variables exist
        if dependent_var not in self.df.columns:
            self.results['statistics'].append({
                'title': 'Regression Error',
                'text': f"Dependent variable '{dependent_var}' not found in the dataset."
            })
            return self
        
        for var in independent_vars:
            if var not in self.df.columns:
                self.results['statistics'].append({
                    'title': 'Regression Error',
                    'text': f"Independent variable '{var}' not found in the dataset."
                })
                return self
        
        # Check if dependent variable is numeric
        if not pd.api.types.is_numeric_dtype(self.df[dependent_var]):
            self.results['statistics'].append({
                'title': 'Regression Error',
                'text': f"Dependent variable '{dependent_var}' must be numeric."
            })
            return self
        
        # Check if all independent variables are numeric
        non_numeric_vars = [var for var in independent_vars if not pd.api.types.is_numeric_dtype(self.df[var])]
        if non_numeric_vars:
            # For categorical variables, we would need to create dummy variables
            # For simplicity, we'll just report an error in this implementation
            self.results['statistics'].append({
                'title': 'Regression Error',
                'text': f"All independent variables must be numeric. Non-numeric variables: {', '.join(non_numeric_vars)}"
            })
            return self
        
        # Prepare data for regression (drop rows with NaN)
        reg_data = self.df[[dependent_var] + independent_vars].dropna()
        
        if len(reg_data) < len(independent_vars) + 2:
            self.results['statistics'].append({
                'title': 'Regression Error',
                'text': f"Not enough observations for regression with {len(independent_vars)} predictors. Need at least {len(independent_vars) + 2} complete observations."
            })
            return self
        
        # Fit the model
        X = reg_data[independent_vars]
        y = reg_data[dependent_var]
        X = sm.add_constant(X)  # Add constant term
        
        model = sm.OLS(y, X).fit()
        
        # Get summary statistics
        summary = model.summary()
        
        # Create coefficients table
        coef_df = pd.DataFrame({
            'Variable': ['Constant'] + independent_vars,
            'Coefficient': [model.params['const']] + [model.params[var] for var in independent_vars],
            'Std Error': [model.bse['const']] + [model.bse[var] for var in independent_vars],
            't-value': [model.tvalues['const']] + [model.tvalues[var] for var in independent_vars],
            'p-value': [model.pvalues['const']] + [model.pvalues[var] for var in independent_vars]
        })
        
        # Format for display
        headers = ['Variable', 'Coefficient', 'Std Error', 't-value', 'p-value', 'Significance']
        rows = []
        
        for _, row in coef_df.iterrows():
            # Add significance stars
            if row['p-value'] < 0.001:
                sig = '***'
            elif row['p-value'] < 0.01:
                sig = '**'
            elif row['p-value'] < 0.05:
                sig = '*'
            elif row['p-value'] < 0.1:
                sig = '.'
            else:
                sig = ''
                
            rows.append([
                row['Variable'],
                f"{row['Coefficient']:.4f}",
                f"{row['Std Error']:.4f}",
                f"{row['t-value']:.4f}",
                f"{row['p-value']:.4f}",
                sig
            ])
        
        self.results['tables'].append({
            'title': 'Regression Coefficients',
            'description': f'Coefficients for multiple linear regression with {dependent_var} as the dependent variable',
            'headers': headers,
            'rows': rows,
            'csvContent': coef_df.to_csv(index=False)
        })
        
        # Create model summary table
        self.results['tables'].append({
            'title': 'Regression Model Summary',
            'description': 'Overall model statistics',
            'headers': ['Statistic', 'Value'],
            'rows': [
                ['R-squared', f"{model.rsquared:.4f}"],
                ['Adjusted R-squared', f"{model.rsquared_adj:.4f}"],
                ['F-statistic', f"{model.fvalue:.4f}"],
                ['Prob (F-statistic)', f"{model.f_pvalue:.4f}"],
                ['Number of observations', f"{len(reg_data)}"],
                ['AIC', f"{model.aic:.4f}"],
                ['BIC', f"{model.bic:.4f}"]
            ],
            'csvContent': f"Statistic,Value\nR-squared,{model.rsquared}\nAdjusted R-squared,{model.rsquared_adj}\nF-statistic,{model.fvalue}\nProb (F-statistic),{model.f_pvalue}"
        })
        
        # Create visualizations
        # Predicted vs actual values
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(model.predict(), y, alpha=0.5)
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Actual Values')
        ax.set_title(f'Predicted vs Actual Values for {dependent_var}')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': 'Predicted vs Actual Values',
            'description': 'Scatter plot comparing predicted values to actual values',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Residual plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.residplot(x=model.predict(), y=model.resid, lowess=True, ax=ax)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')
        ax.axhline(y=0, color='r', linestyle='-')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': 'Residuals Plot',
            'description': 'Plot of residuals against predicted values to check for patterns',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Q-Q plot of residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(model.resid, plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': 'Q-Q Plot of Residuals',
            'description': 'Q-Q plot to check if residuals follow a normal distribution',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Distribution of residuals
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(model.resid, kde=True, ax=ax)
        ax.set_xlabel('Residuals')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Residuals')
        plt.tight_layout()
        
        self.results['visualizations'].append({
            'title': 'Distribution of Residuals',
            'description': 'Histogram showing the distribution of residuals',
            'content': self._fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Create partial regression plots (for multivariate regression)
        if len(independent_vars) > 1:
            for var in independent_vars:
                fig, ax = plt.subplots(figsize=(10, 6))
                sm.graphics.plot_partregress(endog=dependent_var, exog_i=var, 
                                          exog_others=list(set(independent_vars) - {var}), 
                                          data=reg_data, ax=ax)
                ax.set_title(f'Partial Regression Plot: {var}')
                plt.tight_layout()
                
                self.results['visualizations'].append({
                    'title': f'Partial Regression Plot: {var}',
                    'description': f'Effect of {var} on {dependent_var} controlling for other variables',
                    'content': self._fig_to_base64(fig)
                })
                plt.close(fig)
        
        # Add interpretation
        # Identify significant predictors (p < 0.05)
        significant_vars = coef_df[coef_df['p-value'] < 0.05]['Variable'].tolist()
        if 'Constant' in significant_vars:
            significant_vars.remove('Constant')
        
        # Create equation string
        equation = f"{dependent_var} = {model.params['const']:.4f}"
        for var in independent_vars:
            if model.params[var] >= 0:
                equation += f" + {model.params[var]:.4f} × {var}"
            else:
                equation += f" - {abs(model.params[var]):.4f} × {var}"
        
        interpretation = f"A multiple linear regression was conducted to predict {dependent_var} based on {len(independent_vars)} independent variables. "
        
        if model.f_pvalue < 0.05:
            interpretation += f"The model was statistically significant with F({model.df_model}, {model.df_resid}) = {model.fvalue:.4f}, p < {model.f_pvalue:.4f}. "
            interpretation += f"The model explains {model.rsquared:.2%} of the variance in {dependent_var} (adjusted R² = {model.rsquared_adj:.2%}). "
            
            if significant_vars:
                interpretation += f"\n\nThe following predictors were statistically significant (p < 0.05):"
                for var in significant_vars:
                    coef = model.params[var]
                    p_val = model.pvalues[var]
                    interpretation += f"\n- {var}: β = {coef:.4f}, p = {p_val:.4f}"
                    
                    # Add basic interpretation of coefficient
                    if coef > 0:
                        interpretation += f". For each one-unit increase in {var}, {dependent_var} increases by {coef:.4f} units, holding all other variables constant."
                    else:
                        interpretation += f". For each one-unit increase in {var}, {dependent_var} decreases by {abs(coef):.4f} units, holding all other variables constant."
            else:
                interpretation += f"\n\nNone of the individual predictors were statistically significant at the p < 0.05 level."
        else:
            interpretation += f"The overall model was not statistically significant with F({model.df_model}, {model.df_resid}) = {model.fvalue:.4f}, p = {model.f_pvalue:.4f}. The model explains {model.rsquared:.2%} of the variance in {dependent_var}."
        
        # Add regression equation
        interpretation += f"\n\nThe regression equation is:\n{equation}"
        
        # Add notes about assumptions
        interpretation += """

### Regression Assumptions Check
The validity of a linear regression model depends on several assumptions:

1. **Linearity**: The relationship between the independent and dependent variables should be linear. This can be assessed using the residuals vs predicted values plot.

2. **Independence**: The observations should be independent of each other.

3. **Homoscedasticity**: The residuals should have constant variance across all levels of the independent variables. This can be assessed using the residuals vs predicted values plot.

4. **Normality of residuals**: The residuals should be normally distributed. This can be assessed using the Q-Q plot and histogram of residuals.

5. **No multicollinearity**: The independent variables should not be highly correlated with each other.

Review the diagnostic plots to assess whether these assumptions are met."""
        
        self.results['statistics'].append({
            'title': 'Multiple Linear Regression Analysis',
            'text': f"""
## Multiple Linear Regression: {dependent_var}

### Model Summary
- **R-squared**: {model.rsquared:.4f}
- **Adjusted R-squared**: {model.rsquared_adj:.4f}
- **F-statistic**: {model.fvalue:.4f}
- **p-value (F-statistic)**: {model.f_pvalue:.4f}
- **Number of observations**: {len(reg_data)}

### Regression Equation
{equation}

### Coefficients
| Variable | Coefficient | Std Error | t-value | p-value | Significance |
|----------|-------------|-----------|---------|---------|--------------|
{chr(10).join([f"| {row['Variable']} | {row['Coefficient']:.4f} | {row['Std Error']:.4f} | {row['t-value']:.4f} | {row['p-value']:.4f} | {'***' if row['p-value'] < 0.001 else '**' if row['p-value'] < 0.01 else '*' if row['p-value'] < 0.05 else '.' if row['p-value'] < 0.1 else ''} |" for _, row in coef_df.iterrows()])}

Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

### Interpretation
{interpretation}
"""
        })
        
        return self
    
    def survival_analysis(self, time_var, event_var, group_var=None):
        """
        Perform Kaplan-Meier survival analysis.
        
        Args:
            time_var: Time-to-event variable
            event_var: Event indicator variable (1=event occurred, 0=censored)
            group_var: Grouping variable (optional)
            
        Returns:
            self
        """
        try:
            from lifelines import KaplanMeierFitter, CoxPHFitter
            from lifelines.statistics import logrank_test
        except ImportError:
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': "The lifelines package is required for survival analysis but is not available."
            })
            return self
        
        # Check if variables exist
        if time_var not in self.df.columns:
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': f"Time variable '{time_var}' not found in the dataset."
            })
            return self
            
        if event_var not in self.df.columns:
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': f"Event variable '{event_var}' not found in the dataset."
            })
            return self
        
        # Check if grouping variable exists if provided
        if group_var is not None and group_var not in self.df.columns:
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': f"Grouping variable '{group_var}' not found in the dataset."
            })
            return self
        
        # Check if time variable is numeric
        if not pd.api.types.is_numeric_dtype(self.df[time_var]):
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': f"Time variable '{time_var}' must be numeric."
            })
            return self
        
        # Check if event variable is binary-like
        unique_events = self.df[event_var].dropna().unique()
        if not set(unique_events).issubset({0, 1, True, False, '0', '1', 'True', 'False', 'true', 'false', 'yes', 'no', 'Yes', 'No'}):
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': f"Event variable '{event_var}' must be binary (0/1, True/False, etc.)."
            })
            return self
        
        # Convert event variable to 0/1
        event_map = {
            0: 0, '0': 0, 'False': 0, 'false': 0, False: 0, 'no': 0, 'No': 0,
            1: 1, '1': 1, 'True': 1, 'true': 1, True: 1, 'yes': 1, 'Yes': 1
        }
        
        # Prepare survival data
        surv_data = self.df[[time_var, event_var]].copy()
        surv_data[event_var] = surv_data[event_var].map(event_map)
        
        # Add grouping if provided
        if group_var is not None:
            surv_data[group_var] = self.df[group_var]
        
        # Drop rows with NaN
        surv_data = surv_data.dropna()
        
        if len(surv_data) < 5:
            self.results['statistics'].append({
                'title': 'Survival Analysis Error',
                'text': "Not enough observations for survival analysis."
            })
            return self
        
        # Fit Kaplan-Meier model
        kmf = KaplanMeierFitter()
        
        if group_var is None:
            # Single survival curve
            kmf.fit(surv_data[time_var], surv_data[event_var], label='Overall')
            
            # Extract survival data
            survival_df = kmf.survival_function_.reset_index()
            survival_df.columns = ['Time', 'Survival Probability']
            
            # Extract confidence intervals
            ci_df = kmf.confidence_interval_.reset_index()
            ci_df.columns = ['Time', 'Lower CI', 'Upper CI']
            
            # Merge data
            km_results = pd.merge(survival_df, ci_df, on='Time')
            
            # Create survival table
            headers = ['Time', 'Survival Probability', 'Lower 95% CI', 'Upper 95% CI']
            rows = []
            
            # Select representative points (not all time points to avoid huge tables)
            times_to_show = np.linspace(0, km_results['Time'].max(), min(20, len(km_results)))
            times_idx = np.searchsorted(km_results['Time'], times_to_show)
            times_idx = np.unique(times_idx)
            times_idx = times_idx[times_idx < len(km_results)]
            
            for idx in times_idx:
                row = km_results.iloc[idx]
                rows.append([
                    f"{row['Time']:.2f}",
                    f"{row['Survival Probability']:.4f}",
                    f"{row['Lower CI']:.4f}",
                    f"{row['Upper CI']:.4f}"
                ])
            
            self.results['tables'].append({
                'title': 'Kaplan-Meier Survival Estimates',
                'description': 'Survival probability estimates at different time points',
                'headers': headers,
                'rows': rows,
                'csvContent': km_results.to_csv(index=False)
            })
            
            # Create KM plot
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf.plot_survival_function(ax=ax, ci_show=True)
            ax.set_title('Kaplan-Meier Survival Curve')
            ax.set_xlabel(f'Time ({time_var})')
            ax.set_ylabel('Survival Probability')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': 'Kaplan-Meier Survival Curve',
                'description': 'Survival curve with 95% confidence intervals',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Create cumulative hazard plot
            fig, ax = plt.subplots(figsize=(10, 6))
            kmf.plot_cumulative_hazard(ax=ax, ci_show=True)
            ax.set_title('Cumulative Hazard Function')
            ax.set_xlabel(f'Time ({time_var})')
            ax.set_ylabel('Cumulative Hazard')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': 'Cumulative Hazard Function',
                'description': 'Cumulative hazard function with 95% confidence intervals',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Calculate summary statistics
            median_survival = kmf.median_survival_time_
            
            # For restricted mean survival time, calculate at 75th percentile of observed times
            t_max = np.percentile(surv_data[time_var], 75)
            rmst = kmf.restricted_mean_survival_time(t_max)
            
            # Add survival summary
            self.results['statistics'].append({
                'title': 'Survival Analysis Summary',
                'text': f"""
## Kaplan-Meier Survival Analysis

- **Number of observations**: {len(surv_data)}
- **Number of events**: {surv_data[event_var].sum()}
- **Number of censored**: {len(surv_data) - surv_data[event_var].sum()}
- **Median survival time**: {median_survival:.2f}
- **Restricted mean survival time** (up to t={t_max:.2f}): {rmst:.2f}

### Interpretation
The Kaplan-Meier estimator shows the probability of survival (not experiencing the event) over time. The median survival time is {median_survival:.2f}, which means that at this time, 50% of the subjects have experienced the event.

The restricted mean survival time (RMST) up to t={t_max:.2f} is {rmst:.2f}, which represents the average event-free survival time up to that point.
"""
            })
            
        else:
            # Multiple survival curves by group
            # Get unique groups
            groups = surv_data[group_var].unique()
            
            if len(groups) < 2:
                self.results['statistics'].append({
                    'title': 'Survival Analysis Error',
                    'text': f"Grouping variable '{group_var}' must have at least 2 unique values."
                })
                return self
            
            # Fit KM for each group
            fig, ax = plt.subplots(figsize=(10, 6))
            
            group_stats = []
            km_results_by_group = {}
            
            for group in groups:
                group_data = surv_data[surv_data[group_var] == group]
                kmf.fit(group_data[time_var], group_data[event_var], label=str(group))
                kmf.plot_survival_function(ax=ax, ci_show=True)
                
                # Extract survival data for this group
                surv_df = kmf.survival_function_.reset_index()
                surv_df.columns = ['Time', f'Survival_{group}']
                
                km_results_by_group[group] = surv_df
                
                # Calculate summary statistics for this group
                n_subjects = len(group_data)
                n_events = group_data[event_var].sum()
                median = kmf.median_survival_time_
                
                group_stats.append({
                    'Group': str(group),
                    'N': n_subjects,
                    'Events': n_events,
                    'Censored': n_subjects - n_events,
                    'Median Survival': median
                })
            
            ax.set_title(f'Kaplan-Meier Survival Curves by {group_var}')
            ax.set_xlabel(f'Time ({time_var})')
            ax.set_ylabel('Survival Probability')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Kaplan-Meier Survival Curves by {group_var}',
                'description': 'Survival curves with 95% confidence intervals for each group',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Create cumulative hazard plot by group
            fig, ax = plt.subplots(figsize=(10, 6))
            
            for group in groups:
                group_data = surv_data[surv_data[group_var] == group]
                kmf.fit(group_data[time_var], group_data[event_var], label=str(group))
                kmf.plot_cumulative_hazard(ax=ax, ci_show=True)
            
            ax.set_title(f'Cumulative Hazard Functions by {group_var}')
            ax.set_xlabel(f'Time ({time_var})')
            ax.set_ylabel('Cumulative Hazard')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            self.results['visualizations'].append({
                'title': f'Cumulative Hazard Functions by {group_var}',
                'description': 'Cumulative hazard functions with 95% confidence intervals for each group',
                'content': self._fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Create group statistics table
            group_stats_df = pd.DataFrame(group_stats)
            
            headers = ['Group', 'N', 'Events', 'Censored', 'Median Survival']
            rows = []
            
            for _, row in group_stats_df.iterrows():
                rows.append([
                    row['Group'],
                    f"{int(row['N'])}",
                    f"{int(row['Events'])}",
                    f"{int(row['Censored'])}",
                    f"{row['Median Survival']:.2f}"
                ])
            
            self.results['tables'].append({
                'title': 'Survival Statistics by Group',
                'description': f'Summary statistics for each {group_var} group',
                'headers': headers,
                'rows': rows,
                'csvContent': group_stats_df.to_csv(index=False)
            })
            
            # Perform log-rank test for group comparison
            if len(groups) == 2:
                # For two groups
                g1_data = surv_data[surv_data[group_var] == groups[0]]
                g2_data = surv_data[surv_data[group_var] == groups[1]]
                
                results = logrank_test(g1_data[time_var], g2_data[time_var], 
                                     g1_data[event_var], g2_data[event_var])
                
                test_name = "Log-rank test"
                test_statistic = results.test_statistic
                p_value = results.p_value
            else:
                # For multiple groups, use multivariate log-rank test
                from lifelines.statistics import multivariate_logrank_test
                
                results = multivariate_logrank_test(surv_data[time_var], surv_data[group_var], surv_data[event_var])
                
                test_name = "Multivariate log-rank test"
                test_statistic = results.test_statistic
                p_value = results.p_value
            
            # Create log-rank test results table
            self.results['tables'].append({
                'title': f'{test_name} Results',
                'description': f'Testing for differences in survival between {group_var} groups',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['Test statistic', f"{test_statistic:.4f}"],
                    ['p-value', f"{p_value:.4f}"],
                    ['Degrees of freedom', f"{len(groups) - 1}"]
                ],
                'csvContent': f"Statistic,Value\nTest statistic,{test_statistic}\np-value,{p_value}\nDegrees of freedom,{len(groups) - 1}"
            })
            
            # Add survival summary
            significance_level = 0.05
            
            interpretation = f"The Kaplan-Meier survival analysis was conducted to compare survival rates between {len(groups)} groups of {group_var}. "
            
            if p_value < significance_level:
                interpretation += f"The {test_name.lower()} showed a statistically significant difference in survival distributions between the groups (test statistic = {test_statistic:.4f}, p = {p_value:.4f})."
            else:
                interpretation += f"The {test_name.lower()} did not show a statistically significant difference in survival distributions between the groups (test statistic = {test_statistic:.4f}, p = {p_value:.4f})."
            
            # Add group comparisons
            interpretation += "\n\n### Group Comparison"
            for _, row in group_stats_df.iterrows():
                group = row['Group']
                n = int(row['N'])
                events =
