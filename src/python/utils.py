"""
Utility functions for data analysis and processing in AzzamAnalyst.
"""

import pandas as pd
import numpy as np
import json
import io
import base64
import re
from typing import Dict, List, Union, Any, Tuple, Optional


def convert_to_dataframe(data: Any) -> pd.DataFrame:
    """
    Convert various data formats to pandas DataFrame.
    
    Args:
        data: Data in various formats (list of dicts, dict of lists, DataFrame, etc.)
        
    Returns:
        pandas.DataFrame: Converted data
    """
    if isinstance(data, pd.DataFrame):
        return data.copy()
    
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return pd.DataFrame(data)
    
    if isinstance(data, dict) and all(isinstance(data[key], list) for key in data):
        return pd.DataFrame(data)
    
    if isinstance(data, str):
        # Try to parse as JSON
        try:
            parsed_data = json.loads(data)
            return convert_to_dataframe(parsed_data)
        except json.JSONDecodeError:
            # Try to parse as CSV
            try:
                return pd.read_csv(io.StringIO(data))
            except Exception:
                raise ValueError("Could not parse string data as JSON or CSV")
    
    # If all else fails, try to create a single-row DataFrame
    return pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)


def identify_variable_types(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Identify variable types and characteristics in a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dict containing variable information
    """
    variables_info = {}
    
    for col in df.columns:
        info = {
            'name': col,
            'missing_count': df[col].isna().sum(),
            'missing_percentage': df[col].isna().mean() * 100,
            'unique_values': df[col].nunique()
        }
        
        # Determine variable type
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_count = df[col].nunique()
            if unique_count <= 5:
                info['type'] = 'categorical'
                info['subtype'] = 'ordinal' if unique_count > 2 else 'binary'
            else:
                info['type'] = 'continuous'
                
                # Check normality (for larger samples)
                if len(df[col].dropna()) >= 30:
                    try:
                        from scipy import stats
                        stat, p_value = stats.shapiro(df[col].dropna().sample(min(5000, len(df[col].dropna()))))
                        info['normality'] = {
                            'test': 'shapiro',
                            'p_value': p_value,
                            'is_normal': p_value > 0.05
                        }
                    except:
                        pass
            
            # Add numeric stats
            info.update({
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'median': df[col].median(),
                'std': df[col].std(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })
            
        elif pd.api.types.is_datetime64_dtype(df[col]):
            info['type'] = 'datetime'
            # Add temporal stats
            if not df[col].isna().all():
                info.update({
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'range_days': (df[col].max() - df[col].min()).days if not df[col].isna().all() else None
                })
        else:
            # String/Object columns
            if df[col].nunique() <= 10:
                info['type'] = 'categorical'
                info['subtype'] = 'nominal'
                # Add value counts
                value_counts = df[col].value_counts().head(10).to_dict()
                info['value_counts'] = value_counts
            else:
                info['type'] = 'text'
                
                # Check if it might be an ID column
                if df[col].nunique() == len(df) and df[col].nunique() > 10:
                    info['possible_id'] = True
                
                # Add text stats for string columns
                if df[col].dtype == 'object':
                    non_null_values = df[col].dropna()
                    if len(non_null_values) > 0:
                        try:
                            str_lengths = non_null_values.astype(str).str.len()
                            info['max_length'] = str_lengths.max()
                            info['min_length'] = str_lengths.min()
                            info['mean_length'] = str_lengths.mean()
                        except:
                            pass
        
        variables_info[col] = info
    
    return variables_info


def detect_relationships(df: pd.DataFrame, variables_info: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect potential relationships between variables.
    
    Args:
        df: Input DataFrame
        variables_info: Variable information from identify_variable_types
        
    Returns:
        List of potential relationships
    """
    relationships = []
    
    # Get variable types
    categorical_vars = [col for col, info in variables_info.items() 
                     if info.get('type') == 'categorical']
    continuous_vars = [col for col, info in variables_info.items() 
                     if info.get('type') == 'continuous']
    datetime_vars = [col for col, info in variables_info.items() 
                  if info.get('type') == 'datetime']
    
    # Check continuous vs continuous (correlation)
    if len(continuous_vars) >= 2:
        try:
            corr_matrix = df[continuous_vars].corr()
            for i, col1 in enumerate(continuous_vars):
                for j, col2 in enumerate(continuous_vars):
                    if i < j:  # Only check each pair once
                        corr_value = corr_matrix.loc[col1, col2]
                        if abs(corr_value) >= 0.5:  # Moderate or strong correlation
                            relationships.append({
                                'type': 'correlation',
                                'variables': [col1, col2],
                                'strength': abs(corr_value),
                                'direction': 'positive' if corr_value > 0 else 'negative',
                                'description': f"Strong {'positive' if corr_value > 0 else 'negative'} correlation (r={corr_value:.2f})"
                            })
        except:
            pass
    
    # Check categorical vs continuous (group differences)
    for cat_var in categorical_vars:
        if variables_info[cat_var]['unique_values'] <= 10:  # Reasonable number of groups
            for cont_var in continuous_vars:
                try:
                    # Compute group stats
                    group_stats = df.groupby(cat_var)[cont_var].agg(['mean', 'std']).reset_index()
                    
                    # Check if means differ substantially between groups
                    overall_std = df[cont_var].std()
                    if overall_std > 0:
                        max_mean = group_stats['mean'].max()
                        min_mean = group_stats['mean'].min()
                        if (max_mean - min_mean) / overall_std > 0.5:  # At least 0.5 std dev difference
                            relationships.append({
                                'type': 'group_difference',
                                'variables': [cat_var, cont_var],
                                'description': f"Potential group differences in {cont_var} across {cat_var} categories"
                            })
                except:
                    pass
    
    # Check categorical vs categorical (association)
    if len(categorical_vars) >= 2:
        for i, cat1 in enumerate(categorical_vars):
            if variables_info[cat1]['unique_values'] <= 10:
                for j, cat2 in enumerate(categorical_vars):
                    if i < j and variables_info[cat2]['unique_values'] <= 10:
                        try:
                            # Create contingency table
                            from scipy.stats import chi2_contingency
                            contingency_table = pd.crosstab(df[cat1], df[cat2])
                            
                            # Check for potential association
                            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                                chi2, p, dof, expected = chi2_contingency(contingency_table)
                                if p < 0.05:
                                    relationships.append({
                                        'type': 'association',
                                        'variables': [cat1, cat2],
                                        'description': f"Potential association between {cat1} and {cat2} (Chi-square p={p:.4f})"
                                    })
                        except:
                            pass
    
    # Check time-related patterns (time series analysis)
    for datetime_var in datetime_vars:
        for cont_var in continuous_vars:
            try:
                # Sort by datetime
                temp_df = df[[datetime_var, cont_var]].sort_values(by=datetime_var).dropna()
                if len(temp_df) >= 10:
                    # Check for trend using simple linear regression
                    from scipy import stats
                    # Convert datetime to ordinal for correlation
                    temp_df['time_ordinal'] = pd.to_datetime(temp_df[datetime_var]).map(pd.Timestamp.toordinal)
                    corr, p = stats.pearsonr(temp_df['time_ordinal'], temp_df[cont_var])
                    
                    if abs(corr) >= 0.3:  # Moderate or strong trend
                        relationships.append({
                            'type': 'time_trend',
                            'variables': [datetime_var, cont_var],
                            'strength': abs(corr),
                            'direction': 'increasing' if corr > 0 else 'decreasing',
                            'description': f"{cont_var} shows {'an increasing' if corr > 0 else 'a decreasing'} trend over time (r={corr:.2f})"
                        })
            except:
                pass
    
    return relationships


def find_outliers(df: pd.DataFrame, methods: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Find outliers in numeric columns using multiple methods.
    
    Args:
        df: Input DataFrame
        methods: List of methods to use ('iqr', 'zscore', 'isolation_forest')
        
    Returns:
        Dictionary with outlier information by column and method
    """
    if methods is None:
        methods = ['iqr', 'zscore']
    
    outliers_info = {}
    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    for col in numeric_columns:
        col_data = df[col].dropna()
        
        if len(col_data) <= 1:
            continue
            
        outliers_info[col] = {}
        
        if 'iqr' in methods:
            # IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            if len(outliers) > 0:
                outliers_info[col]['iqr'] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(col_data) * 100,
                    'min': outliers.min(),
                    'max': outliers.max(),
                    'indices': outliers.index.tolist()[:100]  # Limit to first 100 indices
                }
        
        if 'zscore' in methods:
            # Z-score method
            from scipy import stats
            z_scores = np.abs(stats.zscore(col_data, nan_policy='omit'))
            outliers = col_data[z_scores > 3]  # More than 3 standard deviations
            
            if len(outliers) > 0:
                outliers_info[col]['zscore'] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(col_data) * 100,
                    'min': outliers.min(),
                    'max': outliers.max(),
                    'indices': outliers.index.tolist()[:100]  # Limit to first 100 indices
                }
        
        if 'isolation_forest' in methods:
            try:
                # Isolation Forest method
                from sklearn.ensemble import IsolationForest
                
                # Only use isolation forest if enough data points
                if len(col_data) > 30:
                    isolation_model = IsolationForest(contamination=0.05, random_state=42)
                    predictions = isolation_model.fit_predict(col_data.values.reshape(-1, 1))
                    outliers = col_data[predictions == -1]  # -1 indicates outliers
                    
                    if len(outliers) > 0:
                        outliers_info[col]['isolation_forest'] = {
                            'count': len(outliers),
                            'percentage': len(outliers) / len(col_data) * 100,
                            'min': outliers.min(),
                            'max': outliers.max(),
                            'indices': outliers.index.tolist()[:100]  # Limit to first 100 indices
                        }
            except:
                pass
    
    return outliers_info


def check_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Check patterns of missing values in the DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with missing value patterns
    """
    result = {
        'total_missing': df.isna().sum().sum(),
        'total_percentage': (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        'rows_with_missing': df.isna().any(axis=1).sum(),
        'rows_complete': (df.isna().any(axis=1) == False).sum(),
        'columns_with_missing': df.isna().any(axis=0).sum(),
        'columns_complete': (df.isna().any(axis=0) == False).sum(),
        'by_column': {},
        'missing_patterns': []
    }
    
    # Missing by column
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_percentage = (missing_count / len(df)) * 100
        
        if missing_count > 0:
            result['by_column'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_percentage)
            }
    
    # Find common missing patterns (combinations of columns)
    if df.shape[1] <= 30:  # Only for datasets with reasonable number of columns
        try:
            # Create missing indicator matrix
            missing_matrix = df.isna().astype(int)
            
            # Find common patterns
            pattern_counts = missing_matrix.value_counts().head(10)
            
            for pattern, count in pattern_counts.items():
                if isinstance(pattern, tuple) and any(pattern):  # Only include patterns with at least one missing value
                    missing_cols = [df.columns[i] for i, v in enumerate(pattern) if v == 1]
                    
                    if missing_cols:  # Only include patterns with missing values
                        result['missing_patterns'].append({
                            'columns': missing_cols,
                            'count': int(count),
                            'percentage': float(count / len(df) * 100)
                        })
        except:
            pass
    
    return result


def suggest_transformations(df: pd.DataFrame, variables_info: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Suggest potential transformations for variables with skewed distributions.
    
    Args:
        df: Input DataFrame
        variables_info: Variable information from identify_variable_types
        
    Returns:
        Dictionary with suggested transformations by column
    """
    suggestions = {}
    
    for col, info in variables_info.items():
        if info.get('type') == 'continuous':
            col_suggestions = []
            
            # Skip columns with too many missing values
            if info.get('missing_percentage', 0) > 50:
                continue
                
            # Get the data
            data = df[col].dropna()
            
            if len(data) < 5:
                continue
                
            # Check for skewness
            skewness = info.get('skewness')
            
            if skewness is not None and abs(skewness) > 1:
                # Positive skew
                if skewness > 1:
                    # Check if data is strictly positive
                    if data.min() > 0:
                        col_suggestions.append({
                            'transform': 'log',
                            'description': 'Log transformation for positively skewed data',
                            'formula': f'log({col})'
                        })
                        
                        col_suggestions.append({
                            'transform': 'sqrt',
                            'description': 'Square root transformation for positively skewed data',
                            'formula': f'sqrt({col})'
                        })
                    else:
                        # Shift to positive then log transform
                        shift = abs(data.min()) + 1 if data.min() <= 0 else 0
                        col_suggestions.append({
                            'transform': 'log_shifted',
                            'description': 'Shifted log transformation for positively skewed data with zero/negative values',
                            'formula': f'log({col} + {shift})'
                        })
                
                # Negative skew
                elif skewness < -1:
                    col_suggestions.append({
                        'transform': 'square',
                        'description': 'Square transformation for negatively skewed data',
                        'formula': f'{col}²'
                    })
                    
                    # Check if data range allows for this
                    max_val = data.max()
                    if max_val != 0:
                        col_suggestions.append({
                            'transform': 'reciprocal',
                            'description': 'Reciprocal transformation for negatively skewed data',
                            'formula': f'1/{col}'
                        })
            
            # Check for outliers that might suggest transformations
            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1
            upper_bound = q3 + 1.5 * iqr
            
            if (data > upper_bound).sum() > len(data) * 0.05:
                # More than 5% outliers on upper end
                col_suggestions.append({
                    'transform': 'robust_scale',
                    'description': 'Robust scaling to handle outliers',
                    'formula': f'(({col} - median({col})) / IQR({col}))'
                })
                
                if not any(s['transform'] == 'log' for s in col_suggestions) and data.min() > 0:
                    col_suggestions.append({
                        'transform': 'log',
                        'description': 'Log transformation to reduce impact of outliers',
                        'formula': f'log({col})'
                    })
            
            if col_suggestions:
                suggestions[col] = col_suggestions
    
    return suggestions


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a comprehensive summary of a DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary information
    """
    # Get basic information
    summary = {
        'shape': {
            'rows': df.shape[0],
            'columns': df.shape[1]
        },
        'column_types': {
            'numeric': len(df.select_dtypes(include=np.number).columns),
            'categorical': len(df.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(df.select_dtypes(include=['datetime64']).columns),
            'boolean': len(df.select_dtypes(include=['bool']).columns)
        },
        'missing_values': {
            'total': df.isna().sum().sum(),
            'percentage': (df.isna().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
        }
    }
    
    # Check variable types
    variables_info = identify_variable_types(df)
    summary['variables_info'] = variables_info
    
    # Get advanced information
    if df.shape[0] > 0 and df.shape[1] > 0:
        # Check for missing patterns
        summary['missing_patterns'] = check_missing_patterns(df)
        
        # Find outliers in numeric columns
        summary['outliers'] = find_outliers(df)
        
        # Detect potential relationships
        summary['relationships'] = detect_relationships(df, variables_info)
        
        # Suggest transformations
        summary['transformations'] = suggest_transformations(df, variables_info)
    
    return summary


def apply_transformation(df: pd.DataFrame, column: str, transform_type: str) -> pd.DataFrame:
    """
    Apply a transformation to a column in the DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column to transform
        transform_type: Type of transformation to apply
        
    Returns:
        DataFrame with transformed column added
    """
    result_df = df.copy()
    
    # Ensure the column exists and is numeric
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not a numeric column in the DataFrame")
    
    # Get data without NaN values
    data = df[column].dropna()
    
    # Apply the requested transformation
    if transform_type == 'log':
        # Check for non-positive values
        if data.min() <= 0:
            shift = abs(data.min()) + 1
            new_col = np.log(df[column] + shift)
            new_name = f"{column}_log_shifted"
        else:
            new_col = np.log(df[column])
            new_name = f"{column}_log"
    
    elif transform_type == 'sqrt':
        # Check for negative values
        if data.min() < 0:
            shift = abs(data.min()) + 1
            new_col = np.sqrt(df[column] + shift)
            new_name = f"{column}_sqrt_shifted"
        else:
            new_col = np.sqrt(df[column])
            new_name = f"{column}_sqrt"
    
    elif transform_type == 'square':
        new_col = df[column] ** 2
        new_name = f"{column}_squared"
    
    elif transform_type == 'reciprocal':
        # Check for zeros
        if (data == 0).any():
            shift = 1 if data.min() >= 0 else abs(data.min()) + 1
            new_col = 1 / (df[column] + shift)
            new_name = f"{column}_reciprocal_shifted"
        else:
            new_col = 1 / df[column]
            new_name = f"{column}_reciprocal"
    
    elif transform_type == 'z_score':
        new_col = (df[column] - data.mean()) / data.std()
        new_name = f"{column}_zscore"
    
    elif transform_type == 'robust_scale':
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        new_col = (df[column] - data.median()) / iqr
        new_name = f"{column}_robust"
    
    elif transform_type == 'log1p':
        new_col = np.log1p(df[column])
        new_name = f"{column}_log1p"
    
    elif transform_type == 'box_cox':
        from scipy import stats
        # Box-Cox only works for positive data
        if data.min() <= 0:
            shift = abs(data.min()) + 1
            shifted_data = df[column] + shift
            new_col, _ = stats.boxcox(shifted_data.dropna())
            # Put NaNs back in their original positions
            result = pd.Series(index=df.index, data=np.nan)
            result.loc[shifted_data.notna()] = new_col
            new_col = result
            new_name = f"{column}_boxcox_shifted"
        else:
            new_col, _ = stats.boxcox(data)
            # Put NaNs back in their original positions
            result = pd.Series(index=df.index, data=np.nan)
            result.loc[df[column].notna()] = new_col
            new_col = result
            new_name = f"{column}_boxcox"
    
    else:
        raise ValueError(f"Unknown transformation type: {transform_type}")
    
    # Add the transformed column to the DataFrame
    result_df[new_name] = new_col
    
    return result_df


def encode_categorical_variables(df: pd.DataFrame, encoding: str = 'one_hot', max_categories: int = 10) -> pd.DataFrame:
    """
    Encode categorical variables using various encoding methods.
    
    Args:
        df: Input DataFrame
        encoding: Encoding method ('one_hot', 'label', 'target', or 'binary')
        max_categories: Maximum number of categories to encode for one-hot encoding
        
    Returns:
        DataFrame with encoded variables
    """
    result_df = df.copy()
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols:
        return result_df
    
    if encoding == 'one_hot':
        # One-hot encoding (with limit on categories)
        for col in categorical_cols:
            # Skip if too many categories
            if df[col].nunique() > max_categories:
                continue
                
            # Create dummies
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=df[col].isna().any())
            
            # Add dummies to result
            result_df = pd.concat([result_df, dummies], axis=1)
            
            # Remove original column
            result_df = result_df.drop(columns=[col])
    
    elif encoding == 'label':
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Create a new column with encoded values
            not_null = df[col].notna()
            result_df.loc[not_null, f"{col}_encoded"] = le.fit_transform(df.loc[not_null, col])
    
    elif encoding == 'target':
        # Target encoding - requires a target variable
        raise NotImplementedError("Target encoding requires specifying a target variable")
    
    elif encoding == 'binary':
        # Binary encoding
        try:
            import category_encoders as ce
            encoder = ce.BinaryEncoder(cols=categorical_cols)
            encoded = encoder.fit_transform(df[categorical_cols])
            
            # Replace original columns with encoded ones
            result_df = result_df.drop(columns=categorical_cols)
            result_df = pd.concat([result_df, encoded], axis=1)
        except ImportError:
            raise ImportError("category_encoders package is required for binary encoding")
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")
    
    return result_df


def convert_wide_to_long(df: pd.DataFrame, id_vars: List[str], value_vars: List[str], 
                        var_name: str = 'variable', value_name: str = 'value') -> pd.DataFrame:
    """
    Convert DataFrame from wide to long format.
    
    Args:
        df: Input DataFrame in wide format
        id_vars: Columns to use as identifier variables
        value_vars: Columns to unpivot
        var_name: Name for the variable column
        value_name: Name for the value column
        
    Returns:
        DataFrame in long format
    """
    return pd.melt(df, id_vars=id_vars, value_vars=value_vars, 
                  var_name=var_name, value_name=value_name)


def convert_long_to_wide(df: pd.DataFrame, index: Union[str, List[str]], 
                       columns: str, values: str) -> pd.DataFrame:
    """
    Convert DataFrame from long to wide format.
    
    Args:
        df: Input DataFrame in long format
        index: Column(s) to use as index
        columns: Column to use for the new DataFrame's columns
        values: Column to use for the new DataFrame's values
        
    Returns:
        DataFrame in wide format
    """
    return df.pivot(index=index, columns=columns, values=values).reset_index()


def bin_numeric_variable(df: pd.DataFrame, column: str, bins: int = 5, method: str = 'equal_width') -> pd.DataFrame:
    """
    Create bins from a numeric variable.
    
    Args:
        df: Input DataFrame
        column: Numeric column to bin
        bins: Number of bins or bin edges
        method: Binning method ('equal_width', 'equal_frequency', or 'custom')
        
    Returns:
        DataFrame with binned variable added
    """
    result_df = df.copy()
    
    # Check if column exists and is numeric
    if column not in df.columns or not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' is not a numeric column in the DataFrame")
    
    # Apply the requested binning method
    if method == 'equal_width':
        # Equal-width binning
        result_df[f"{column}_binned"] = pd.cut(df[column], bins=bins)
    
    elif method == 'equal_frequency':
        # Equal-frequency binning
        result_df[f"{column}_binned"] = pd.qcut(df[column], q=bins, duplicates='drop')
    
    elif method == 'custom':
        # Custom bin edges
        if not isinstance(bins, (list, np.ndarray)):
            raise ValueError("For 'custom' method, bins must be a list of bin edges")
        
        result_df[f"{column}_binned"] = pd.cut(df[column], bins=bins)
    
    else:
        raise ValueError(f"Unknown binning method: {method}")
    
    return result_df


def detect_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Dict[str, Any]:
    """
    Detect duplicate rows in the DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Columns to consider for identifying duplicates (None for all columns)
        
    Returns:
        Dictionary with duplicate information
    """
    # Find duplicates
    duplicates = df.duplicated(subset=subset, keep=False)
    duplicate_rows = df[duplicates]
    
    # Create result
    result = {
        'count': duplicates.sum(),
        'percentage': (duplicates.sum() / len(df)) * 100,
        'by_group': {},
        'sample': []
    }
    
    # Get sample of duplicates
    if not duplicate_rows.empty:
        # Group duplicates
        if subset is not None:
            dupe_groups = duplicate_rows.groupby(subset).size().reset_index(name='count')
            dupe_groups = dupe_groups.sort_values('count', ascending=False)
            
            # Add top duplicate groups
            for _, row in dupe_groups.head(10).iterrows():
                group_id = tuple(row[col] for col in subset)
                result['by_group'][str(group_id)] = int(row['count'])
        
        # Add sample duplicate rows
        result['sample'] = duplicate_rows.head(5).to_dict(orient='records')
    
    return result


def compute_descriptive_statistics(df: pd.DataFrame, columns: List[str] = None, 
                                 include_percentiles: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Compute comprehensive descriptive statistics for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to compute statistics for (None for all columns)
        include_percentiles: Whether to include additional percentiles
        
    Returns:
        Dictionary with statistics by column
    """
    if columns is None:
        columns = df.columns.tolist()
    
    result = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col]
        col_stats = {
            'count': len(col_data),
            'missing': col_data.isna().sum(),
            'missing_percentage': col_data.isna().mean() * 100
        }
        
        # Compute statistics based on data type
        if pd.api.types.is_numeric_dtype(col_data):
            data = col_data.dropna()
            
            if len(data) > 0:
                # Basic statistics
                col_stats.update({
                    'mean': float(data.mean()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                    'median': float(data.median()),
                    'skewness': float(data.skew()),
                    'kurtosis': float(data.kurtosis())
                })
                
                # Percentiles
                if include_percentiles:
                    percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
                    for p in percentiles:
                        col_stats[f"percentile_{int(p*100)}"] = float(data.quantile(p))
                
                # Mode (might be multiple values)
                modes = data.mode()
                if not modes.empty:
                    col_stats['mode'] = float(modes.iloc[0])
                    col_stats['multimodal'] = len(modes) > 1
        
        elif pd.api.types.is_datetime64_dtype(col_data):
            data = col_data.dropna()
            
            if len(data) > 0:
                # Datetime statistics
                col_stats.update({
                    'min': data.min().isoformat(),
                    'max': data.max().isoformat(),
                    'range_days': (data.max() - data.min()).days
                })
        
        else:
            # Categorical/text statistics
            data = col_data.dropna()
            
            if len(data) > 0:
                # Value counts
                value_counts = data.value_counts()
                
                col_stats.update({
                    'unique_count': data.nunique(),
                    'unique_percentage': data.nunique() / len(data) * 100,
                    'top_value': value_counts.index[0],
                    'top_count': int(value_counts.iloc[0])
                })
                
                # Top values
                top_values = {}
                for val, count in value_counts.head(10).items():
                    top_values[str(val)] = int(count)
                
                col_stats['top_values'] = top_values
        
        result[col] = col_stats
    
    return result


def generate_markdown_report(data_summary: Dict[str, Any]) -> str:
    """
    Generate a markdown report from data summary information.
    
    Args:
        data_summary: Data summary from get_data_summary()
        
    Returns:
        Markdown formatted report
    """
    # Basic dataset information
    report = []
    report.append("# Data Analysis Report\n")
    
    # Dataset overview
    report.append("## Dataset Overview\n")
    shape = data_summary.get('shape', {})
    report.append(f"- **Rows**: {shape.get('rows', 'N/A')}")
    report.append(f"- **Columns**: {shape.get('columns', 'N/A')}")
    
    column_types = data_summary.get('column_types', {})
    report.append(f"- **Column Types**:")
    report.append(f"  - Numeric: {column_types.get('numeric', 0)}")
    report.append(f"  - Categorical: {column_types.get('categorical', 0)}")
    report.append(f"  - Datetime: {column_types.get('datetime', 0)}")
    report.append(f"  - Boolean: {column_types.get('boolean', 0)}")
    
    # Missing values
    missing = data_summary.get('missing_values', {})
    report.append(f"- **Missing Values**: {missing.get('total', 0)} ({missing.get('percentage', 0):.2f}%)")
    
    # Variable information
    report.append("\n## Variable Information\n")
    
    variables_info = data_summary.get('variables_info', {})
    
    # Group variables by type
    continuous_vars = {}
    categorical_vars = {}
    datetime_vars = {}
    text_vars = {}
    
    for var, info in variables_info.items():
        var_type = info.get('type')
        if var_type == 'continuous':
            continuous_vars[var] = info
        elif var_type == 'categorical':
            categorical_vars[var] = info
        elif var_type == 'datetime':
            datetime_vars[var] = info
        elif var_type == 'text':
            text_vars[var] = info
    
    # Continuous variables
    if continuous_vars:
        report.append("### Continuous Variables\n")
        for var, info in continuous_vars.items():
            report.append(f"#### {var}\n")
            report.append(f"- **Missing**: {info.get('missing_count', 0)} ({info.get('missing_percentage', 0):.2f}%)")
            report.append(f"- **Range**: {info.get('min', 'N/A')} to {info.get('max', 'N/A')}")
            report.append(f"- **Mean**: {info.get('mean', 'N/A'):.4f}")
            report.append(f"- **Median**: {info.get('median', 'N/A'):.4f}")
            report.append(f"- **Standard Deviation**: {info.get('std', 'N/A'):.4f}")
            report.append(f"- **Skewness**: {info.get('skewness', 'N/A'):.4f}")
            report.append(f"- **Kurtosis**: {info.get('kurtosis', 'N/A'):.4f}")
            report.append("")
    
    # Categorical variables
    if categorical_vars:
        report.append("### Categorical Variables\n")
        for var, info in categorical_vars.items():
            report.append(f"#### {var}\n")
            report.append(f"- **Missing**: {info.get('missing_count', 0)} ({info.get('missing_percentage', 0):.2f}%)")
            report.append(f"- **Unique Values**: {info.get('unique_values', 0)}")
            
            # Add value counts if available
            value_counts = info.get('value_counts', {})
            if value_counts:
                report.append(f"- **Value Counts**:")
                for value, count in value_counts.items():
                    report.append(f"  - {value}: {count}")
            
            report.append("")
    
    # Datetime variables
    if datetime_vars:
        report.append("### Datetime Variables\n")
        for var, info in datetime_vars.items():
            report.append(f"#### {var}\n")
            report.append(f"- **Missing**: {info.get('missing_count', 0)} ({info.get('missing_percentage', 0):.2f}%)")
            report.append(f"- **Range**: {info.get('min', 'N/A')} to {info.get('max', 'N/A')}")
            report.append(f"- **Range (days)**: {info.get('range_days', 'N/A')}")
            report.append("")
    
    # Relationships
    relationships = data_summary.get('relationships', [])
    if relationships:
        report.append("\n## Detected Relationships\n")
        
        for rel in relationships:
            rel_type = rel.get('type')
            variables = rel.get('variables', [])
            description = rel.get('description', '')
            
            if rel_type == 'correlation':
                strength = rel.get('strength', 0)
                direction = rel.get('direction', '')
                report.append(f"- **Correlation** between `{variables[0]}` and `{variables[1]}`: {strength:.2f} ({direction})")
            elif rel_type == 'group_difference':
                report.append(f"- **Group Difference**: {description}")
            elif rel_type == 'association':
                report.append(f"- **Association**: {description}")
            elif rel_type == 'time_trend':
                report.append(f"- **Time Trend**: {description}")
            else:
                report.append(f"- {description}")
    
    # Outliers
    outliers = data_summary.get('outliers', {})
    if outliers:
        report.append("\n## Outliers\n")
        
        for col, methods in outliers.items():
            report.append(f"### {col}\n")
            
            for method, info in methods.items():
                count = info.get('count', 0)
                percentage = info.get('percentage', 0)
                
                report.append(f"- **{method.upper()}**: {count} outliers ({percentage:.2f}%)")
                report.append(f"  - Range: {info.get('min', 'N/A')} to {info.get('max', 'N/A')}")
            
            report.append("")
    
    # Suggested transformations
    transformations = data_summary.get('transformations', {})
    if transformations:
        report.append("\n## Suggested Transformations\n")
        
        for col, suggestions in transformations.items():
            report.append(f"### {col}\n")
            
            for suggestion in suggestions:
                transform = suggestion.get('transform', '')
                description = suggestion.get('description', '')
                formula = suggestion.get('formula', '')
                
                report.append(f"- **{transform}**: {description}")
                report.append(f"  - Formula: `{formula}`")
            
            report.append("")
    
    # Join all parts
    return "\n".join(report)
