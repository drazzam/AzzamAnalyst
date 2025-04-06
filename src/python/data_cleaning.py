import pandas as pd
import numpy as np
from io import StringIO
import json
import re
import datetime

class DataCleaner:
    """
    A comprehensive data cleaning class for biostatistical datasets.
    Implements a multi-step cleaning pipeline with validation at each stage.
    """
    
    def __init__(self, data, file_type='csv', sheet_name=None):
        """
        Initialize the DataCleaner with data.
        
        Args:
            data: The data to clean, can be a DataFrame or a string containing CSV/Excel data
            file_type: The type of file ('csv', 'excel', etc.)
            sheet_name: Sheet name for Excel files
        """
        self.original_data = data
        self.data = None
        self.file_type = file_type
        self.sheet_name = sheet_name
        self.column_metadata = {}
        self.cleaning_log = []
        
        # Load the data into a DataFrame
        self._load_data()
    
    def _load_data(self):
        """Load the data into a pandas DataFrame."""
        if isinstance(self.original_data, pd.DataFrame):
            self.data = self.original_data.copy()
        elif isinstance(self.original_data, list) and all(isinstance(item, dict) for item in self.original_data):
            # List of dictionaries
            self.data = pd.DataFrame(self.original_data)
        elif isinstance(self.original_data, str):
            # CSV string
            if self.file_type == 'csv':
                self.data = pd.read_csv(StringIO(self.original_data))
            # Excel string (would need to use a package to parse)
            elif self.file_type == 'excel':
                # Excel parsing would need actual file bytes
                pass
        
        # Log initial state
        self.cleaning_log.append({
            'step': 'Initial Load',
            'rows': len(self.data),
            'columns': len(self.data.columns),
            'description': 'Loaded data into DataFrame'
        })
    
    def clean_column_names(self):
        """Clean and standardize column names."""
        old_columns = self.data.columns.tolist()
        
        # Clean column names: lowercase, replace spaces with underscores, remove special chars
        new_columns = []
        for col in old_columns:
            # Convert to string if not already
            col_str = str(col)
            # Replace spaces with underscores and remove special characters
            new_col = re.sub(r'[^\w\s]', '', col_str).lower().replace(' ', '_')
            # Ensure unique by appending numbers if needed
            suffix = 1
            base_col = new_col
            while new_col in new_columns:
                new_col = f"{base_col}_{suffix}"
                suffix += 1
            new_columns.append(new_col)
        
        # Create mapping of old to new column names
        column_mapping = dict(zip(old_columns, new_columns))
        
        # Rename columns
        self.data.columns = new_columns
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Clean Column Names',
            'description': 'Standardized column names',
            'column_mapping': column_mapping
        })
        
        return self
    
    def remove_duplicate_rows(self):
        """Remove duplicate rows from the dataset."""
        row_count_before = len(self.data)
        self.data = self.data.drop_duplicates().reset_index(drop=True)
        row_count_after = len(self.data)
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Remove Duplicates',
            'rows_before': row_count_before,
            'rows_after': row_count_after,
            'rows_removed': row_count_before - row_count_after,
            'description': f'Removed {row_count_before - row_count_after} duplicate rows'
        })
        
        return self
    
    def remove_empty_rows_and_columns(self):
        """Remove entirely empty rows and columns."""
        row_count_before = len(self.data)
        col_count_before = len(self.data.columns)
        
        # Remove rows where all values are NaN
        self.data = self.data.dropna(how='all').reset_index(drop=True)
        
        # Remove columns where all values are NaN
        self.data = self.data.dropna(axis=1, how='all')
        
        row_count_after = len(self.data)
        col_count_after = len(self.data.columns)
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Remove Empty Rows and Columns',
            'rows_before': row_count_before,
            'rows_after': row_count_after,
            'cols_before': col_count_before,
            'cols_after': col_count_after,
            'rows_removed': row_count_before - row_count_after,
            'cols_removed': col_count_before - col_count_after,
            'description': f'Removed {row_count_before - row_count_after} empty rows and {col_count_before - col_count_after} empty columns'
        })
        
        return self
    
    def standardize_missing_values(self):
        """Standardize missing values across the dataset."""
        # List of common missing value indicators
        missing_values = ['', ' ', 'NA', 'N/A', 'NaN', 'nan', 'NULL', 'null', 'None', 'none', '-', '.', '?']
        
        # Replace with NaN
        self.data = self.data.replace(missing_values, np.nan)
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Standardize Missing Values',
            'missing_indicators': missing_values,
            'description': 'Standardized missing values to NaN'
        })
        
        return self
    
    def infer_column_types(self):
        """Infer and convert column types."""
        # Save original types
        original_types = self.data.dtypes.astype(str).to_dict()
        
        # Try to convert string columns to more appropriate types
        for col in self.data.columns:
            # Skip columns that are already numeric
            if pd.api.types.is_numeric_dtype(self.data[col]):
                continue
            
            # Try converting to numeric
            try:
                numeric_series = pd.to_numeric(self.data[col], errors='coerce')
                # If we don't lose too many values, convert
                if numeric_series.notna().sum() >= self.data[col].notna().sum() * 0.8:
                    self.data[col] = numeric_series
                    continue
            except:
                pass
            
            # Try converting to datetime
            try:
                datetime_series = pd.to_datetime(self.data[col], errors='coerce')
                # If we don't lose too many values, convert
                if datetime_series.notna().sum() >= self.data[col].notna().sum() * 0.8:
                    self.data[col] = datetime_series
                    continue
            except:
                pass
            
            # Try to find boolean columns
            if self.data[col].nunique() <= 2:
                # Check for common boolean patterns
                bool_patterns = [
                    ['yes', 'no'],
                    ['true', 'false'],
                    ['t', 'f'],
                    ['y', 'n'],
                    ['1', '0'],
                    [1, 0],
                    [1.0, 0.0]
                ]
                
                # Convert to lowercase for string comparison
                if self.data[col].dtype == 'object':
                    values = self.data[col].dropna().astype(str).str.lower().unique().tolist()
                else:
                    values = self.data[col].dropna().unique().tolist()
                
                for pattern in bool_patterns:
                    if all(val in [str(p).lower() for p in pattern] for val in values):
                        self.data[col] = self.data[col].map({
                            pattern[0]: True, 
                            str(pattern[0]).lower(): True,
                            pattern[1]: False, 
                            str(pattern[1]).lower(): False
                        })
                        break
        
        # Get new types
        new_types = self.data.dtypes.astype(str).to_dict()
        
        # Track changes
        type_changes = {}
        for col in original_types:
            if col in new_types and original_types[col] != new_types[col]:
                type_changes[col] = {
                    'original': original_types[col],
                    'new': new_types[col]
                }
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Infer Column Types',
            'type_changes': type_changes,
            'description': f'Changed data types for {len(type_changes)} columns'
        })
        
        return self
    
    def clean_string_columns(self):
        """Clean string columns: trim whitespace, standardize case."""
        # Process only object columns
        string_columns = self.data.select_dtypes(include=['object']).columns.tolist()
        
        for col in string_columns:
            # Remove leading/trailing whitespace
            if self.data[col].dtype == 'object':
                self.data[col] = self.data[col].astype(str).str.strip()
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Clean String Columns',
            'columns_cleaned': string_columns,
            'description': f'Cleaned {len(string_columns)} string columns'
        })
        
        return self
    
    def detect_outliers(self, method='iqr', threshold=1.5):
        """
        Detect outliers in numeric columns.
        
        Args:
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            Dictionary with outlier information
        """
        outliers_info = {}
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            col_data = self.data[col].dropna()
            
            if len(col_data) <= 1:
                continue
                
            if method == 'iqr':
                q1 = col_data.quantile(0.25)
                q3 = col_data.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            elif method == 'zscore':
                mean = col_data.mean()
                std = col_data.std()
                z_scores = abs((col_data - mean) / std)
                outliers = col_data[z_scores > threshold]
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")
            
            if len(outliers) > 0:
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(col_data) * 100,
                    'min': outliers.min(),
                    'max': outliers.max(),
                    'indices': outliers.index.tolist()
                }
        
        # Log outlier detection
        self.cleaning_log.append({
            'step': 'Detect Outliers',
            'method': method,
            'threshold': threshold,
            'columns_with_outliers': list(outliers_info.keys()),
            'description': f'Detected outliers in {len(outliers_info)} columns'
        })
        
        return outliers_info
    
    def handle_outliers(self, columns=None, method='clip'):
        """
        Handle outliers in numeric columns.
        
        Args:
            columns: List of columns to process (None for all numeric)
            method: Method for handling outliers ('clip', 'remove', or 'impute')
        
        Returns:
            self
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers_handled = {}
        
        for col in columns:
            if col not in self.data.columns or not pd.api.types.is_numeric_dtype(self.data[col]):
                continue
                
            col_data = self.data[col].dropna()
            
            if len(col_data) <= 1:
                continue
                
            # Calculate bounds using IQR method
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Find outliers
            outliers_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            outlier_count = outliers_mask.sum()
            
            if outlier_count > 0:
                if method == 'clip':
                    # Clip values to bounds
                    self.data[col] = self.data[col].clip(lower=lower_bound, upper=upper_bound)
                    action = 'clipped'
                elif method == 'remove':
                    # Set outliers to NaN
                    self.data.loc[outliers_mask, col] = np.nan
                    action = 'removed'
                elif method == 'impute':
                    # Replace with median
                    median = col_data.median()
                    self.data.loc[outliers_mask, col] = median
                    action = 'imputed with median'
                
                outliers_handled[col] = {
                    'count': outlier_count,
                    'percentage': outlier_count / len(self.data) * 100,
                    'action': action
                }
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Handle Outliers',
            'method': method,
            'columns_processed': list(outliers_handled.keys()),
            'description': f'Handled outliers in {len(outliers_handled)} columns using {method} method'
        })
        
        return self
    
    def handle_missing_values(self, strategy='auto'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy: Strategy for handling missing values
                - 'auto': Automatically choose based on column type and missingness
                - 'drop_rows': Drop rows with any missing values
                - 'drop_columns': Drop columns with missing values above a threshold
                - 'mean': Replace missing values with column mean (numeric)
                - 'median': Replace missing values with column median (numeric)
                - 'mode': Replace missing values with column mode (categorical)
                - 'none': Do not handle missing values
        
        Returns:
            self
        """
        # Calculate missingness by column
        missing_by_column = self.data.isnull().sum()
        missing_percentage = missing_by_column / len(self.data) * 100
        
        # Track changes
        missing_handled = {}
        
        if strategy == 'drop_rows':
            # Drop rows with any missing values
            row_count_before = len(self.data)
            self.data = self.data.dropna().reset_index(drop=True)
            row_count_after = len(self.data)
            
            missing_handled['rows_dropped'] = row_count_before - row_count_after
            
        elif strategy == 'drop_columns':
            # Drop columns with more than 50% missing values
            threshold = 50.0  # Percentage
            cols_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
            
            if cols_to_drop:
                self.data = self.data.drop(columns=cols_to_drop)
                missing_handled['columns_dropped'] = cols_to_drop
                
        elif strategy in ['mean', 'median', 'mode']:
            # Impute missing values based on strategy
            for col in self.data.columns:
                missing_count = missing_by_column[col]
                
                if missing_count == 0:
                    continue
                    
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    if strategy == 'mean':
                        fill_value = self.data[col].mean()
                        impute_type = 'mean'
                    elif strategy == 'median':
                        fill_value = self.data[col].median()
                        impute_type = 'median'
                    else:  # mode
                        fill_value = self.data[col].mode()[0]
                        impute_type = 'mode'
                else:
                    # For non-numeric, always use mode
                    fill_value = self.data[col].mode()[0] if not self.data[col].mode().empty else None
                    impute_type = 'mode'
                
                if fill_value is not None:
                    self.data[col] = self.data[col].fillna(fill_value)
                    missing_handled[col] = {
                        'count': missing_count,
                        'percentage': missing_percentage[col],
                        'impute_type': impute_type,
                        'impute_value': str(fill_value)
                    }
                    
        elif strategy == 'auto':
            # Automatically choose strategy based on column type and missingness
            for col in self.data.columns:
                missing_count = missing_by_column[col]
                missing_pct = missing_percentage[col]
                
                if missing_count == 0:
                    continue
                    
                # If more than 80% missing, consider dropping the column
                if missing_pct > 80:
                    self.data = self.data.drop(columns=[col])
                    if 'columns_dropped' not in missing_handled:
                        missing_handled['columns_dropped'] = []
                    missing_handled['columns_dropped'].append(col)
                    continue
                
                # Choose imputation method based on data type
                if pd.api.types.is_numeric_dtype(self.data[col]):
                    # For numeric columns, use median (more robust than mean)
                    fill_value = self.data[col].median()
                    impute_type = 'median'
                else:
                    # For categorical, use mode
                    fill_value = self.data[col].mode()[0] if not self.data[col].mode().empty else None
                    impute_type = 'mode'
                
                if fill_value is not None:
                    self.data[col] = self.data[col].fillna(fill_value)
                    missing_handled[col] = {
                        'count': missing_count,
                        'percentage': missing_pct,
                        'impute_type': impute_type,
                        'impute_value': str(fill_value)
                    }
        
        # Log changes
        self.cleaning_log.append({
            'step': 'Handle Missing Values',
            'strategy': strategy,
            'columns_processed': list(missing_handled.keys()) if isinstance(missing_handled, dict) else [],
            'description': f'Handled missing values using {strategy} strategy'
        })
        
        return self
    
    def identify_variables(self):
        """
        Identify variable types and store metadata.
        """
        for col in self.data.columns:
            meta = {
                'name': col,
                'missing_count': self.data[col].isna().sum(),
                'missing_percentage': self.data[col].isna().mean() * 100,
                'unique_values': self.data[col].nunique()
            }
            
            # Determine variable type
            if pd.api.types.is_numeric_dtype(self.data[col]):
                if self.data[col].dropna().nunique() <= 5:
                    meta['type'] = 'categorical_numeric'
                else:
                    meta['type'] = 'continuous'
                    
                # Add numeric metadata
                meta.update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'median': self.data[col].median(),
                    'std': self.data[col].std()
                })
            elif pd.api.types.is_datetime64_dtype(self.data[col]):
                meta['type'] = 'datetime'
                # Add datetime metadata
                meta.update({
                    'min': self.data[col].min().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(self.data[col].min()) else None,
                    'max': self.data[col].max().strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(self.data[col].max()) else None
                })
            else:
                # Categorical or text
                if self.data[col].dropna().nunique() <= 10:
                    meta['type'] = 'categorical'
                    # Add value counts
                    meta['value_counts'] = self.data[col].value_counts().to_dict()
                else:
                    meta['type'] = 'text'
                    
                # Add length info for string columns
                if self.data[col].dtype == 'object':
                    try:
                        meta['max_length'] = self.data[col].astype(str).str.len().max()
                        meta['min_length'] = self.data[col].astype(str).str.len().min()
                    except:
                        pass
            
            self.column_metadata[col] = meta
        
        # Log variable identification
        self.cleaning_log.append({
            'step': 'Identify Variables',
            'variable_types': {col: meta['type'] for col, meta in self.column_metadata.items()},
            'description': 'Identified variable types and metadata'
        })
        
        return self
    
    def run_cleaning_pipeline(self):
        """Run the full data cleaning pipeline."""
        return (self
                .clean_column_names()
                .remove_duplicate_rows()
                .remove_empty_rows_and_columns()
                .standardize_missing_values()
                .clean_string_columns()
                .infer_column_types()
                .handle_outliers()
                .handle_missing_values(strategy='auto')
                .identify_variables())
    
    def get_cleaning_summary(self):
        """Get a summary of all cleaning operations."""
        return {
            'original_shape': {
                'rows': len(self.original_data) if hasattr(self.original_data, '__len__') else None,
                'columns': len(self.original_data.columns) if hasattr(self.original_data, 'columns') else None
            },
            'cleaned_shape': {
                'rows': len(self.data),
                'columns': len(self.data.columns)
            },
            'cleaning_log': self.cleaning_log,
            'column_metadata': self.column_metadata
        }
    
    def to_dict(self):
        """Convert the cleaned DataFrame to a list of records."""
        return self.data.to_dict('records')
    
    def to_json(self):
        """Convert the cleaned DataFrame to JSON string."""
        return json.dumps(self.data.to_dict('records'))
    
    def to_csv(self):
        """Convert the cleaned DataFrame to CSV string."""
        return self.data.to_csv(index=False)


# Helper function to clean a dataset
def clean_dataset(data, file_type='csv', sheet_name=None):
    """
    Clean a dataset using the DataCleaner.
    
    Args:
        data: The data to clean
        file_type: The type of file
        sheet_name: Sheet name for Excel files
        
    Returns:
        Dictionary containing cleaned data and cleaning summary
    """
    cleaner = DataCleaner(data, file_type=file_type, sheet_name=sheet_name)
    cleaner.run_cleaning_pipeline()
    
    return {
        'cleaned_data': cleaner.to_dict(),
        'summary': cleaner.get_cleaning_summary()
    }
