import { runPython, generateVisualization } from './pyodideService';

// Get the Gemini API key from local storage
const getGeminiApiKey = () => {
  return localStorage.getItem('geminiApiKey');
};

/**
 * Run statistical analysis based on user query and data
 * @param {string} userQuery - User's query
 * @param {Object} processedData - Processed data from uploaded files
 * @param {Function} updateCallback - Callback to update the chat with intermediate results
 * @returns {Promise<Object>} - Analysis results
 */
export const runAnalysis = async (userQuery, processedData, updateCallback) => {
  try {
    // Extract the Gemini API key
    const apiKey = getGeminiApiKey();
    if (!apiKey) {
      throw new Error('API key not found. Please set up your Gemini API key.');
    }

    // Send an initial response to the user
    updateCallback("I'm analyzing your request. This may take a moment...");

    // Step 1: Use Gemini API to interpret the user's request
    const geminiResponse = await interpretUserRequest(userQuery, processedData, apiKey);
    
    // Step 2: Based on the interpretation, generate the appropriate analysis code
    const analysisCode = await generateAnalysisCode(geminiResponse, processedData);
    
    // Step 3: Execute the analysis code using Pyodide
    updateCallback("Running the analysis. Please wait...");
    const analysisResults = await executeAnalysis(analysisCode);
    
    // Step 4: Process and format the results
    const formattedResults = await formatResults(analysisResults, geminiResponse);
    
    // Step 5: Generate a summary of the analysis
    const summary = await generateSummary(formattedResults, userQuery, apiKey);
    
    // Update the chat with the summary
    updateCallback(summary);
    
    // Return the complete results
    return {
      summary: summary,
      ...formattedResults,
      code: analysisCode
    };
  } catch (error) {
    console.error('Error in runAnalysis:', error);
    updateCallback(`I encountered an error while analyzing your request: ${error.message}. Please try again or rephrase your query.`);
    return null;
  }
};

/**
 * Use Gemini API to interpret the user's request
 * @param {string} userQuery - User's query
 * @param {Object} processedData - Processed data
 * @param {string} apiKey - Gemini API key
 * @returns {Promise<Object>} - Interpretation of the request
 */
const interpretUserRequest = async (userQuery, processedData, apiKey) => {
  try {
    // Prepare a summary of the available data
    const datasetSummary = processedData.datasets.map(dataset => ({
      name: dataset.name,
      type: dataset.type,
      rowCount: dataset.rowCount,
      headers: dataset.headers,
      // Sample data (first 5 rows)
      sampleData: dataset.data.slice(0, 5)
    }));
    
    const textContentSummary = processedData.textContent.map(text => ({
      name: text.name,
      type: text.type,
      // First 500 characters as a preview
      preview: text.content.substring(0, 500) + (text.content.length > 500 ? '...' : '')
    }));
    
    // Construct the prompt
    const prompt = `
You are a biostatistical AI assistant. Based on the user's query and the available data, determine what analysis to perform.

USER QUERY: ${userQuery}

AVAILABLE DATA:
${JSON.stringify(datasetSummary, null, 2)}

TEXT CONTENT:
${JSON.stringify(textContentSummary, null, 2)}

Return a structured JSON response with the following fields:
1. "analysisType": The type of analysis to perform (e.g., "descriptive_statistics", "hypothesis_test", "correlation", "regression", "visualization")
2. "dataset": The name of the primary dataset to use
3. "variables": Array of relevant variables/columns to include
4. "additionalParameters": Any specific parameters needed for the analysis
5. "explanation": Brief explanation of what analysis will be performed and why
6. "visualizationType": If visualization is needed, what type (e.g., "histogram", "scatter", "boxplot")

RESPONSE (JSON format only):
`;

    // Call the Gemini API
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0.2,
          maxOutputTokens: 1024,
        }
      })
    });

    const data = await response.json();
    
    if (!data.candidates || data.candidates.length === 0) {
      throw new Error('Failed to get a response from Gemini API');
    }
    
    // Extract and parse the generated content
    const content = data.candidates[0].content.parts[0].text;
    
    // Find the JSON part in the response
    const jsonMatch = content.match(/```json\n([\s\S]*?)\n```/) || 
                      content.match(/```\n([\s\S]*?)\n```/) || 
                      content.match(/{[\s\S]*}/);
    
    let parsedResponse;
    if (jsonMatch) {
      // Try to parse the JSON
      try {
        parsedResponse = JSON.parse(jsonMatch[1] || jsonMatch[0]);
      } catch (e) {
        console.error('Failed to parse Gemini response as JSON:', e);
        // Fallback: try to extract the JSON by finding the first { and last }
        const jsonString = content.substring(
          content.indexOf('{'),
          content.lastIndexOf('}') + 1
        );
        try {
          parsedResponse = JSON.parse(jsonString);
        } catch (e2) {
          console.error('Failed to parse Gemini response with fallback method:', e2);
          throw new Error('Failed to parse the AI response');
        }
      }
    } else {
      throw new Error('Failed to extract JSON from the AI response');
    }
    
    return parsedResponse;
  } catch (error) {
    console.error('Error in interpretUserRequest:', error);
    throw error;
  }
};

/**
 * Generate Python code for the requested analysis
 * @param {Object} interpretation - Interpretation of the user's request
 * @param {Object} processedData - Processed data
 * @returns {Promise<string>} - Python code for the analysis
 */
const generateAnalysisCode = async (interpretation, processedData) => {
  try {
    // Find the requested dataset
    const dataset = processedData.datasets.find(ds => ds.name === interpretation.dataset) || 
                   processedData.datasets[0]; // Use the first dataset if the requested one isn't found
    
    if (!dataset) {
      throw new Error('No dataset available for analysis');
    }
    
    // Basic imports and setup
    let code = `
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import io
import base64
from matplotlib.figure import Figure
import json
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Helper function to convert matplotlib figure to base64 for embedding in HTML
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;height:auto;" />'

# Style settings for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = [10, 6]
plt.rcParams['font.size'] = 12

# Load the dataset
data = ${JSON.stringify(dataset.data)}
df = pd.DataFrame(data)

# Initial data cleaning
# Replace common NA strings with actual NaN values
df = df.replace(['', 'NA', 'N/A', 'NULL', 'null', 'NaN', 'nan'], np.nan)

# Convert numeric columns to appropriate types
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    except:
        pass

# Print dataset information
print(f"Dataset: {${JSON.stringify(dataset.name)}}")
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")
print(f"Columns: {list(df.columns)}")
print("\\nFirst 5 rows:")
print(df.head())

# Initialize results dictionary
results = {
    'tables': [],
    'visualizations': [],
    'statistics': [],
    'summary': ''
}
`;

    // Add analysis code based on the interpretation
    switch (interpretation.analysisType) {
      case 'descriptive_statistics':
        code += generateDescriptiveStatisticsCode(interpretation);
        break;
      case 'hypothesis_test':
        code += generateHypothesisTestCode(interpretation);
        break;
      case 'correlation':
        code += generateCorrelationCode(interpretation);
        break;
      case 'regression':
        code += generateRegressionCode(interpretation);
        break;
      case 'visualization':
        code += generateVisualizationCode(interpretation);
        break;
      default:
        // Default to basic descriptive statistics if the analysis type is unknown
        code += generateDescriptiveStatisticsCode({
          variables: dataset.headers
        });
    }

    // Add code to return the results as JSON
    code += `
# Convert results to JSON
json.dumps(results)
`;

    return code;
  } catch (error) {
    console.error('Error in generateAnalysisCode:', error);
    throw error;
  }
};

/**
 * Generate code for descriptive statistics
 * @param {Object} interpretation - Analysis interpretation
 * @returns {string} - Python code
 */
const generateDescriptiveStatisticsCode = (interpretation) => {
  const variables = interpretation.variables || [];
  const variablesCode = variables.length > 0 
    ? `selected_columns = ${JSON.stringify(variables)}\ndf_selected = df[selected_columns]` 
    : 'df_selected = df';

  return `
# Perform descriptive statistics analysis
${variablesCode}

# Calculate descriptive statistics
desc_stats = df_selected.describe(include='all').transpose()
desc_stats['missing'] = df_selected.isnull().sum()
desc_stats['missing_percent'] = (df_selected.isnull().sum() / len(df_selected) * 100).round(2)
desc_stats['unique_values'] = df_selected.nunique()

# Convert to records for table display
desc_stats_reset = desc_stats.reset_index()
desc_stats_reset.columns = ['Variable'] + list(desc_stats_reset.columns[1:])
desc_stats_records = desc_stats_reset.to_dict('records')

# Create a formatted table for descriptive statistics
headers = ['Variable', 'Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 'Missing', 'Missing %', 'Unique Values']
rows = []

for record in desc_stats_records:
    row = [record['Variable']]
    for header in headers[1:]:
        value = record.get(header.lower(), '')
        if isinstance(value, (int, float)) and not pd.isna(value):
            if header in ['Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']:
                value = f"{value:.2f}"
            elif header == 'Missing %':
                value = f"{value:.2f}%"
            else:
                value = f"{value}"
        elif pd.isna(value):
            value = 'N/A'
        row.append(value)
    rows.append(row)

# Add to results
results['tables'].append({
    'title': 'Descriptive Statistics',
    'description': 'Summary statistics for the selected variables',
    'headers': headers,
    'rows': rows,
    'csvContent': df_selected.describe(include='all').to_csv()
})

# Generate histograms for numeric columns
numeric_cols = df_selected.select_dtypes(include=['number']).columns.tolist()
if numeric_cols:
    for col in numeric_cols:
        if df_selected[col].nunique() > 5:  # Only create histogram for columns with enough unique values
            fig, ax = plt.subplots()
            df_selected[col].dropna().plot(kind='hist', bins=20, ax=ax, alpha=0.7, grid=True)
            ax.set_title(f'Distribution of {col}')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Distribution of {col}',
                'description': f'Histogram showing the frequency distribution of {col}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)

# Generate bar charts for categorical columns
categorical_cols = df_selected.select_dtypes(exclude=['number']).columns.tolist()
for col in categorical_cols:
    if 1 < df_selected[col].nunique() <= 20:  # Only create bar chart for columns with reasonable number of categories
        value_counts = df_selected[col].value_counts().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots()
        value_counts.plot(kind='bar', ax=ax, grid=True)
        ax.set_title(f'Frequency of {col} Categories')
        ax.set_xlabel(col)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add to results
        results['visualizations'].append({
            'title': f'Frequency of {col} Categories',
            'description': f'Bar chart showing the count of each category in {col}',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)

# Generate summary text
results['statistics'].append({
    'title': 'Descriptive Statistics Summary',
    'text': f"""
## Descriptive Statistics Summary

The dataset contains {df_selected.shape[0]} observations and {df_selected.shape[1]} variables.

### Key findings:
- Numeric variables: {len(numeric_cols)}
- Categorical variables: {len(categorical_cols)}
- Variables with missing values: {sum(df_selected.isnull().sum() > 0)}

### Numeric Variables Summary:
{df_selected[numeric_cols].describe().to_markdown() if numeric_cols else "No numeric variables found."}

### Categorical Variables Summary:
{"  \\n".join([f"- {col}: {df_selected[col].nunique()} unique values" for col in categorical_cols]) if categorical_cols else "No categorical variables found."}
"""
})
`;
};

/**
 * Generate code for hypothesis testing
 * @param {Object} interpretation - Analysis interpretation
 * @returns {string} - Python code
 */
const generateHypothesisTestCode = (interpretation) => {
  const variables = interpretation.variables || [];
  const params = interpretation.additionalParameters || {};
  
  return `
# Perform hypothesis testing
variables = ${JSON.stringify(variables)}
test_type = "${params.testType || 't_test'}"

if test_type == 't_test' and len(variables) >= 2:
    # Perform t-test between two variables
    var1 = variables[0]
    var2 = variables[1]
    
    # Check if valid numeric columns
    if var1 in df.columns and var2 in df.columns:
        col1 = df[var1].dropna()
        col2 = df[var2].dropna()
        
        if len(col1) > 0 and len(col2) > 0:
            t_stat, p_value = stats.ttest_ind(col1, col2, nan_policy='omit')
            
            # Create result table
            results['tables'].append({
                'title': f't-Test Results: {var1} vs {var2}',
                'description': 'Independent samples t-test results',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['t-statistic', f"{t_stat:.4f}"],
                    ['p-value', f"{p_value:.4f}"],
                    ['Mean of {var1}', f"{col1.mean():.4f}"],
                    ['Mean of {var2}', f"{col2.mean():.4f}"],
                    ['Std Dev of {var1}', f"{col1.std():.4f}"],
                    ['Std Dev of {var2}', f"{col2.std():.4f}"],
                    ['Sample Size of {var1}', f"{len(col1)}"],
                    ['Sample Size of {var2}', f"{len(col2)}"]
                ],
                'csvContent': f"Statistic,Value\\nt-statistic,{t_stat:.4f}\\np-value,{p_value:.4f}"
            })
            
            # Create comparison visualization
            fig, ax = plt.subplots()
            ax.boxplot([col1, col2], labels=[var1, var2])
            ax.set_title(f'Comparison of {var1} and {var2}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            results['visualizations'].append({
                'title': f'Comparison of {var1} and {var2}',
                'description': 'Box plot comparing the distributions',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add interpretation
            significance_level = 0.05
            interpretation = ""
            if p_value < significance_level:
                interpretation = f"The p-value ({p_value:.4f}) is less than the significance level of {significance_level}, suggesting that there is a statistically significant difference between the means of {var1} and {var2}."
            else:
                interpretation = f"The p-value ({p_value:.4f}) is greater than the significance level of {significance_level}, suggesting that there is not a statistically significant difference between the means of {var1} and {var2}."
            
            results['statistics'].append({
                'title': 't-Test Analysis',
                'text': f"""
## Independent Samples t-Test: {var1} vs {var2}

- **t-statistic**: {t_stat:.4f}
- **p-value**: {p_value:.4f}
- **Mean of {var1}**: {col1.mean():.4f}
- **Mean of {var2}**: {col2.mean():.4f}
- **Sample size of {var1}**: {len(col1)}
- **Sample size of {var2}**: {len(col2)}

### Interpretation
{interpretation}
                """,
                'interpretation': interpretation
            })

elif test_type == 'anova' and len(variables) >= 2:
    # Perform one-way ANOVA
    # Assumes first variable is numeric and second is categorical
    numeric_var = variables[0]
    category_var = variables[1]
    
    if numeric_var in df.columns and category_var in df.columns:
        # Create groups based on the categorical variable
        groups = []
        group_names = []
        
        for category in df[category_var].dropna().unique():
            group_data = df[df[category_var] == category][numeric_var].dropna()
            if len(group_data) > 0:
                groups.append(group_data)
                group_names.append(str(category))
        
        if len(groups) > 1:
            # Perform ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Create result table
            results['tables'].append({
                'title': f'ANOVA Results: {numeric_var} by {category_var}',
                'description': 'One-way ANOVA test results',
                'headers': ['Statistic', 'Value'],
                'rows': [
                    ['F-statistic', f"{f_stat:.4f}"],
                    ['p-value', f"{p_value:.4f}"],
                    ['Number of groups', f"{len(groups)}"]
                ],
                'csvContent': f"Statistic,Value\\nF-statistic,{f_stat:.4f}\\np-value,{p_value:.4f}\\nNumber of groups,{len(groups)}"
            })
            
            # Create box plot visualization
            fig, ax = plt.subplots()
            ax.boxplot(groups, labels=group_names)
            ax.set_title(f'{numeric_var} by {category_var}')
            ax.set_xlabel(category_var)
            ax.set_ylabel(numeric_var)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            results['visualizations'].append({
                'title': f'Box Plot: {numeric_var} by {category_var}',
                'description': 'Box plot comparing the distributions across categories',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add interpretation
            significance_level = 0.05
            interpretation = ""
            if p_value < significance_level:
                interpretation = f"The p-value ({p_value:.4f}) is less than the significance level of {significance_level}, suggesting that there are statistically significant differences in {numeric_var} across the different categories of {category_var}."
            else:
                interpretation = f"The p-value ({p_value:.4f}) is greater than the significance level of {significance_level}, suggesting that there are not statistically significant differences in {numeric_var} across the different categories of {category_var}."
            
            results['statistics'].append({
                'title': 'One-way ANOVA Analysis',
                'text': f"""
## One-way ANOVA: {numeric_var} by {category_var}

- **F-statistic**: {f_stat:.4f}
- **p-value**: {p_value:.4f}
- **Number of groups**: {len(groups)}
- **Group names**: {', '.join(group_names)}

### Interpretation
{interpretation}
                """,
                'interpretation': interpretation
            })

elif test_type == 'chi_square' and len(variables) >= 2:
    # Perform chi-square test of independence
    var1 = variables[0]
    var2 = variables[1]
    
    if var1 in df.columns and var2 in df.columns:
        # Create contingency table
        contingency_table = pd.crosstab(df[var1], df[var2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        # Create result table
        results['tables'].append({
            'title': f'Chi-Square Test: {var1} vs {var2}',
            'description': 'Chi-square test of independence results',
            'headers': ['Statistic', 'Value'],
            'rows': [
                ['Chi-square statistic', f"{chi2:.4f}"],
                ['p-value', f"{p_value:.4f}"],
                ['Degrees of freedom', f"{dof}"]
            ],
            'csvContent': f"Statistic,Value\\nChi-square statistic,{chi2:.4f}\\np-value,{p_value:.4f}\\nDegrees of freedom,{dof}"
        })
        
        # Create contingency table for display
        contingency_df = contingency_table.reset_index()
        headers = [var1] + list(contingency_table.columns)
        rows = []
        
        for _, row in contingency_df.iterrows():
            rows.append([row[var1]] + [row[col] for col in contingency_table.columns])
        
        results['tables'].append({
            'title': 'Contingency Table',
            'description': f'Cross-tabulation of {var1} and {var2}',
            'headers': headers,
            'rows': rows,
            'csvContent': contingency_table.to_csv()
        })
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        contingency_table.plot(kind='bar', stacked=True, ax=ax)
        ax.set_title(f'Contingency Plot: {var1} vs {var2}')
        ax.set_xlabel(var1)
        ax.set_ylabel('Count')
        plt.legend(title=var2)
        plt.tight_layout()
        
        results['visualizations'].append({
            'title': f'Contingency Plot: {var1} vs {var2}',
            'description': 'Stacked bar plot of the contingency table',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Add interpretation
        significance_level = 0.05
        interpretation = ""
        if p_value < significance_level:
            interpretation = f"The p-value ({p_value:.4f}) is less than the significance level of {significance_level}, suggesting that there is a statistically significant association between {var1} and {var2}."
        else:
            interpretation = f"The p-value ({p_value:.4f}) is greater than the significance level of {significance_level}, suggesting that there is not a statistically significant association between {var1} and {var2}."
        
        results['statistics'].append({
            'title': 'Chi-Square Test Analysis',
            'text': f"""
## Chi-Square Test of Independence: {var1} vs {var2}

- **Chi-square statistic**: {chi2:.4f}
- **p-value**: {p_value:.4f}
- **Degrees of freedom**: {dof}

### Contingency Table
{contingency_table.to_markdown()}

### Interpretation
{interpretation}
            """,
            'interpretation': interpretation
        })
`;
};

/**
 * Generate code for correlation analysis
 * @param {Object} interpretation - Analysis interpretation
 * @returns {string} - Python code
 */
const generateCorrelationCode = (interpretation) => {
  const variables = interpretation.variables || [];
  const variablesCode = variables.length > 0 
    ? `selected_columns = ${JSON.stringify(variables)}\ndf_selected = df[selected_columns]` 
    : 'df_selected = df.select_dtypes(include=["number"])';

  return `
# Perform correlation analysis
${variablesCode}

# Check if we have enough numeric columns
if df_selected.shape[1] >= 2:
    # Calculate correlation matrix
    corr_matrix = df_selected.corr(method='pearson')
    
    # Format the correlation matrix for display
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
    results['tables'].append({
        'title': 'Correlation Matrix',
        'description': 'Pearson correlation coefficients between variables',
        'headers': headers,
        'rows': rows,
        'csvContent': corr_matrix.to_csv()
    })
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add variable names as labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns)
    ax.set_yticklabels(corr_matrix.columns)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Correlation Coefficient')
    
    # Add correlation values to cells
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                    ha='center', va='center', color=text_color)
    
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    
    # Add to results
    results['visualizations'].append({
        'title': 'Correlation Heatmap',
        'description': 'Heatmap showing Pearson correlation coefficients between variables',
        'content': fig_to_base64(fig)
    })
    plt.close(fig)
    
    # Create scatter plot matrix for variables with highest correlations
    # Find pairs with highest absolute correlation (excluding self-correlations)
    corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            corr_pairs.append((col1, col2, abs(corr_value)))
    
    # Sort by absolute correlation value
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Create scatter plots for top correlations
    for idx, (col1, col2, corr_value) in enumerate(corr_pairs[:min(5, len(corr_pairs))]):
        fig, ax = plt.subplots()
        ax.scatter(df_selected[col1], df_selected[col2], alpha=0.6)
        
        # Add regression line
        z = np.polyfit(df_selected[col1].dropna(), df_selected[col2].dropna(), 1)
        p = np.poly1d(z)
        ax.plot(df_selected[col1], p(df_selected[col1]), "r--", alpha=0.8)
        
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f'Correlation: {col1} vs {col2} (r = {corr_matrix.loc[col1, col2]:.3f})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add to results
        results['visualizations'].append({
            'title': f'Scatter Plot: {col1} vs {col2}',
            'description': f'Scatter plot with regression line, correlation coefficient: {corr_matrix.loc[col1, col2]:.3f}',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)
    
    # Create summary of strongest correlations
    strong_pos_corrs = []
    strong_neg_corrs = []
    
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            col1 = corr_matrix.columns[i]
            col2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]
            
            if corr_value > 0.6:
                strong_pos_corrs.append((col1, col2, corr_value))
            elif corr_value < -0.6:
                strong_neg_corrs.append((col1, col2, corr_value))
    
    # Add correlation analysis to results
    results['statistics'].append({
        'title': 'Correlation Analysis',
        'text': f"""
## Correlation Analysis

The analysis examined Pearson correlations between {len(corr_matrix.columns)} numeric variables.

### Strong Positive Correlations (r > 0.6)
{"  \\n".join([f"- {col1} and {col2}: r = {corr:.3f}" for col1, col2, corr in strong_pos_corrs]) if strong_pos_corrs else "No strong positive correlations found."}

### Strong Negative Correlations (r < -0.6)
{"  \\n".join([f"- {col1} and {col2}: r = {corr:.3f}" for col1, col2, corr in strong_neg_corrs]) if strong_neg_corrs else "No strong negative correlations found."}

### Correlation Matrix
{corr_matrix.to_markdown()}
"""
    })
else:
    # Not enough numeric variables
    results['statistics'].append({
        'title': 'Correlation Analysis',
        'text': "Correlation analysis requires at least 2 numeric variables. Please select more numeric variables to analyze."
    })
`;
};

/**
 * Generate code for regression analysis
 * @param {Object} interpretation - Analysis interpretation
 * @returns {string} - Python code
 */
const generateRegressionCode = (interpretation) => {
  const variables = interpretation.variables || [];
  const params = interpretation.additionalParameters || {};
  
  // Check if we have dependent and independent variables specified
  const dependentVar = params.dependentVar || (variables.length > 0 ? variables[0] : null);
  const independentVars = params.independentVars || (variables.length > 1 ? variables.slice(1) : []);
  
  if (!dependentVar || independentVars.length === 0) {
    return `
# Not enough variables specified for regression analysis
results['statistics'].append({
    'title': 'Regression Analysis',
    'text': "Regression analysis requires specifying a dependent variable and at least one independent variable. Please select the appropriate variables."
})
`;
  }
  
  return `
# Perform regression analysis
dependent_var = "${dependentVar}"
independent_vars = ${JSON.stringify(independentVars)}

# Check if all variables exist in the dataframe
if dependent_var in df.columns and all(var in df.columns for var in independent_vars):
    # Create a clean dataframe with only the needed variables
    reg_df = df[[dependent_var] + independent_vars].dropna()
    
    # Check if we have enough data
    if len(reg_df) > len(independent_vars) + 1:
        # Build formula for OLS (e.g., "y ~ x1 + x2 + x3")
        formula = f"{dependent_var} ~ " + " + ".join(independent_vars)
        
        # Fit the model
        model = ols(formula, data=reg_df).fit()
        
        # Get summary
        summary = model.summary()
        
        # Create result table with coefficients
        coef_df = pd.DataFrame({
            'Variable': ['Intercept'] + independent_vars,
            'Coefficient': [model.params['Intercept']] + [model.params[var] for var in independent_vars],
            'Std Error': [model.bse['Intercept']] + [model.bse[var] for var in independent_vars],
            't-value': [model.tvalues['Intercept']] + [model.tvalues[var] for var in independent_vars],
            'p-value': [model.pvalues['Intercept']] + [model.pvalues[var] for var in independent_vars]
        })
        
        # Format for display
        rows = []
        for _, row in coef_df.iterrows():
            rows.append([
                row['Variable'],
                f"{row['Coefficient']:.4f}",
                f"{row['Std Error']:.4f}",
                f"{row['t-value']:.4f}",
                f"{row['p-value']:.4f}"
            ])
        
        # Add to results
        results['tables'].append({
            'title': 'Regression Coefficients',
            'description': f'Multiple linear regression results with {dependent_var} as the dependent variable',
            'headers': ['Variable', 'Coefficient', 'Std Error', 't-value', 'p-value'],
            'rows': rows,
            'csvContent': coef_df.to_csv(index=False)
        })
        
        # Add model fit statistics
        results['tables'].append({
            'title': 'Regression Model Fit',
            'description': 'Overall model statistics',
            'headers': ['Statistic', 'Value'],
            'rows': [
                ['R-squared', f"{model.rsquared:.4f}"],
                ['Adjusted R-squared', f"{model.rsquared_adj:.4f}"],
                ['F-statistic', f"{model.fvalue:.4f}"],
                ['p-value (F-statistic)', f"{model.f_pvalue:.4f}"],
                ['Number of observations', f"{model.nobs}"]
            ],
            'csvContent': f"Statistic,Value\\nR-squared,{model.rsquared:.4f}\\nAdjusted R-squared,{model.rsquared_adj:.4f}"
        })
        
        # Create predicted vs actual plot
        fig, ax = plt.subplots()
        ax.scatter(reg_df[dependent_var], model.predict(), alpha=0.6)
        ax.plot([reg_df[dependent_var].min(), reg_df[dependent_var].max()], 
                [reg_df[dependent_var].min(), reg_df[dependent_var].max()], 
                'r--', alpha=0.8)
        ax.set_xlabel(f'Actual {dependent_var}')
        ax.set_ylabel(f'Predicted {dependent_var}')
        ax.set_title('Predicted vs Actual Values')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add to results
        results['visualizations'].append({
            'title': 'Predicted vs Actual Values',
            'description': 'Scatter plot comparing predicted values to actual values',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Create residual plot
        fig, ax = plt.subplots()
        ax.scatter(model.predict(), model.resid, alpha=0.6)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residuals vs Predicted Values')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add to results
        results['visualizations'].append({
            'title': 'Residuals Plot',
            'description': 'Scatter plot of residuals against predicted values',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)
        
        # Create partial dependence plots for each independent variable
        for var in independent_vars:
            fig, ax = plt.subplots()
            ax.scatter(reg_df[var], reg_df[dependent_var], alpha=0.6)
            
            # Sort for line plot
            sorted_df = reg_df.sort_values(by=var)
            
            # Create a temp df for prediction
            pred_df = reg_df.copy()
            
            # Get mean of all other independent variables
            for other_var in [v for v in independent_vars if v != var]:
                pred_df[other_var] = reg_df[other_var].mean()
            
            # Sort by current variable
            pred_df = pred_df.sort_values(by=var)
            
            # Get predictions
            predictions = model.predict(pred_df)
            
            # Plot the partial dependence
            ax.plot(pred_df[var], predictions, 'r-', alpha=0.8)
            
            ax.set_xlabel(var)
            ax.set_ylabel(dependent_var)
            ax.set_title(f'Partial Dependence Plot: {var}')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Partial Dependence Plot: {var}',
                'description': f'Effect of {var} on {dependent_var} while holding other variables constant',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
        
        # Add interpretation based on p-values and R-squared
        significant_vars = coef_df[coef_df['p-value'] < 0.05]['Variable'].tolist()
        if 'Intercept' in significant_vars:
            significant_vars.remove('Intercept')
            
        # Create markdown summary
        results['statistics'].append({
            'title': 'Regression Analysis Results',
            'text': f"""
## Multiple Linear Regression Analysis

### Model Information
- **Dependent Variable**: {dependent_var}
- **Independent Variables**: {', '.join(independent_vars)}
- **Observations**: {model.nobs}

### Model Fit
- **R-squared**: {model.rsquared:.4f}
- **Adjusted R-squared**: {model.rsquared_adj:.4f}
- **F-statistic**: {model.fvalue:.4f}
- **p-value (F-statistic)**: {model.f_pvalue:.4f}

### Coefficients
{coef_df.to_markdown(index=False)}

### Interpretation
The model explains {model.rsquared:.1%} of the variance in {dependent_var}.

{"The overall model is statistically significant (p < 0.05)." if model.f_pvalue < 0.05 else "The overall model is not statistically significant (p > 0.05)."}

#### Significant Predictors:
{"  \\n".join([f"- {var}: Coefficient = {model.params[var]:.4f}, p = {model.pvalues[var]:.4f}" for var in significant_vars]) if significant_vars else "No statistically significant predictors at the 0.05 level."}

#### Prediction Equation:
{dependent_var} = {model.params['Intercept']:.4f} {"".join([f" + {model.params[var]:.4f} × {var}" if model.params[var] > 0 else f" - {abs(model.params[var]):.4f} × {var}" for var in independent_vars])}
"""
        })
    else:
        # Not enough data
        results['statistics'].append({
            'title': 'Regression Analysis',
            'text': f"Not enough data points ({len(reg_df)}) for regression with {len(independent_vars)} predictors. Need at least {len(independent_vars) + 2} complete observations."
        })
else:
    # Variables not found
    results['statistics'].append({
        'title': 'Regression Analysis',
        'text': f"One or more variables not found in the dataset. Please check the variable names."
    })
`;
};

/**
 * Generate code for data visualization
 * @param {Object} interpretation - Analysis interpretation
 * @returns {string} - Python code
 */
const generateVisualizationCode = (interpretation) => {
  const variables = interpretation.variables || [];
  const vizType = interpretation.visualizationType || 'histogram';
  
  let code = '';
  
  switch (vizType) {
    case 'histogram':
      code = `
# Create histogram visualization
variables = ${JSON.stringify(variables)}

# For each specified variable
for var in variables:
    if var in df.columns:
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[var]):
            fig, ax = plt.subplots()
            df[var].dropna().plot(kind='hist', bins=20, ax=ax, alpha=0.7, grid=True)
            ax.set_title(f'Distribution of {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            
            # Add descriptive statistics
            mean_val = df[var].mean()
            median_val = df[var].median()
            std_val = df[var].std()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Distribution of {var}',
                'description': f'Histogram showing the frequency distribution of {var}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add statistics
            results['statistics'].append({
                'title': f'Descriptive Statistics for {var}',
                'text': f"""
## Distribution of {var}

- **Mean**: {mean_val:.4f}
- **Median**: {median_val:.4f}
- **Standard Deviation**: {std_val:.4f}
- **Minimum**: {df[var].min():.4f}
- **Maximum**: {df[var].max():.4f}
- **Range**: {(df[var].max() - df[var].min()):.4f}
- **Skewness**: {df[var].skew():.4f}
- **Kurtosis**: {df[var].kurtosis():.4f}
"""
            })
`;
      break;
      
    case 'scatter':
      code = `
# Create scatter plot visualization
variables = ${JSON.stringify(variables)}

# Check if we have at least two variables
if len(variables) >= 2:
    x_var = variables[0]
    y_var = variables[1]
    
    if x_var in df.columns and y_var in df.columns:
        # Check if both are numeric
        if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):
            fig, ax = plt.subplots()
            ax.scatter(df[x_var], df[y_var], alpha=0.6)
            
            # Add regression line
            if len(df[x_var].dropna()) > 1 and len(df[y_var].dropna()) > 1:
                mask = ~(df[x_var].isna() | df[y_var].isna())
                if mask.sum() > 1:
                    x = df.loc[mask, x_var]
                    y = df.loc[mask, y_var]
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    ax.plot(x, p(x), "r--", alpha=0.8)
                    
                    # Calculate correlation
                    correlation = df[x_var].corr(df[y_var])
                    
                    # Add correlation text
                    plt.annotate(f'r = {correlation:.3f}', 
                                 xy=(0.05, 0.95), 
                                 xycoords='axes fraction',
                                 bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
            
            ax.set_title(f'Scatter Plot: {y_var} vs {x_var}')
            ax.set_xlabel(x_var)
            ax.set_ylabel(y_var)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Scatter Plot: {y_var} vs {x_var}',
                'description': f'Scatter plot showing the relationship between {x_var} and {y_var}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add correlation analysis
            correlation = df[x_var].corr(df[y_var])
            results['statistics'].append({
                'title': f'Correlation Analysis: {x_var} and {y_var}',
                'text': f"""
## Correlation Analysis: {x_var} and {y_var}

- **Pearson Correlation Coefficient**: {correlation:.4f}
- **Strength of Relationship**: {"Very Strong" if abs(correlation) > 0.8 else "Strong" if abs(correlation) > 0.6 else "Moderate" if abs(correlation) > 0.4 else "Weak" if abs(correlation) > 0.2 else "Very Weak"}
- **Direction**: {"Positive" if correlation > 0 else "Negative"}

The scatter plot shows a {"positive" if correlation > 0 else "negative"} {"strong" if abs(correlation) > 0.6 else "moderate" if abs(correlation) > 0.4 else "weak"} relationship between {x_var} and {y_var}.
"""
            })
`;
      break;
      
    case 'boxplot':
      code = `
# Create box plot visualization
variables = ${JSON.stringify(variables)}

# Check if we have the necessary variables
if len(variables) >= 1:
    numeric_var = variables[0]
    
    if len(variables) >= 2:
        # Box plot grouped by a categorical variable
        category_var = variables[1]
        
        if numeric_var in df.columns and category_var in df.columns:
            # Check if numeric_var is numeric
            if pd.api.types.is_numeric_dtype(df[numeric_var]):
                # Get unique categories (limit to top 10 if there are many)
                categories = df[category_var].dropna().unique()
                if len(categories) > 10:
                    # Get the top 10 most frequent categories
                    categories = df[category_var].value_counts().nlargest(10).index.tolist()
                
                # Filter data
                plot_df = df[df[category_var].isin(categories)]
                
                fig, ax = plt.subplots(figsize=(max(8, len(categories) * 0.8), 6))
                plot_df.boxplot(column=numeric_var, by=category_var, ax=ax, grid=False)
                ax.set_title(f'Box Plot of {numeric_var} by {category_var}')
                ax.set_ylabel(numeric_var)
                plt.suptitle('')  # Remove default title
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Add to results
                results['visualizations'].append({
                    'title': f'Box Plot of {numeric_var} by {category_var}',
                    'description': f'Box plot showing the distribution of {numeric_var} across different categories of {category_var}',
                    'content': fig_to_base64(fig)
                })
                plt.close(fig)
                
                # Calculate summary statistics by group
                group_stats = df.groupby(category_var)[numeric_var].agg(['count', 'mean', 'median', 'std', 'min', 'max']).reset_index()
                
                # Format for display
                headers = [category_var, 'Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
                rows = []
                
                for _, row in group_stats.iterrows():
                    formatted_row = [
                        row[category_var],
                        f"{row['count']}",
                        f"{row['mean']:.4f}",
                        f"{row['median']:.4f}",
                        f"{row['std']:.4f}" if not pd.isna(row['std']) else "N/A",
                        f"{row['min']:.4f}",
                        f"{row['max']:.4f}"
                    ]
                    rows.append(formatted_row)
                
                # Add to results
                results['tables'].append({
                    'title': f'Summary Statistics of {numeric_var} by {category_var}',
                    'description': f'Descriptive statistics for {numeric_var} grouped by {category_var}',
                    'headers': headers,
                    'rows': rows,
                    'csvContent': group_stats.to_csv(index=False)
                })
                
                # Add ANOVA analysis if there are at least 2 groups with data
                valid_groups = []
                for category in categories:
                    group_data = df[df[category_var] == category][numeric_var].dropna()
                    if len(group_data) > 0:
                        valid_groups.append(group_data)
                
                if len(valid_groups) >= 2:
                    # Perform ANOVA
                    f_stat, p_value = stats.f_oneway(*valid_groups)
                    
                    # Add statistics
                    results['statistics'].append({
                        'title': f'ANOVA: {numeric_var} by {category_var}',
                        'text': f"""
## One-way ANOVA: {numeric_var} by {category_var}

- **F-statistic**: {f_stat:.4f}
- **p-value**: {p_value:.4f}

### Interpretation
{"There are statistically significant differences in " + numeric_var + " across the categories of " + category_var + " (p < 0.05)." if p_value < 0.05 else "There are no statistically significant differences in " + numeric_var + " across the categories of " + category_var + " (p > 0.05)."}

### Group Summary
{group_stats.to_markdown(index=False)}
"""
                    })
    else:
        # Simple box plot for a single variable
        if numeric_var in df.columns and pd.api.types.is_numeric_dtype(df[numeric_var]):
            fig, ax = plt.subplots()
            df.boxplot(column=numeric_var, ax=ax, grid=False)
            ax.set_title(f'Box Plot of {numeric_var}')
            ax.set_ylabel(numeric_var)
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Box Plot of {numeric_var}',
                'description': f'Box plot showing the distribution of {numeric_var}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Calculate summary statistics
            stats_df = df[numeric_var].describe().reset_index()
            stats_df.columns = ['Statistic', 'Value']
            
            # Format for display
            rows = []
            for _, row in stats_df.iterrows():
                rows.append([
                    row['Statistic'],
                    f"{row['Value']:.4f}" if isinstance(row['Value'], (int, float)) else row['Value']
                ])
            
            # Add outlier information
            q1 = df[numeric_var].quantile(0.25)
            q3 = df[numeric_var].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[numeric_var] < lower_bound) | (df[numeric_var] > upper_bound)][numeric_var]
            
            rows.append(['Outliers Count', f"{len(outliers)}"])
            rows.append(['Outlier Percentage', f"{(len(outliers) / df[numeric_var].count() * 100):.2f}%"])
            
            # Add to results
            results['tables'].append({
                'title': f'Summary Statistics of {numeric_var}',
                'description': f'Descriptive statistics for {numeric_var}',
                'headers': ['Statistic', 'Value'],
                'rows': rows,
                'csvContent': stats_df.to_csv(index=False)
            })
            
            # Add interpretation
            results['statistics'].append({
                'title': f'Box Plot Analysis of {numeric_var}',
                'text': f"""
## Box Plot Analysis of {numeric_var}

### Summary Statistics
- **Median**: {df[numeric_var].median():.4f}
- **IQR (Interquartile Range)**: {iqr:.4f}
- **Lower Quartile (Q1)**: {q1:.4f}
- **Upper Quartile (Q3)**: {q3:.4f}
- **Minimum (non-outlier)**: {max(df[numeric_var].min(), lower_bound):.4f}
- **Maximum (non-outlier)**: {min(df[numeric_var].max(), upper_bound):.4f}

### Outliers
- **Number of Outliers**: {len(outliers)}
- **Percentage of Outliers**: {(len(outliers) / df[numeric_var].count() * 100):.2f}%
- **Outlier Threshold (Lower)**: {lower_bound:.4f}
- **Outlier Threshold (Upper)**: {upper_bound:.4f}

### Distribution Shape
- **Skewness**: {"Positively skewed (right tail)" if df[numeric_var].skew() > 0.5 else "Negatively skewed (left tail)" if df[numeric_var].skew() < -0.5 else "Approximately symmetric"}
"""
            })
`;
      break;
      
    case 'bar':
      code = `
# Create bar chart visualization
variables = ${JSON.stringify(variables)}

# Check if we have the necessary variables
if len(variables) >= 1:
    category_var = variables[0]
    
    if category_var in df.columns:
        # Get value counts
        value_counts = df[category_var].value_counts().nlargest(15)  # Limit to top 15 categories
        
        fig, ax = plt.subplots(figsize=(max(8, len(value_counts) * 0.5), 6))
        value_counts.plot(kind='bar', ax=ax)
        ax.set_title(f'Frequency of {category_var} Categories')
        ax.set_xlabel(category_var)
        ax.set_ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add to results
        results['visualizations'].append({
            'title': f'Bar Chart of {category_var}',
            'description': f'Bar chart showing the frequency of each category in {category_var}',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)
        
        # If we have a second variable, create a grouped bar chart
        if len(variables) >= 2 and variables[1] in df.columns:
            value_var = variables[1]
            
            # Check if value_var is numeric
            if pd.api.types.is_numeric_dtype(df[value_var]):
                # Get top categories (limit to avoid overcrowded plot)
                top_categories = value_counts.index[:10]
                
                # Calculate mean of value_var for each category
                grouped_data = df[df[category_var].isin(top_categories)].groupby(category_var)[value_var].agg(['mean', 'count', 'std']).reset_index()
                grouped_data = grouped_data.sort_values('mean', ascending=False)
                
                fig, ax = plt.subplots(figsize=(max(8, len(grouped_data) * 0.5), 6))
                bars = ax.bar(grouped_data[category_var], grouped_data['mean'], yerr=grouped_data['std'], capsize=5)
                
                # Add data labels
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.2f}', ha='center', va='bottom')
                
                ax.set_title(f'Mean {value_var} by {category_var}')
                ax.set_xlabel(category_var)
                ax.set_ylabel(f'Mean {value_var}')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Add to results
                results['visualizations'].append({
                    'title': f'Mean {value_var} by {category_var}',
                    'description': f'Bar chart showing the mean {value_var} for each category of {category_var}',
                    'content': fig_to_base64(fig)
                })
                plt.close(fig)
                
                # Add table with statistics by category
                headers = [category_var, f'Mean {value_var}', f'Std Dev', 'Count', 'Min', 'Max']
                rows = []
                
                for cat in grouped_data[category_var]:
                    cat_data = df[df[category_var] == cat][value_var]
                    rows.append([
                        cat,
                        f"{cat_data.mean():.4f}",
                        f"{cat_data.std():.4f}" if len(cat_data) > 1 else "N/A",
                        f"{len(cat_data)}",
                        f"{cat_data.min():.4f}",
                        f"{cat_data.max():.4f}"
                    ])
                
                # Add to results
                results['tables'].append({
                    'title': f'Statistics of {value_var} by {category_var}',
                    'description': f'Summary statistics of {value_var} for each category of {category_var}',
                    'headers': headers,
                    'rows': rows,
                    'csvContent': grouped_data.to_csv(index=False)
                })
                
                # Add ANOVA analysis
                valid_groups = []
                valid_group_names = []
                
                for cat in grouped_data[category_var]:
                    group_data = df[df[category_var] == cat][value_var].dropna()
                    if len(group_data) > 0:
                        valid_groups.append(group_data)
                        valid_group_names.append(cat)
                
                if len(valid_groups) >= 2:
                    # Perform ANOVA
                    f_stat, p_value = stats.f_oneway(*valid_groups)
                    
                    # Add statistics
                    results['statistics'].append({
                        'title': f'ANOVA: {value_var} by {category_var}',
                        'text': f"""
## One-way ANOVA: {value_var} by {category_var}

- **F-statistic**: {f_stat:.4f}
- **p-value**: {p_value:.4f}

### Interpretation
{"There are statistically significant differences in the mean " + value_var + " across the categories of " + category_var + " (p < 0.05)." if p_value < 0.05 else "There are no statistically significant differences in the mean " + value_var + " across the categories of " + category_var + " (p > 0.05)."}

### Group Means
{grouped_data.to_markdown(index=False)}
"""
                    })
        
        # Create pie chart for categorical variable
        if df[category_var].nunique() <= 10:  # Only create pie chart if reasonable number of categories
            fig, ax = plt.subplots()
            value_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90, explode=[0.05] * len(value_counts))
            ax.set_title(f'Distribution of {category_var}')
            ax.set_ylabel('')  # Hide the label
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Pie Chart of {category_var}',
                'description': f'Pie chart showing the distribution of categories in {category_var}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
        
        # Add frequency table
        freq_table = df[category_var].value_counts(normalize=True).reset_index()
        freq_table.columns = [category_var, 'Frequency', 'Percentage']
        freq_table['Percentage'] = freq_table['Frequency'] * 100
        freq_table['Cumulative Percentage'] = freq_table['Percentage'].cumsum()
        
        headers = [category_var, 'Count', 'Percentage', 'Cumulative %']
        rows = []
        
        for idx, row in freq_table.iterrows():
            rows.append([
                row[category_var],
                f"{row['Frequency']}",
                f"{row['Percentage']:.2f}%",
                f"{row['Cumulative Percentage']:.2f}%"
            ])
        
        # Add to results
        results['tables'].append({
            'title': f'Frequency Table for {category_var}',
            'description': f'Counts and percentages for each category of {category_var}',
            'headers': headers,
            'rows': rows,
            'csvContent': freq_table.to_csv(index=False)
        })
`;
      break;
      
    case 'line':
      code = `
# Create line chart visualization
variables = ${JSON.stringify(variables)}

# Check if we have necessary variables
if len(variables) >= 2:
    x_var = variables[0]
    y_var = variables[1]
    
    if x_var in df.columns and y_var in df.columns:
        # Sort by x_var
        plot_df = df.sort_values(by=x_var)
        
        fig, ax = plt.subplots()
        ax.plot(plot_df[x_var], plot_df[y_var], marker='o', markersize=4, alpha=0.7)
        ax.set_title(f'{y_var} over {x_var}')
        ax.set_xlabel(x_var)
        ax.set_ylabel(y_var)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Add to results
        results['visualizations'].append({
            'title': f'Line Chart: {y_var} over {x_var}',
            'description': f'Line plot showing the trend of {y_var} across {x_var}',
            'content': fig_to_base64(fig)
        })
        plt.close(fig)
        
        # If we have a third variable for grouping
        if len(variables) >= 3:
            group_var = variables[2]
            
            if group_var in df.columns:
                # Get top groups (limit to avoid overcrowded plot)
                top_groups = df[group_var].value_counts().nlargest(5).index.tolist()
                
                fig, ax = plt.subplots()
                
                for group in top_groups:
                    group_df = df[df[group_var] == group].sort_values(by=x_var)
                    ax.plot(group_df[x_var], group_df[y_var], marker='o', markersize=4, label=str(group), alpha=0.7)
                
                ax.set_title(f'{y_var} over {x_var} by {group_var}')
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.legend(title=group_var)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Add to results
                results['visualizations'].append({
                    'title': f'Line Chart: {y_var} over {x_var} by {group_var}',
                    'description': f'Line plot showing the trend of {y_var} across {x_var} for different {group_var} groups',
                    'content': fig_to_base64(fig)
                })
                plt.close(fig)
        
        # Add basic trend analysis
        if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):
            # Calculate trend (simple linear regression)
            mask = ~(df[x_var].isna() | df[y_var].isna())
            if mask.sum() > 1:
                x = df.loc[mask, x_var]
                y = df.loc[mask, y_var]
                
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                # Add statistics
                results['statistics'].append({
                    'title': f'Trend Analysis: {y_var} over {x_var}',
                    'text': f"""
## Trend Analysis: {y_var} over {x_var}

- **Slope**: {slope:.4f}
- **Intercept**: {intercept:.4f}
- **R-squared**: {r_value**2:.4f}
- **p-value**: {p_value:.4f}

### Interpretation
The {"positive" if slope > 0 else "negative"} slope indicates that as {x_var} increases, {y_var} tends to {"increase" if slope > 0 else "decrease"}.

{"The relationship is statistically significant (p < 0.05)." if p_value < 0.05 else "The relationship is not statistically significant (p > 0.05)."}

The R-squared value of {r_value**2:.4f} indicates that {r_value**2*100:.1f}% of the variation in {y_var} can be explained by changes in {x_var}.

### Equation
{y_var} = {slope:.4f} × {x_var} + {intercept:.4f}
"""
                })
                
                # Create scatter plot with trend line
                fig, ax = plt.subplots()
                ax.scatter(x, y, alpha=0.6)
                
                # Add regression line
                x_range = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_range, intercept + slope * x_range, 'r--', 
                        label=f'y = {slope:.3f}x + {intercept:.3f} (r²={r_value**2:.3f})')
                
                ax.set_title(f'Scatter Plot with Trend Line: {y_var} vs {x_var}')
                ax.set_xlabel(x_var)
                ax.set_ylabel(y_var)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                
                # Add to results
                results['visualizations'].append({
                    'title': f'Scatter Plot with Trend Line: {y_var} vs {x_var}',
                    'description': f'Scatter plot with regression line showing the relationship between {x_var} and {y_var}',
                    'content': fig_to_base64(fig)
                })
                plt.close(fig)
`;
      break;
      
    default:
      code = `
# Create general descriptive visualizations
variables = ${JSON.stringify(variables)}

# For each specified variable
for var in variables:
    if var in df.columns:
        # Check data type and create appropriate visualization
        if pd.api.types.is_numeric_dtype(df[var]):
            # For numeric variables, create histogram
            fig, ax = plt.subplots()
            df[var].dropna().plot(kind='hist', bins=20, ax=ax, alpha=0.7, grid=True)
            ax.set_title(f'Distribution of {var}')
            ax.set_xlabel(var)
            ax.set_ylabel('Frequency')
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Distribution of {var}',
                'description': f'Histogram showing the frequency distribution of {var}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)`;
            
            # Add descriptive statistics
            desc_stats = df[var].describe()
            results['statistics'].append({
                'title': f'Descriptive Statistics for {var}',
                'text': f"""
## Descriptive Statistics for {var}

- **Mean**: {desc_stats['mean']:.4f}
- **Median**: {df[var].median():.4f}
- **Standard Deviation**: {desc_stats['std']:.4f}
- **Minimum**: {desc_stats['min']:.4f}
- **Maximum**: {desc_stats['max']:.4f}
- **25th Percentile**: {desc_stats['25%']:.4f}
- **75th Percentile**: {desc_stats['75%']:.4f}
"""
            })
        else:
            # For categorical variables, create bar chart
            value_counts = df[var].value_counts().nlargest(15)  # Limit to top 15 categories
            
            fig, ax = plt.subplots()
            value_counts.plot(kind='bar', ax=ax)
            ax.set_title(f'Frequency of {var} Categories')
            ax.set_xlabel(var)
            ax.set_ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Add to results
            results['visualizations'].append({
                'title': f'Bar Chart of {var} Categories',
                'description': f'Bar chart showing the count of each category in {var}',
                'content': fig_to_base64(fig)
            })
            plt.close(fig)
            
            # Add frequency statistics
            results['statistics'].append({
                'title': f'Frequency Statistics for {var}',
                'text': f"""
## Frequency Statistics for {var}

- **Number of unique values**: {df[var].nunique()}
- **Most common value**: {df[var].value_counts().index[0]} (count: {df[var].value_counts().iloc[0]})
- **Least common value**: {df[var].value_counts().index[-1]} (count: {df[var].value_counts().iloc[-1]})
- **Missing values**: {df[var].isna().sum()} ({df[var].isna().mean()*100:.2f}%)
"""
            })

# If we have at least two numeric variables, add correlation visualization
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
selected_numeric_cols = [var for var in variables if var in numeric_cols]

if len(selected_numeric_cols) >= 2:
    # Create correlation matrix
    corr_matrix = df[selected_numeric_cols].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(max(8, len(selected_numeric_cols) * 0.8), max(6, len(selected_numeric_cols) * 0.7)))
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    
    # Add variable names as labels
    ax.set_xticks(np.arange(len(selected_numeric_cols)))
    ax.set_yticks(np.arange(len(selected_numeric_cols)))
    ax.set_xticklabels(selected_numeric_cols)
    ax.set_yticklabels(selected_numeric_cols)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Correlation Coefficient')
    
    # Add correlation values to cells
    for i in range(len(selected_numeric_cols)):
        for j in range(len(selected_numeric_cols)):
            text_color = 'white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", 
                    ha='center', va='center', color=text_color)
    
    ax.set_title('Correlation Heatmap')
    plt.tight_layout()
    
    # Add to results
    results['visualizations'].append({
        'title': 'Correlation Heatmap',
        'description': 'Heatmap showing the correlation between numeric variables',
        'content': fig_to_base64(fig)
    })
    plt.close(fig)
    
    # Add correlation analysis
    results['statistics'].append({
        'title': 'Correlation Analysis',
        'text': f"""
## Correlation Analysis

The correlation heatmap shows the Pearson correlation coefficients between the numeric variables.

### Key observations:
{"  \\n".join([f"- **{col1} and {col2}**: Strong {'positive' if corr_matrix.loc[col1, col2] > 0 else 'negative'} correlation (r = {corr_matrix.loc[col1, col2]:.3f})" 
              for col1 in selected_numeric_cols for col2 in selected_numeric_cols 
              if col1 != col2 and abs(corr_matrix.loc[col1, col2]) > 0.7]) or "No strong correlations (r > 0.7) found between the variables."}
"""
    })
