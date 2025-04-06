let pyodide = null;
let isPyodideInitialized = false;

/**
 * Initialize Pyodide environment
 * @returns {Promise<void>} - Promise that resolves when Pyodide is initialized
 */
export const initializePyodide = async () => {
  if (isPyodideInitialized) return;
  
  try {
    // Load Pyodide script dynamically
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js';
    document.head.appendChild(script);
    
    // Wait for script to load
    await new Promise((resolve, reject) => {
      script.onload = resolve;
      script.onerror = () => reject(new Error('Failed to load Pyodide script'));
    });
    
    console.log('Loading Pyodide...');
    pyodide = await window.loadPyodide();
    console.log('Pyodide loaded successfully');
    
    // Install key packages for biostatistical analysis
    console.log('Installing Python packages...');
    await pyodide.runPythonAsync(`
      import micropip
      await micropip.install([
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'statsmodels',
        'scikit-learn'
      ])
      
      # Basic setup for matplotlib
      import matplotlib.pyplot as plt
      plt.switch_backend('agg')
      
      # Helper functions
      import numpy as np
      import pandas as pd
      import io
      import base64
      import json
      from js import document
      
      # Function to convert matplotlib figure to base64 encoded image
      def fig_to_base64(fig):
          buf = io.BytesIO()
          fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
          buf.seek(0)
          img_str = base64.b64encode(buf.read()).decode('utf-8')
          return f'data:image/png;base64,{img_str}'
      
      print("Python environment ready")
    `);
    
    isPyodideInitialized = true;
    console.log('Python environment ready');
  } catch (error) {
    console.error('Error initializing Pyodide:', error);
    throw error;
  }
};

/**
 * Run Python code in the Pyodide environment
 * @param {string} code - Python code to execute
 * @returns {Promise<any>} - Result of Python execution
 */
export const runPython = async (code) => {
  if (!isPyodideInitialized) {
    throw new Error('Pyodide not initialized. Call initializePyodide() first.');
  }
  
  try {
    return await pyodide.runPythonAsync(code);
  } catch (error) {
    console.error('Error running Python code:', error);
    throw error;
  }
};

/**
 * Run a statistical analysis using Python
 * @param {string} analysisType - Type of analysis to run
 * @param {Object} data - Data to analyze
 * @param {Object} params - Parameters for the analysis
 * @returns {Promise<Object>} - Analysis results
 */
export const runStatisticalAnalysis = async (analysisType, data, params = {}) => {
  if (!isPyodideInitialized) {
    throw new Error('Pyodide not initialized. Call initializePyodide() first.');
  }
  
  try {
    // Convert data and params to Python-friendly format
    pyodide.globals.set('analysis_type', analysisType);
    pyodide.globals.set('data_json', JSON.stringify(data));
    pyodide.globals.set('params_json', JSON.stringify(params));
    
    // Run the analysis
    const result = await pyodide.runPythonAsync(`
      import json
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      from scipy import stats
      import io
      import base64
      
      # Parse inputs
      analysis_type = globals()['analysis_type']
      data = json.loads(globals()['data_json'])
      params = json.loads(globals()['params_json'])
      
      # Convert to DataFrame if needed
      if isinstance(data, list) and all(isinstance(item, dict) for item in data):
          df = pd.DataFrame(data)
      elif isinstance(data, dict) and all(isinstance(data[key], list) for key in data):
          df = pd.DataFrame(data)
      else:
          df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
      
      # Helper function for visualization
      def fig_to_base64(fig):
          buf = io.BytesIO()
          fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
          buf.seek(0)
          img_str = base64.b64encode(buf.read()).decode('utf-8')
          return f'<img src="data:image/png;base64,{img_str}" />'
      
      # Run the requested analysis
      result = {}
      
      if analysis_type == 'descriptive_stats':
          result['summary'] = df.describe().to_json()
          result['missing'] = df.isnull().sum().to_dict()
          
          # Create histograms for numeric columns
          numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
          if numeric_cols:
              fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(10, 4 * len(numeric_cols)))
              if len(numeric_cols) == 1:
                  axes = [axes]
              
              for i, col in enumerate(numeric_cols):
                  axes[i].hist(df[col].dropna(), bins=20, alpha=0.7)
                  axes[i].set_title(f'Distribution of {col}')
                  axes[i].set_xlabel(col)
                  axes[i].set_ylabel('Frequency')
              
              plt.tight_layout()
              result['histograms'] = fig_to_base64(fig)
              plt.close(fig)
      
      elif analysis_type == 'correlation':
          numeric_df = df.select_dtypes(include=[np.number])
          if not numeric_df.empty:
              # Calculate correlation matrix
              corr_matrix = numeric_df.corr()
              result['correlation_matrix'] = corr_matrix.to_json()
              
              # Create heatmap
              fig, ax = plt.subplots(figsize=(10, 8))
              im = ax.imshow(corr_matrix, cmap='coolwarm')
              
              # Add labels
              ax.set_xticks(np.arange(len(corr_matrix.columns)))
              ax.set_yticks(np.arange(len(corr_matrix.columns)))
              ax.set_xticklabels(corr_matrix.columns)
              ax.set_yticklabels(corr_matrix.columns)
              
              # Rotate x labels
              plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
              
              # Add colorbar
              cbar = ax.figure.colorbar(im, ax=ax)
              
              # Add correlation values to cells
              for i in range(len(corr_matrix.columns)):
                  for j in range(len(corr_matrix.columns)):
                      ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                              ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.7 else "white")
              
              ax.set_title("Correlation Matrix")
              plt.tight_layout()
              result['correlation_heatmap'] = fig_to_base64(fig)
              plt.close(fig)
          else:
              result['error'] = "No numeric columns found for correlation analysis"
      
      # Add more analysis types as needed
      
      # Convert result to JSON
      json.dumps(result)
    `);
    
    return JSON.parse(result);
  } catch (error) {
    console.error('Error running statistical analysis:', error);
    throw error;
  }
};

/**
 * Generate visualization using Python
 * @param {string} vizType - Type of visualization to generate
 * @param {Object} data - Data to visualize
 * @param {Object} params - Parameters for the visualization
 * @returns {Promise<string>} - HTML representation of the visualization
 */
export const generateVisualization = async (vizType, data, params = {}) => {
  if (!isPyodideInitialized) {
    throw new Error('Pyodide not initialized. Call initializePyodide() first.');
  }
  
  try {
    // Convert data and params to Python-friendly format
    pyodide.globals.set('viz_type', vizType);
    pyodide.globals.set('data_json', JSON.stringify(data));
    pyodide.globals.set('params_json', JSON.stringify(params));
    
    // Generate the visualization
    const result = await pyodide.runPythonAsync(`
      import json
      import pandas as pd
      import numpy as np
      import matplotlib.pyplot as plt
      import io
      import base64
      
      # Parse inputs
      viz_type = globals()['viz_type']
      data = json.loads(globals()['data_json'])
      params = json.loads(globals()['params_json'])
      
      # Convert to DataFrame if needed
      if isinstance(data, list) and all(isinstance(item, dict) for item in data):
          df = pd.DataFrame(data)
      elif isinstance(data, dict) and all(isinstance(data[key], list) for key in data):
          df = pd.DataFrame(data)
      else:
          df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
      
      # Helper function for visualization
      def fig_to_base64(fig):
          buf = io.BytesIO()
          fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
          buf.seek(0)
          img_str = base64.b64encode(buf.read()).decode('utf-8')
          return f'<img src="data:image/png;base64,{img_str}" style="max-width:100%;height:auto;" />'
      
      # Generate the requested visualization
      html_output = ""
      
      if viz_type == 'histogram':
          column = params.get('column')
          if column and column in df.columns:
              fig, ax = plt.subplots(figsize=(10, 6))
              ax.hist(df[column].dropna(), bins=params.get('bins', 20), alpha=0.7)
              ax.set_title(f'Distribution of {column}')
              ax.set_xlabel(column)
              ax.set_ylabel('Frequency')
              plt.grid(alpha=0.3)
              plt.tight_layout()
              html_output = fig_to_base64(fig)
              plt.close(fig)
          else:
              html_output = "<p>Error: Column not found in data</p>"
      
      elif viz_type == 'scatter':
          x_col = params.get('x')
          y_col = params.get('y')
          if x_col in df.columns and y_col in df.columns:
              fig, ax = plt.subplots(figsize=(10, 6))
              ax.scatter(df[x_col], df[y_col], alpha=0.7)
              
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
              
              ax.set_title(f'Scatter Plot of {y_col} vs {x_col}')
              ax.set_xlabel(x_col)
              ax.set_ylabel(y_col)
              plt.grid(alpha=0.3)
              plt.tight_layout()
              html_output = fig_to_base64(fig)
              plt.close(fig)
          else:
              html_output = "<p>Error: One or both columns not found in data</p>"
      
      # Add more visualization types as needed
      
      html_output
    `);
    
    return result;
  } catch (error) {
    console.error('Error generating visualization:', error);
    throw error;
  }
};
