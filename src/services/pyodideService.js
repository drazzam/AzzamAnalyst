/**
 * Service for loading and interacting with Pyodide (Python in WebAssembly)
 */

let pyodide = null;
let isPyodideInitialized = false;
let initializationPromise = null;

/**
 * Load Pyodide and initialize the environment
 * @returns {Promise<void>} - Promise that resolves when Pyodide is ready
 */
export const initializePyodide = async () => {
  // Return existing promise if already initializing
  if (initializationPromise) {
    return initializationPromise;
  }
  
  // Return immediately if already initialized
  if (isPyodideInitialized && pyodide) {
    return Promise.resolve();
  }
  
  // Start initialization
  initializationPromise = (async () => {
    try {
      console.log('Loading Pyodide...');
      
      // Check if Pyodide is already loaded
      if (!window.loadPyodide) {
        console.log('Pyodide loader not found, loading script...');
        await loadPyodideScript();
      }
      
      // Load Pyodide
      pyodide = await window.loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.23.4/full/"
      });
      
      console.log('Pyodide loaded successfully');
      
      // Install required packages
      console.log('Installing Python packages...');
      await installPythonPackages();
      
      // Set up libraries and helper functions
      await setupPythonEnvironment();
      
      // Mark as initialized
      isPyodideInitialized = true;
      console.log('Pyodide environment ready');
      
      return pyodide;
    } catch (error) {
      console.error('Failed to initialize Pyodide:', error);
      initializationPromise = null;
      throw error;
    }
  })();
  
  return initializationPromise;
};

/**
 * Load the Pyodide script
 * @returns {Promise<void>} - Promise that resolves when the script is loaded
 */
const loadPyodideScript = () => {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.23.4/full/pyodide.js';
    script.onload = resolve;
    script.onerror = () => reject(new Error('Failed to load Pyodide script'));
    document.head.appendChild(script);
  });
};

/**
 * Install required Python packages using micropip
 * @returns {Promise<void>} - Promise that resolves when packages are installed
 */
const installPythonPackages = async () => {
  try {
    // Load micropip
    await pyodide.loadPackage('micropip');
    
    // Install required packages
    await pyodide.runPythonAsync(`
      import micropip
      await micropip.install(['numpy', 'pandas', 'matplotlib', 'scipy', 'seaborn', 'scikit-learn', 'statsmodels'])
    `);
    
    console.log('Python packages installed successfully');
  } catch (error) {
    console.error('Error installing Python packages:', error);
    throw error;
  }
};

/**
 * Set up the Python environment with helper functions and configurations
 * @returns {Promise<void>} - Promise that resolves when environment is set up
 */
const setupPythonEnvironment = async () => {
  try {
    // Set up matplotlib for non-interactive use
    await pyodide.runPythonAsync(`
      import matplotlib.pyplot as plt
      plt.switch_backend('agg')
      
      # Basic setup for common libraries
      import numpy as np
      import pandas as pd
      import io
      import base64
      import json
      
      # Helper function to convert matplotlib figure to base64 encoded image
      def fig_to_base64(fig):
          buf = io.BytesIO()
          fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
          buf.seek(0)
          img_str = base64.b64encode(buf.read()).decode('utf-8')
          return f'data:image/png;base64,{img_str}'
      
      print("Python environment setup complete")
    `);
  } catch (error) {
    console.error('Error setting up Python environment:', error);
    throw error;
  }
};

/**
 * Run Python code and return the result
 * @param {string} code - Python code to execute
 * @returns {Promise<any>} - Result from Python execution
 */
export const runPython = async (code) => {
  // Initialize Pyodide if not already done
  if (!isPyodideInitialized) {
    try {
      await initializePyodide();
    } catch (error) {
      throw new Error(`Failed to initialize Pyodide: ${error.message}`);
    }
  }
  
  try {
    // Execute the code
    return await pyodide.runPythonAsync(code);
  } catch (error) {
    console.error('Error executing Python code:', error);
    
    // Enhance error message with Python traceback if available
    let errorMessage = error.message;
    if (error.message.includes('PythonError')) {
      try {
        const traceback = await pyodide.runPythonAsync(`
          import traceback
          import sys
          traceback.format_exc()
        `);
        errorMessage = `Python Error: ${traceback}`;
      } catch (e) {
        // If we can't get the traceback, just use the original error
      }
    }
    
    throw new Error(errorMessage);
  }
};

/**
 * Check if Pyodide is initialized
 * @returns {boolean} - Whether Pyodide is initialized
 */
export const isPyodideReady = () => {
  return isPyodideInitialized;
};

/**
 * Get the Pyodide instance
 * @returns {object|null} - Pyodide instance or null if not initialized
 */
export const getPyodide = () => {
  return pyodide;
};

/**
 * Create a Python proxy for a JavaScript object
 * @param {object} obj - JavaScript object to proxy
 * @returns {object} - Python proxy object
 */
export const createPythonProxy = (obj) => {
  if (!isPyodideInitialized) {
    throw new Error('Pyodide not initialized');
  }
  
  return pyodide.toPy(obj);
};

/**
 * Convert a Python object to a JavaScript object
 * @param {object} obj - Python object to convert
 * @returns {object} - JavaScript object
 */
export const pythonToJs = (obj) => {
  if (!isPyodideInitialized) {
    throw new Error('Pyodide not initialized');
  }
  
  return pyodide.toJs(obj);
};

/**
 * Restart the Python environment (clear all variables and state)
 * @returns {Promise<void>} - Promise that resolves when environment is restarted
 */
export const restartPythonEnvironment = async () => {
  if (!isPyodideInitialized) {
    return;
  }
  
  try {
    // Reset the Python globals
    await pyodide.runPythonAsync(`
      import sys
      import builtins
      
      # Clear all user-defined globals
      user_globals = dict(globals())
      for key in list(user_globals.keys()):
          if key not in dir(builtins) and not key.startswith('__'):
              del globals()[key]
      
      # Re-import commonly used libraries
      import numpy as np
      import pandas as pd
      import matplotlib.pyplot as plt
      import io
      import base64
      import json
      
      # Reset matplotlib
      plt.close('all')
      
      print("Python environment restarted")
    `);
  } catch (error) {
    console.error('Error restarting Python environment:', error);
    throw error;
  }
};
