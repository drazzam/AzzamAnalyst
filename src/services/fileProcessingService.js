import Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { runPython } from './pyodideService';

/**
 * Process different types of files and convert them to appropriate data structures
 * @param {Array} files - Array of File objects
 * @param {Function} progressCallback - Callback function to report progress
 * @returns {Object} - Processed data organized by file and type
 */
export const processFiles = async (files, progressCallback) => {
  const result = {
    datasets: [],
    textContent: [],
    metadata: {
      totalBytes: files.reduce((sum, file) => sum + file.size, 0),
      fileCount: files.length,
      fileTypes: {}
    }
  };

  let processedCount = 0;

  for (const file of files) {
    try {
      // Get file extension
      const extension = file.name.split('.').pop().toLowerCase();
      
      // Update metadata
      if (!result.metadata.fileTypes[extension]) {
        result.metadata.fileTypes[extension] = 0;
      }
      result.metadata.fileTypes[extension]++;
      
      // Process file based on type
      if (extension === 'csv') {
        const dataset = await processCSV(file);
        result.datasets.push({
          name: file.name,
          type: 'csv',
          data: dataset,
          headers: dataset.length > 0 ? Object.keys(dataset[0]) : [],
          rowCount: dataset.length
        });
      } else if (['xlsx', 'xls'].includes(extension)) {
        const excelData = await processExcel(file);
        result.datasets.push({
          name: file.name,
          type: 'excel',
          data: excelData.data,
          sheets: excelData.sheets,
          activeSheet: excelData.activeSheet,
          headers: excelData.headers,
          rowCount: excelData.data.length
        });
      } else if (['docx', 'doc'].includes(extension)) {
        const textContent = await processDocx(file);
        result.textContent.push({
          name: file.name,
          type: 'document',
          content: textContent
        });
      } else if (extension === 'pdf') {
        const pdfContent = await processPDF(file);
        result.textContent.push({
          name: file.name,
          type: 'pdf',
          content: pdfContent
        });
      } else if (extension === 'txt') {
        const textContent = await processTextFile(file);
        result.textContent.push({
          name: file.name,
          type: 'text',
          content: textContent
        });
      }
      
      // Update progress
      processedCount++;
      if (progressCallback) {
        progressCallback(processedCount / files.length);
      }
    } catch (error) {
      console.error(`Error processing file ${file.name}:`, error);
      throw new Error(`Failed to process ${file.name}: ${error.message}`);
    }
  }

  // Run Python data cleaning and validation for datasets
  if (result.datasets.length > 0) {
    try {
      result.datasets = await cleanDatasets(result.datasets);
    } catch (error) {
      console.error('Error in Python data cleaning:', error);
      // Continue with the original datasets if cleaning fails
    }
  }

  return result;
};

/**
 * Process CSV file
 * @param {File} file - CSV file
 * @returns {Array} - Array of objects where each object is a row
 */
const processCSV = (file) => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => {
        if (results.errors.length > 0) {
          reject(new Error(`CSV parsing error: ${results.errors[0].message}`));
        } else {
          resolve(results.data);
        }
      },
      error: (error) => {
        reject(error);
      }
    });
  });
};

/**
 * Process Excel file
 * @param {File} file - Excel file
 * @returns {Object} - Object containing data and sheet information
 */
const processExcel = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target.result);
        const workbook = XLSX.read(data, { 
          type: 'array',
          cellDates: true,
          cellNF: true
        });
        
        // Get the first sheet as default
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        
        // Convert sheet to JSON
        const jsonData = XLSX.utils.sheet_to_json(worksheet, { 
          header: 'A',
          defval: null,
        });
        
        // Get headers (first row)
        const headers = jsonData.length > 0 ? Object.keys(jsonData[0]).filter(key => key !== '__rowNum__') : [];
        
        // Get all sheet names
        const sheets = workbook.SheetNames;
        
        resolve({
          data: jsonData,
          sheets: sheets,
          activeSheet: firstSheetName,
          headers: headers
        });
      } catch (error) {
        reject(new Error(`Excel parsing error: ${error.message}`));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Error reading Excel file'));
    };
    
    reader.readAsArrayBuffer(file);
  });
};

/**
 * Process DOCX file
 * @param {File} file - DOCX file
 * @returns {String} - Extracted text content
 */
const processDocx = async (file) => {
  // Using mammoth.js through Pyodide to extract text
  const reader = new FileReader();
  
  return new Promise((resolve, reject) => {
    reader.onload = async (e) => {
      try {
        // For the demo, we'll just read the file as text
        // In a real implementation, use mammoth.js through Pyodide
        resolve("Text content extracted from DOCX file: " + file.name);
      } catch (error) {
        reject(new Error(`DOCX parsing error: ${error.message}`));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Error reading DOCX file'));
    };
    
    reader.readAsText(file);
  });
};

/**
 * Process PDF file
 * @param {File} file - PDF file
 * @returns {String} - Extracted text content
 */
const processPDF = async (file) => {
  // Using pdf.js through Pyodide to extract text
  const reader = new FileReader();
  
  return new Promise((resolve, reject) => {
    reader.onload = async (e) => {
      try {
        // For the demo, we'll just read the file as text
        // In a real implementation, use pdf.js through Pyodide
        resolve("Text content extracted from PDF file: " + file.name);
      } catch (error) {
        reject(new Error(`PDF parsing error: ${error.message}`));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Error reading PDF file'));
    };
    
    reader.readAsText(file);
  });
};

/**
 * Process text file
 * @param {File} file - Text file
 * @returns {String} - Text content
 */
const processTextFile = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    
    reader.onload = (e) => {
      try {
        resolve(e.target.result);
      } catch (error) {
        reject(new Error(`Text file parsing error: ${error.message}`));
      }
    };
    
    reader.onerror = () => {
      reject(new Error('Error reading text file'));
    };
    
    reader.readAsText(file);
  });
};

/**
 * Clean datasets using Python
 * @param {Array} datasets - Array of datasets
 * @returns {Array} - Cleaned datasets
 */
const cleanDatasets = async (datasets) => {
  // Convert datasets to a format that can be passed to Python
  const datasetsJSON = JSON.stringify(datasets);
  
  // Run Python data cleaning script
  const cleanedDatasetsJSON = await runPython(`
import json
import pandas as pd
import numpy as np
from io import StringIO

# Parse the input datasets
datasets = json.loads('''${datasetsJSON}''')
cleaned_datasets = []

# Process each dataset
for dataset in datasets:
    # For CSV and Excel datasets
    if dataset["type"] in ["csv", "excel"]:
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset["data"])
        
        # Basic cleaning operations
        # 1. Drop completely empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # 2. Handle missing values (replace with NaN for proper identification)
        df = df.replace(['', ' ', 'NULL', 'NA', 'N/A', '#NA', '#N/A'], np.nan)
        
        # 3. Convert data types (automatic detection)
        for col in df.columns:
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                # If numeric conversion fails, leave as is
                pass
                
            # Try to convert to datetime if possible (for string columns only)
            if df[col].dtype == 'object':
                try:
                    temp_series = pd.to_datetime(df[col], errors='coerce')
                    # If more than 80% of the values are valid dates, convert the column
                    if temp_series.notnull().mean() > 0.8:
                        df[col] = temp_series
                except:
                    pass
        
        # 4. Remove duplicate rows
        df = df.drop_duplicates()
        
        # 5. Trim whitespace from string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
        
        # Update the dataset with cleaned data
        dataset["data"] = df.to_dict('records')
        dataset["rowCount"] = len(df)
        dataset["headers"] = list(df.columns)
        
        # Add data quality metrics
        dataset["quality"] = {
            "missing_values_percentage": (df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2),
            "duplicate_rows_removed": len(dataset["data"]) - len(df),
            "column_data_types": {col: str(df[col].dtype) for col in df.columns}
        }
    
    cleaned_datasets.append(dataset)

# Return the cleaned datasets
json.dumps(cleaned_datasets)
  `);
  
  return JSON.parse(cleanedDatasetsJSON);
};
