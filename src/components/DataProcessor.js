import React, { useState, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Stepper, 
  Step, 
  StepLabel, 
  Button,
  CircularProgress,
  Alert,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import { runPython } from '../services/pyodideService';
import { cleanData } from '../services/dataCleaningService';

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`data-tabpanel-${index}`}
      aria-labelledby={`data-tab-${index}`}
      {...other}
      style={{ height: '100%' }}
    >
      {value === index && (
        <Box sx={{ p: 2, height: '100%', overflow: 'auto' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function DataProcessor({ data, onDataProcessed }) {
  const [activeStep, setActiveStep] = useState(0);
  const [cleaningLog, setCleaningLog] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [processedData, setProcessedData] = useState(null);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState(0);
  const [datasetTab, setDatasetTab] = useState(0);
  
  const steps = ['Data Inspection', 'Data Cleaning', 'Data Preparation'];
  
  useEffect(() => {
    // Initialize with the provided data
    if (data && !processedData) {
      setProcessedData(data);
    }
  }, [data, processedData]);
  
  const handleNext = () => {
    if (activeStep === steps.length - 1) {
      // Last step - finish processing
      onDataProcessed(processedData);
    } else {
      setActiveStep((prevStep) => prevStep + 1);
      
      // If moving to cleaning step, start the cleaning process
      if (activeStep === 0) {
        startDataCleaning();
      }
    }
  };
  
  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };
  
  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };
  
  const handleDatasetTabChange = (event, newValue) => {
    setDatasetTab(newValue);
  };
  
  const startDataCleaning = async () => {
    setIsProcessing(true);
    setProcessingProgress(0);
    setError(null);
    
    try {
      // Mock progress updates
      const progressInterval = setInterval(() => {
        setProcessingProgress(prev => {
          const newProgress = prev + 5;
          return newProgress > 90 ? 90 : newProgress;
        });
      }, 300);
      
      // Perform data cleaning
      const cleaningResult = await cleanData(processedData, (progress) => {
        setProcessingProgress(Math.min(90 + (progress * 10), 99));
      });
      
      clearInterval(progressInterval);
      setProcessingProgress(100);
      
      // Update the processed data with cleaned data
      setProcessedData(cleaningResult.data);
      setCleaningLog(cleaningResult.log);
      
      // Move to next step after a short delay
      setTimeout(() => {
        setIsProcessing(false);
      }, 500);
    } catch (err) {
      console.error('Error cleaning data:', err);
      setError(`Error cleaning data: ${err.message}`);
      setIsProcessing(false);
    }
  };
  
  const renderDatasetTable = (dataset) => {
    if (!dataset || !dataset.data || dataset.data.length === 0) {
      return (
        <Typography color="text.secondary" align="center">
          No data available.
        </Typography>
      );
    }
    
    // Get columns
    const columns = dataset.headers || Object.keys(dataset.data[0]);
    
    // Limit rows for display
    const rowsToShow = dataset.data.slice(0, 100);
    
    return (
      <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
        <Table stickyHeader size="small">
          <TableHead>
            <TableRow>
              <TableCell>#</TableCell>
              {columns.map((column, index) => (
                <TableCell key={index}>
                  {column}
                  {dataset.quality && dataset.quality.column_data_types && dataset.quality.column_data_types[column] && (
                    <Chip 
                      label={dataset.quality.column_data_types[column]} 
                      size="small" 
                      sx={{ ml: 1, fontSize: '0.6rem' }} 
                    />
                  )}
                </TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {rowsToShow.map((row, rowIndex) => (
              <TableRow key={rowIndex}>
                <TableCell>{rowIndex + 1}</TableCell>
                {columns.map((column, colIndex) => (
                  <TableCell key={colIndex}>
                    {row[column] !== null && row[column] !== undefined ? String(row[column]) : ''}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };
  
  const renderDatasetSummary = (dataset) => {
    if (!dataset) return null;
    
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Dataset Summary
        </Typography>
        <Typography variant="body2" paragraph>
          This dataset contains {dataset.rowCount} rows and {dataset.headers ? dataset.headers.length : 0} columns.
        </Typography>
        
        {dataset.quality && (
          <Box mt={2}>
            <Typography variant="subtitle1" gutterBottom>
              Data Quality Metrics
            </Typography>
            <TableContainer component={Paper} variant="outlined">
              <Table size="small">
                <TableBody>
                  <TableRow>
                    <TableCell component="th" scope="row">Missing Values</TableCell>
                    <TableCell>{dataset.quality.missing_values_percentage?.toFixed(2)}%</TableCell>
                  </TableRow>
                  <TableRow>
                    <TableCell component="th" scope="row">Duplicate Rows Removed</TableCell>
                    <TableCell>{dataset.quality.duplicate_rows_removed || 0}</TableCell>
                  </TableRow>
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}
      </Box>
    );
  };
  
  const renderTextContent = (textContent) => {
    if (!textContent || textContent.length === 0) {
      return (
        <Typography color="text.secondary" align="center">
          No text content available.
        </Typography>
      );
    }
    
    return textContent.map((content, index) => (
      <Paper key={index} variant="outlined" sx={{ p: 2, mb: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          {content.name} ({content.type})
        </Typography>
        <Box 
          sx={{ 
            maxHeight: 300, 
            overflow: 'auto',
            whiteSpace: 'pre-wrap',
            fontFamily: 'monospace',
            fontSize: '0.875rem',
            bgcolor: '#f5f5f5',
            p: 2,
            borderRadius: 1
          }}
        >
          {content.content}
        </Box>
      </Paper>
    ));
  };
  
  const renderCleaningLog = () => {
    if (!cleaningLog || cleaningLog.length === 0) {
      return (
        <Typography color="text.secondary" align="center">
          No cleaning log available yet.
        </Typography>
      );
    }
    
    return (
      <Box>
        <Typography variant="h6" gutterBottom>
          Data Cleaning Steps
        </Typography>
        <TableContainer component={Paper} variant="outlined">
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Step</TableCell>
                <TableCell>Description</TableCell>
                <TableCell>Details</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {cleaningLog.map((step, index) => (
                <TableRow key={index}>
                  <TableCell>{step.step}</TableCell>
                  <TableCell>{step.description}</TableCell>
                  <TableCell>
                    {step.rows_before && step.rows_after && (
                      <Typography variant="body2">
                        Rows: {step.rows_before} → {step.rows_after} ({step.rows_removed} removed)
                      </Typography>
                    )}
                    {step.cols_before && step.cols_after && (
                      <Typography variant="body2">
                        Columns: {step.cols_before} → {step.cols_after} ({step.cols_removed} removed)
                      </Typography>
                    )}
                    {step.columns_processed && (
                      <Typography variant="body2">
                        Processed {step.columns_processed.length} columns
                      </Typography>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    );
  };
  
  // Render appropriate content for each step
  const renderStepContent = () => {
    if (!processedData) {
      return (
        <Typography color="text.secondary" align="center">
          No data available.
        </Typography>
      );
    }
    
    switch (activeStep) {
      case 0: // Data Inspection
        return (
          <Box>
            <Tabs value={activeTab} onChange={handleTabChange} aria-label="data tabs">
              <Tab label="Datasets" />
              <Tab label="Text Content" />
              <Tab label="Metadata" />
            </Tabs>
            
            <TabPanel value={activeTab} index={0}>
              {processedData.datasets && processedData.datasets.length > 0 ? (
                <Box>
                  <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
                    <Tabs 
                      value={datasetTab} 
                      onChange={handleDatasetTabChange} 
                      aria-label="dataset tabs"
                      variant="scrollable"
                      scrollButtons="auto"
                    >
                      {processedData.datasets.map((dataset, index) => (
                        <Tab key={index} label={dataset.name} />
                      ))}
                    </Tabs>
                  </Box>
                  
                  {processedData.datasets.map((dataset, index) => (
                    <TabPanel key={index} value={datasetTab} index={index}>
                      {renderDatasetSummary(dataset)}
                      <Box mt={3}>
                        <Typography variant="h6" gutterBottom>
                          Data Preview
                        </Typography>
                        {renderDatasetTable(dataset)}
                      </Box>
                    </TabPanel>
                  ))}
                </Box>
              ) : (
                <Typography color="text.secondary" align="center">
                  No datasets available.
                </Typography>
              )}
            </TabPanel>
            
            <TabPanel value={activeTab} index={1}>
              {renderTextContent(processedData.textContent)}
            </TabPanel>
            
            <TabPanel value={activeTab} index={2}>
              <Paper variant="outlined" sx={{ p: 2 }}>
                <Typography variant="h6" gutterBottom>
                  Metadata
                </Typography>
                <TableContainer>
                  <Table size="small">
                    <TableBody>
                      <TableRow>
                        <TableCell component="th" scope="row">Total Files</TableCell>
                        <TableCell>{processedData.metadata?.fileCount}</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell component="th" scope="row">Total Size</TableCell>
                        <TableCell>{(processedData.metadata?.totalBytes / 1024).toFixed(2)} KB</TableCell>
                      </TableRow>
                      <TableRow>
                        <TableCell component="th" scope="row">File Types</TableCell>
                        <TableCell>
                          {processedData.metadata?.fileTypes && Object.entries(processedData.metadata.fileTypes).map(([type, count]) => (
                            <Chip 
                              key={type} 
                              label={`${type}: ${count}`} 
                              size="small" 
                              sx={{ mr: 1, mb: 1 }} 
                            />
                          ))}
                        </TableCell>
                      </TableRow>
                    </TableBody>
                  </Table>
                </TableContainer>
              </Paper>
            </TabPanel>
          </Box>
        );
        
      case 1: // Data Cleaning
        return (
          <Box>
            {isProcessing ? (
              <Box display="flex" flexDirection="column" alignItems="center" my={4}>
                <CircularProgress size={40} />
                <Typography variant="body1" mt={2}>
                  Cleaning data... {processingProgress}%
                </Typography>
              </Box>
            ) : (
              <Box>
                {error ? (
                  <Alert severity="error" sx={{ mb: 2 }}>
                    {error}
                  </Alert>
                ) : (
                  <Alert severity="success" sx={{ mb: 2 }}>
                    Data cleaning completed successfully!
                  </Alert>
                )}
                
                {renderCleaningLog()}
              </Box>
            )}
          </Box>
        );
        
      case 2: // Data Preparation
        return (
          <Box>
            <Alert severity="info" sx={{ mb: 2 }}>
              Your data is now ready for analysis. Continue to start interacting with the AI.
            </Alert>
            
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography variant="h6" gutterBottom>
                Data Statistics
              </Typography>
              <TableContainer>
                <Table size="small">
                  <TableBody>
                    <TableRow>
                      <TableCell component="th" scope="row">Datasets</TableCell>
                      <TableCell>{processedData.datasets?.length || 0}</TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row">Total Variables</TableCell>
                      <TableCell>
                        {processedData.datasets?.reduce((sum, dataset) => sum + (dataset.headers?.length || 0), 0) || 0}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row">Total Observations</TableCell>
                      <TableCell>
                        {processedData.datasets?.reduce((sum, dataset) => sum + (dataset.rowCount || 0), 0) || 0}
                      </TableCell>
                    </TableRow>
                    <TableRow>
                      <TableCell component="th" scope="row">Text Content Items</TableCell>
                      <TableCell>{processedData.textContent?.length || 0}</TableCell>
                    </TableRow>
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
            
            <Typography variant="body1" mt={3}>
              Now you can interact with the AI to analyze your data. You can ask questions about your data, request statistical analyses, or create visualizations.
            </Typography>
          </Box>
        );
        
      default:
        return null;
    }
  };
  
  return (
    <Box mb={4}>
      <Paper elevation={0} sx={{ p: 3, border: '1px solid', borderColor: 'divider' }}>
        <Stepper activeStep={activeStep} alternativeLabel>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>
        
        <Box mt={4}>
          {renderStepContent()}
        </Box>
        
        <Box mt={3} display="flex" justifyContent="space-between">
          <Button
            disabled={activeStep === 0 || isProcessing}
            onClick={handleBack}
          >
            Back
          </Button>
          <Button
            variant="contained"
            color="primary"
            onClick={handleNext}
            disabled={isProcessing}
          >
            {activeStep === steps.length - 1 ? 'Finish' : 'Next'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
}

export default DataProcessor;
