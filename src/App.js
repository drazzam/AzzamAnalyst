import React, { useState, useEffect } from 'react';
import { Container, Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import APIKeySetup from './components/APIKeySetup';
import FileUpload from './components/FileUpload';
import DataProcessor from './components/DataProcessor';
import ChatInterface from './components/ChatInterface';
import ResultsDisplay from './components/ResultsDisplay';
import { initializePyodide } from './services/pyodideService';
import './styles/App.css';

// Create a theme instance
const theme = createTheme({
  palette: {
    primary: {
      main: '#2e7d32', // Green shade
    },
    secondary: {
      main: '#1565c0', // Blue shade
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
  },
});

function App() {
  const [apiKeySet, setApiKeySet] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [loadingMessage, setLoadingMessage] = useState('Initializing AzzamAnalyst...');
  const [currentStep, setCurrentStep] = useState('setup'); // setup, upload, process, analyze
  const [processedData, setProcessedData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  // Load Pyodide when the app starts
  useEffect(() => {
    const loadPyodide = async () => {
      try {
        setLoadingMessage('Loading Python environment...');
        await initializePyodide();
        setPyodideReady(true);
        setIsLoading(false);
      } catch (error) {
        console.error('Failed to load Pyodide:', error);
        setErrorMessage('Failed to initialize Python environment. Please try refreshing the page.');
        setIsLoading(false);
      }
    };

    loadPyodide();
    
    // Check if API key is already stored
    const storedApiKey = localStorage.getItem('geminiApiKey');
    if (storedApiKey) {
      setApiKeySet(true);
      if (currentStep === 'setup') {
        setCurrentStep('upload');
      }
    }
  }, [currentStep]);

  // Handle API key setup
  const handleApiKeySubmit = (apiKey) => {
    localStorage.setItem('geminiApiKey', apiKey);
    setApiKeySet(true);
    setCurrentStep('upload');
  };

  // Handle file upload and processing
  const handleFileProcessed = (data) => {
    setProcessedData(data);
    setCurrentStep('analyze');
  };

  // Handle analysis results
  const handleAnalysisResults = (results) => {
    setAnalysisResults(results);
  };

  // Render different components based on the current step
  const renderCurrentStep = () => {
    if (isLoading) {
      return (
        <Box 
          display="flex" 
          flexDirection="column" 
          alignItems="center" 
          justifyContent="center" 
          minHeight="80vh"
        >
          <div className="loading-spinner"></div>
          <p>{loadingMessage}</p>
        </Box>
      );
    }

    if (errorMessage) {
      return (
        <Box 
          display="flex" 
          flexDirection="column" 
          alignItems="center" 
          justifyContent="center" 
          minHeight="80vh"
        >
          <div className="error-message">{errorMessage}</div>
        </Box>
      );
    }

    switch (currentStep) {
      case 'setup':
        return <APIKeySetup onApiKeySubmit={handleApiKeySubmit} />;
      case 'upload':
        return <FileUpload onFileProcessed={handleFileProcessed} />;
      case 'analyze':
        return (
          <>
            <DataProcessor 
              data={processedData} 
              onDataProcessed={setProcessedData} 
            />
            <Box display="flex" flexDirection="column" gap={2} mt={2}>
              <ChatInterface 
                processedData={processedData} 
                onAnalysisResults={handleAnalysisResults}
                pyodideReady={pyodideReady} 
              />
              <ResultsDisplay results={analysisResults} />
            </Box>
          </>
        );
      default:
        return <div>Unknown step</div>;
    }
  };

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Container maxWidth="lg">
        <Box my={4}>
          <h1 className="app-title">AzzamAnalyst</h1>
          <h2 className="app-subtitle">Advanced Biostatistical AI Agent</h2>
          {renderCurrentStep()}
        </Box>
      </Container>
    </ThemeProvider>
  );
}

export default App;
