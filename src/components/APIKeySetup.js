import React, { useState } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography,
  Link,
  Alert,
  IconButton,
  InputAdornment
} from '@mui/material';
import Visibility from '@mui/icons-material/Visibility';
import VisibilityOff from '@mui/icons-material/VisibilityOff';
import { testGeminiApiKey } from '../services/geminiService';

function APIKeySetup({ onApiKeySubmit }) {
  const [apiKey, setApiKey] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState(null);
  const [showApiKey, setShowApiKey] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!apiKey.trim()) {
      setError('Please enter a valid Google Gemini API key');
      return;
    }

    setIsValidating(true);
    setError(null);

    try {
      // Test if the API key is valid
      const isValid = await testGeminiApiKey(apiKey);
      
      if (isValid) {
        onApiKeySubmit(apiKey);
      } else {
        setError('Invalid API key. Please check and try again.');
      }
    } catch (err) {
      setError(`Error validating API key: ${err.message}`);
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <Box 
      display="flex" 
      flexDirection="column" 
      alignItems="center" 
      justifyContent="center" 
      minHeight="60vh"
    >
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          width: '100%', 
          maxWidth: 600,
          borderRadius: 2
        }}
      >
        <Typography variant="h5" component="h2" gutterBottom align="center">
          Welcome to AzzamAnalyst
        </Typography>
        
        <Typography variant="body1" paragraph>
          To get started, please enter your Google Gemini API key. This key will be stored locally in your browser and used to power the AI capabilities of AzzamAnalyst.
        </Typography>
        
        <Typography variant="body2" paragraph>
          Don't have a Google Gemini API key? You can get one from the{' '}
          <Link 
            href="https://ai.google.dev/" 
            target="_blank" 
            rel="noopener noreferrer"
          >
            Google AI Developer site
          </Link>.
        </Typography>
        
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}
        
        <form onSubmit={handleSubmit}>
          <TextField
            label="Google Gemini API Key"
            variant="outlined"
            fullWidth
            margin="normal"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            type={showApiKey ? 'text' : 'password'}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    aria-label="toggle api key visibility"
                    onClick={() => setShowApiKey(!showApiKey)}
                    edge="end"
                  >
                    {showApiKey ? <VisibilityOff /> : <Visibility />}
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          
          <Button 
            type="submit" 
            variant="contained" 
            color="primary" 
            fullWidth 
            size="large"
            disabled={isValidating}
            sx={{ mt: 2 }}
          >
            {isValidating ? 'Validating...' : 'Continue'}
          </Button>
          
          <Typography variant="body2" align="center" sx={{ mt: 2 }}>
            Your API key is stored only in your browser's local storage and is never sent to our servers.
          </Typography>
        </form>
      </Paper>
    </Box>
  );
}

export default APIKeySetup;
