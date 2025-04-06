import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  Paper, 
  TextField, 
  IconButton, 
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Divider,
  CircularProgress,
  Button,
  Tooltip
} from '@mui/material';
import { styled } from '@mui/material/styles';
import SendIcon from '@mui/icons-material/Send';
import AutoGraphIcon from '@mui/icons-material/AutoGraph';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import PersonIcon from '@mui/icons-material/Person';
import MicIcon from '@mui/icons-material/Mic';
import MicOffIcon from '@mui/icons-material/MicOff';
import HistoryIcon from '@mui/icons-material/History';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { runAnalysis } from '../services/statisticalAnalysisService';
import ReactMarkdown from 'react-markdown';

// Styled components
const ChatContainer = styled(Paper)(({ theme }) => ({
  height: '500px',
  display: 'flex',
  flexDirection: 'column',
  border: `1px solid ${theme.palette.divider}`,
  borderRadius: theme.shape.borderRadius,
  overflow: 'hidden',
}));

const MessagesList = styled(List)(({ theme }) => ({
  flex: 1,
  overflowY: 'auto',
  padding: theme.spacing(2),
  backgroundColor: theme.palette.background.default,
}));

const InputContainer = styled(Box)(({ theme }) => ({
  display: 'flex',
  padding: theme.spacing(1, 2),
  borderTop: `1px solid ${theme.palette.divider}`,
  backgroundColor: theme.palette.background.paper,
}));

const MessageItem = styled(ListItem)(({ theme, sender }) => ({
  marginBottom: theme.spacing(1),
  '& .MuiListItemText-root': {
    backgroundColor: sender === 'user' 
      ? theme.palette.primary.light 
      : theme.palette.secondary.light,
    borderRadius: sender === 'user'
      ? '20px 20px 0 20px'
      : '20px 20px 20px 0',
    padding: theme.spacing(1, 2),
    color: theme.palette.getContrastText(
      sender === 'user' 
        ? theme.palette.primary.light 
        : theme.palette.secondary.light
    ),
  },
  justifyContent: sender === 'user' ? 'flex-end' : 'flex-start',
  '& .MuiListItemAvatar-root': {
    order: sender === 'user' ? 1 : 0,
  }
}));

function ChatInterface({ processedData, onAnalysisResults, pyodideReady }) {
  const [message, setMessage] = useState('');
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hi! I'm your biostatistical AI assistant. How can I help you analyze your data?",
      sender: 'assistant',
      timestamp: new Date(),
    }
  ]);
  const [isListening, setIsListening] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const recognition = useRef(null);

  // Suggested queries
  const suggestedQueries = [
    "Summarize my data",
    "Run descriptive statistics",
    "Check for outliers and data issues",
    "Plot the distribution of variables",
    "Perform correlation analysis"
  ];

  // Setup speech recognition
  useEffect(() => {
    if ('SpeechRecognition' in window || 'webkitSpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      recognition.current = new SpeechRecognition();
      recognition.current.continuous = false;
      recognition.current.interimResults = false;
      
      recognition.current.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setMessage(transcript);
      };
      
      recognition.current.onend = () => {
        setIsListening(false);
      };
    }
    
    return () => {
      if (recognition.current) {
        recognition.current.stop();
      }
    };
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    
    if (!message.trim()) return;
    
    // Add user message
    const userMessage = {
      id: messages.length + 1,
      text: message,
      sender: 'user',
      timestamp: new Date(),
    };
    
    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setIsProcessing(true);
    
    try {
      // Process the message with Gemini and prepare analysis
      const analysisResponse = await runAnalysis(
        message, 
        processedData,
        (botMessage) => {
          setMessages(prev => [
            ...prev, 
            {
              id: prev.length + 1,
              text: botMessage,
              sender: 'assistant',
              timestamp: new Date(),
            }
          ]);
        }
      );
      
      // Send the analysis results to the parent component
      if (analysisResponse) {
        onAnalysisResults(analysisResponse);
      }
    } catch (error) {
      console.error('Error processing message:', error);
      
      // Add error message
      setMessages(prev => [
        ...prev, 
        {
          id: prev.length + 1,
          text: `Sorry, I encountered an error: ${error.message}. Please try rephrasing your question.`,
          sender: 'assistant',
          timestamp: new Date(),
        }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleToggleListen = () => {
    if (isListening) {
      recognition.current.stop();
      setIsListening(false);
    } else {
      recognition.current.start();
      setIsListening(true);
    }
  };

  const handleSuggestedQuery = (query) => {
    setMessage(query);
    // Focus the input
    inputRef.current.focus();
  };

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
        <Typography variant="h6">
          <AutoGraphIcon fontSize="small" sx={{ verticalAlign: 'middle', mr: 1 }} />
          Analysis Chat
        </Typography>
        
        <Tooltip title="You can ask questions about your data. Try asking for statistical analyses, visualizations, or insights.">
          <IconButton size="small">
            <InfoOutlinedIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      </Box>
      
      <ChatContainer elevation={2}>
        <MessagesList>
          {messages.map((msg) => (
            <MessageItem 
              key={msg.id} 
              sender={msg.sender}
              alignItems="flex-start"
              disableGutters
            >
              <ListItemAvatar>
                <Avatar>
                  {msg.sender === 'user' ? <PersonIcon /> : <SmartToyIcon />}
                </Avatar>
              </ListItemAvatar>
              <ListItemText 
                primary={
                  <Box>
                    <ReactMarkdown>
                      {msg.text}
                    </ReactMarkdown>
                  </Box>
                }
                secondary={formatTimestamp(msg.timestamp)}
                secondaryTypographyProps={{
                  variant: 'caption',
                  align: msg.sender === 'user' ? 'right' : 'left'
                }}
              />
            </MessageItem>
          ))}
          <div ref={messagesEndRef} />
          
          {isProcessing && (
            <Box display="flex" justifyContent="center" my={2}>
              <CircularProgress size={24} />
            </Box>
          )}
        </MessagesList>
        
        <Box px={2} py={1} bgcolor="background.paper">
          <Typography variant="caption" color="textSecondary">
            Suggested queries:
          </Typography>
          <Box display="flex" flexWrap="wrap" gap={1} my={1}>
            {suggestedQueries.map((query, index) => (
              <Button 
                key={index}
                size="small" 
                variant="outlined"
                onClick={() => handleSuggestedQuery(query)}
                disabled={isProcessing}
              >
                {query}
              </Button>
            ))}
          </Box>
        </Box>
        
        <Divider />
        
        <InputContainer component="form" onSubmit={handleSubmit}>
          <TextField
            fullWidth
            placeholder="Ask me about your data..."
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            variant="outlined"
            size="small"
            disabled={isProcessing || !pyodideReady}
            inputRef={inputRef}
            InputProps={{
              startAdornment: (
                <IconButton 
                  size="small" 
                  onClick={handleToggleListen}
                  disabled={isProcessing || !('SpeechRecognition' in window || 'webkitSpeechRecognition' in window)}
                  color={isListening ? 'primary' : 'default'}
                  sx={{ mr: 1 }}
                >
                  {isListening ? <MicIcon /> : <MicOffIcon />}
                </IconButton>
              ),
              endAdornment: (
                <Tooltip title="Run analysis">
                  <span>
                    <IconButton
                      type="submit"
                      disabled={!message.trim() || isProcessing || !pyodideReady}
                      color="primary"
                    >
                      {isProcessing ? <CircularProgress size={24} /> : <SendIcon />}
                    </IconButton>
                  </span>
                </Tooltip>
              )
            }}
          />
        </InputContainer>
      </ChatContainer>
    </Box>
  );
}

export default ChatInterface;
