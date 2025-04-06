import React, { useState, useRef } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Button, 
  LinearProgress, 
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Alert,
  Grid
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import InsertDriveFileIcon from '@mui/icons-material/InsertDriveFile';
import DeleteIcon from '@mui/icons-material/Delete';
import DescriptionIcon from '@mui/icons-material/Description';
import TableChartIcon from '@mui/icons-material/TableChart';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import { processFiles } from '../services/fileProcessingService';

// Supported file types
const ACCEPTED_FILE_TYPES = [
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', // .xlsx
  'application/vnd.ms-excel', // .xls
  'text/csv', // .csv
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document', // .docx
  'application/msword', // .doc
  'application/pdf', // .pdf
  'text/plain', // .txt
];

// Helper function to check if a file type is supported
const isFileTypeSupported = (file) => {
  return ACCEPTED_FILE_TYPES.includes(file.type);
};

// Helper function to get the appropriate icon for a file
const getFileIcon = (file) => {
  if (file.type.includes('spreadsheet') || file.type === 'text/csv' || file.type.includes('excel')) {
    return <TableChartIcon />;
  } else if (file.type.includes('word') || file.type.includes('document')) {
    return <DescriptionIcon />;
  } else if (file.type === 'application/pdf') {
    return <PictureAsPdfIcon />;
  } else {
    return <InsertDriveFileIcon />;
  }
};

// Styled components
const UploadBox = styled(Paper)(({ theme }) => ({
  border: '2px dashed',
  borderColor: theme.palette.primary.main,
  backgroundColor: theme.palette.background.default,
  padding: theme.spacing(5),
  textAlign: 'center',
  cursor: 'pointer',
  transition: 'border-color 0.3s ease-in-out, background-color 0.3s ease-in-out',
  '&:hover': {
    borderColor: theme.palette.primary.dark,
    backgroundColor: theme.palette.action.hover,
  },
}));

const HiddenInput = styled('input')({
  display: 'none',
});

function FileUpload({ onFileProcessed }) {
  const [files, setFiles] = useState([]);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processingProgress, setProcessingProgress] = useState(0);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    addFilesToList(selectedFiles);
  };

  const addFilesToList = (selectedFiles) => {
    const newValidFiles = selectedFiles.filter(file => {
      if (!isFileTypeSupported(file)) {
        setError(`File type not supported: ${file.name}`);
        return false;
      }
      return true;
    });

    setFiles(prev => [...prev, ...newValidFiles]);
    setError(null);
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      addFilesToList(Array.from(e.dataTransfer.files));
    }
  };

  const handleRemoveFile = (indexToRemove) => {
    setFiles(files.filter((_, index) => index !== indexToRemove));
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const handleProcessFiles = async () => {
    if (files.length === 0) {
      setError('Please upload at least one file.');
      return;
    }

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

      // Process the files
      const processingResult = await processFiles(files, (progress) => {
        setProcessingProgress(Math.min(90 + (progress * 10), 99));
      });

      clearInterval(progressInterval);
      setProcessingProgress(100);

      // Delay a bit to show 100% progress
      setTimeout(() => {
        onFileProcessed(processingResult);
      }, 500);
    } catch (err) {
      console.error('Error processing files:', err);
      setError(`Error processing files: ${err.message}`);
      setIsProcessing(false);
    }
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom align="center">
        Upload Your Data Files
      </Typography>
      
      <Typography variant="body1" paragraph align="center">
        Upload your data files for analysis. Supported formats: Excel (.xlsx, .xls), CSV, Word (.docx, .doc), PDF, and text files.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}
      
      <Box 
        onDragEnter={handleDragEnter}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
        onDrop={handleDrop}
      >
        <UploadBox 
          elevation={dragActive ? 3 : 1}
          onClick={handleUploadClick}
          sx={{
            borderColor: dragActive ? 'primary.dark' : 'primary.main',
            backgroundColor: dragActive ? 'action.hover' : 'background.default',
          }}
        >
          <HiddenInput
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileChange}
            accept=".xlsx,.xls,.csv,.docx,.doc,.pdf,.txt"
          />
          <CloudUploadIcon color="primary" sx={{ fontSize: 60, mb: 2 }} />
          <Typography variant="h6" gutterBottom>
            Drag and drop files here
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Or click to browse your files
          </Typography>
          <Box mt={2}>
            <Chip 
              label="Excel (.xlsx, .xls)" 
              size="small" 
              sx={{ m: 0.5 }} 
            />
            <Chip 
              label="CSV" 
              size="small" 
              sx={{ m: 0.5 }} 
            />
            <Chip 
              label="Word (.docx, .doc)" 
              size="small" 
              sx={{ m: 0.5 }} 
            />
            <Chip 
              label="PDF" 
              size="small" 
              sx={{ m: 0.5 }} 
            />
            <Chip 
              label="Text (.txt)" 
              size="small" 
              sx={{ m: 0.5 }} 
            />
          </Box>
        </UploadBox>
      </Box>
      
      {files.length > 0 && (
        <Box mt={3}>
          <Typography variant="h6" gutterBottom>
            Selected Files ({files.length})
          </Typography>
          <Paper variant="outlined" sx={{ maxHeight: 300, overflow: 'auto', mb: 2 }}>
            <List dense>
              {files.map((file, index) => (
                <ListItem
                  key={index}
                  secondaryAction={
                    <Button
                      edge="end"
                      aria-label="delete"
                      onClick={() => handleRemoveFile(index)}
                      disabled={isProcessing}
                    >
                      <DeleteIcon color="error" />
                    </Button>
                  }
                >
                  <ListItemIcon>
                    {getFileIcon(file)}
                  </ListItemIcon>
                  <ListItemText
                    primary={file.name}
                    secondary={`${(file.size / 1024).toFixed(2)} KB`}
                  />
                </ListItem>
              ))}
            </List>
          </Paper>
          
          <Grid container spacing={2}>
            <Grid item xs={12} sm={6}>
              <Button
                variant="outlined"
                color="primary"
                onClick={handleUploadClick}
                fullWidth
                disabled={isProcessing}
                startIcon={<CloudUploadIcon />}
              >
                Add More Files
              </Button>
            </Grid>
            <Grid item xs={12} sm={6}>
              <Button
                variant="contained"
                color="primary"
                onClick={handleProcessFiles}
                fullWidth
                disabled={isProcessing || files.length === 0}
              >
                {isProcessing ? 'Processing...' : 'Process Files'}
              </Button>
            </Grid>
          </Grid>
          
          {isProcessing && (
            <Box mt={2}>
              <Typography variant="body2" gutterBottom>
                Processing files... {processingProgress}%
              </Typography>
              <LinearProgress 
                variant="determinate" 
                value={processingProgress} 
                sx={{ height: 8, borderRadius: 4 }}
              />
            </Box>
          )}
        </Box>
      )}
    </Box>
  );
}

export default FileUpload;
