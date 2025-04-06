import React, { useState } from 'react';
import { 
  Box, 
  Button, 
  Dialog, 
  DialogTitle, 
  DialogContent, 
  DialogActions, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  TextField, 
  Typography,
  Paper,
  Grid,
  Checkbox,
  FormControlLabel,
  CircularProgress
} from '@mui/material';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import PictureAsPdfIcon from '@mui/icons-material/PictureAsPdf';
import ImageIcon from '@mui/icons-material/Image';
import DescriptionIcon from '@mui/icons-material/Description';
import SaveIcon from '@mui/icons-material/Save';
import { saveAs } from 'file-saver';

// This function would typically use a library like html2canvas
const captureElement = async (elementId) => {
  // Placeholder for actual implementation
  console.log(`Capturing element with ID: ${elementId}`);
  return new Promise(resolve => setTimeout(() => resolve('data:image/png;base64,dummy'), 500));
};

// This function would typically use a PDF generation library like jsPDF
const generatePDF = async (content, options) => {
  // Placeholder for actual implementation
  console.log('Generating PDF with content:', content, 'and options:', options);
  return new Promise(resolve => setTimeout(() => resolve(new Blob(['dummy pdf content'], {type: 'application/pdf'})), 500));
};

function ExportTools({ results, selectedItem, exportType }) {
  const [open, setOpen] = useState(false);
  const [format, setFormat] = useState(exportType || 'png');
  const [quality, setQuality] = useState('high');
  const [filename, setFilename] = useState(
    selectedItem ? 
    `${selectedItem.title.replace(/\s+/g, '_').toLowerCase()}_export` : 
    'azzamanalyst_export'
  );
  const [includeMetadata, setIncludeMetadata] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);

  const handleOpen = () => {
    setOpen(true);
  };

  const handleClose = () => {
    setOpen(false);
  };

  const handleExport = async () => {
    setIsProcessing(true);
    
    try {
      if (format === 'png' || format === 'jpg') {
        // For image exports
        const elementId = selectedItem?.id || 'results-display';
        const dataUrl = await captureElement(elementId);
        
        // Convert data URL to Blob
        const res = await fetch(dataUrl);
        const blob = await res.blob();
        
        // Save file
        saveAs(blob, `${filename}.${format}`);
      } else if (format === 'pdf') {
        // For PDF exports
        const pdfOptions = {
          quality: quality === 'high' ? 300 : 150,
          includeMetadata: includeMetadata,
          title: selectedItem?.title || 'Analysis Results',
          author: 'AzzamAnalyst'
        };
        
        const pdfBlob = await generatePDF(results, pdfOptions);
        saveAs(pdfBlob, `${filename}.pdf`);
      } else if (format === 'csv' && selectedItem?.csvContent) {
        // For CSV exports
        const blob = new Blob([selectedItem.csvContent], { type: 'text/csv;charset=utf-8' });
        saveAs(blob, `${filename}.csv`);
      } else if (format === 'txt') {
        // For text exports
        let content = '';
        
        if (selectedItem?.text) {
          content = selectedItem.text;
        } else if (selectedItem?.title && selectedItem?.description) {
          content = `${selectedItem.title}\n\n${selectedItem.description}`;
        } else if (results?.summary) {
          content = results.summary;
        }
        
        const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
        saveAs(blob, `${filename}.txt`);
      }
    } catch (error) {
      console.error('Error during export:', error);
      alert('Export failed. Please try again.');
    } finally {
      setIsProcessing(false);
      handleClose();
    }
  };

  return (
    <Box>
      <Button
        variant="outlined"
        color="primary"
        startIcon={<SaveAltIcon />}
        onClick={handleOpen}
        size="small"
      >
        Export
      </Button>
      
      <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
        <DialogTitle>Export Options</DialogTitle>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                margin="normal"
                label="Filename"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Export Format</InputLabel>
                <Select
                  value={format}
                  onChange={(e) => setFormat(e.target.value)}
                  label="Export Format"
                >
                  <MenuItem value="png">
                    <Box display="flex" alignItems="center">
                      <ImageIcon fontSize="small" sx={{ mr: 1 }} />
                      PNG Image
                    </Box>
                  </MenuItem>
                  <MenuItem value="jpg">
                    <Box display="flex" alignItems="center">
                      <ImageIcon fontSize="small" sx={{ mr: 1 }} />
                      JPG Image
                    </Box>
                  </MenuItem>
                  <MenuItem value="pdf">
                    <Box display="flex" alignItems="center">
                      <PictureAsPdfIcon fontSize="small" sx={{ mr: 1 }} />
                      PDF Document
                    </Box>
                  </MenuItem>
                  {selectedItem?.csvContent && (
                    <MenuItem value="csv">
                      <Box display="flex" alignItems="center">
                        <DescriptionIcon fontSize="small" sx={{ mr: 1 }} />
                        CSV Data
                      </Box>
                    </MenuItem>
                  )}
                  <MenuItem value="txt">
                    <Box display="flex" alignItems="center">
                      <DescriptionIcon fontSize="small" sx={{ mr: 1 }} />
                      Text Document
                    </Box>
                  </MenuItem>
                </Select>
              </FormControl>
            </Grid>
            
            {(format === 'png' || format === 'jpg' || format === 'pdf') && (
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth margin="normal">
                  <InputLabel>Quality</InputLabel>
                  <Select
                    value={quality}
                    onChange={(e) => setQuality(e.target.value)}
                    label="Quality"
                  >
                    <MenuItem value="standard">Standard</MenuItem>
                    <MenuItem value="high">High (300 DPI)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
            )}
            
            {format === 'pdf' && (
              <Grid item xs={12}>
                <FormControlLabel
                  control={
                    <Checkbox
                      checked={includeMetadata}
                      onChange={(e) => setIncludeMetadata(e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Include metadata (date, time, analysis parameters)"
                />
              </Grid>
            )}
            
            <Grid item xs={12}>
              <Paper variant="outlined" sx={{ p: 2, mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Export Preview
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {selectedItem ? (
                    <>Exporting: <strong>{selectedItem.title}</strong></>
                  ) : (
                    'Exporting all current results'
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Format: <strong>{format.toUpperCase()}</strong>
                  {(format === 'png' || format === 'jpg' || format === 'pdf') && (
                    <>, Quality: <strong>{quality === 'high' ? 'High (300 DPI)' : 'Standard'}</strong></>
                  )}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Filename: <strong>{filename}.{format}</strong>
                </Typography>
              </Paper>
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleClose} disabled={isProcessing}>
            Cancel
          </Button>
          <Button 
            onClick={handleExport} 
            color="primary" 
            variant="contained"
            startIcon={isProcessing ? <CircularProgress size={20} /> : <SaveIcon />}
            disabled={isProcessing}
          >
            {isProcessing ? 'Processing...' : 'Export'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

export default ExportTools;
