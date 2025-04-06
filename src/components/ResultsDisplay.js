import React, { useState } from 'react';
import { 
  Box, 
  Paper, 
  Typography, 
  Tabs, 
  Tab, 
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  IconButton,
  Tooltip,
  Grid
} from '@mui/material';
import SaveAltIcon from '@mui/icons-material/SaveAlt';
import FileDownloadIcon from '@mui/icons-material/FileDownload';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import FullscreenIcon from '@mui/icons-material/Fullscreen';
import FullscreenExitIcon from '@mui/icons-material/FullscreenExit';
import { saveAs } from 'file-saver';
import ReactMarkdown from 'react-markdown';

// This function would typically come from a library like html2canvas or similar
const exportAsPNG = async (elementId, filename) => {
  // Placeholder for actual implementation
  alert(`Exporting ${filename} as PNG. In a real implementation, this would capture the content of the visualization.`);
};

// This function would typically use a PDF generation library like jsPDF
const exportAsPDF = async (elementId, filename) => {
  // Placeholder for actual implementation
  alert(`Exporting ${filename} as PDF. In a real implementation, this would generate a high-resolution PDF.`);
};

// Tab panel component
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`results-tabpanel-${index}`}
      aria-labelledby={`results-tab-${index}`}
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

function ResultsDisplay({ results }) {
  const [tabValue, setTabValue] = useState(0);
  const [fullscreen, setFullscreen] = useState(false);

  // If no results are available yet
  if (!results) {
    return (
      <Paper 
        elevation={2} 
        sx={{ p: 3, mt: 2, textAlign: 'center', minHeight: '200px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      >
        <Box>
          <Typography variant="body1" color="text.secondary">
            Your analysis results will appear here after you submit a query.
          </Typography>
          <Typography variant="body2" color="text.secondary" mt={1}>
            Try asking for descriptive statistics, correlations, or visualizations.
          </Typography>
        </Box>
      </Paper>
    );
  }

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const handleCopyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    // Could show a small notification here
  };

  const handleToggleFullscreen = () => {
    setFullscreen(!fullscreen);
  };

  return (
    <Paper 
      elevation={2} 
      sx={{ 
        mt: 2, 
        display: 'flex', 
        flexDirection: 'column',
        height: fullscreen ? 'calc(100vh - 140px)' : '500px',
        position: fullscreen ? 'fixed' : 'relative',
        top: fullscreen ? '70px' : 'auto',
        left: fullscreen ? '0' : 'auto',
        right: fullscreen ? '0' : 'auto',
        zIndex: fullscreen ? 1200 : 1,
        width: fullscreen ? '100%' : 'auto',
        transition: 'all 0.3s ease'
      }}
    >
      <Box display="flex" justifyContent="space-between" alignItems="center" p={2} pb={1}>
        <Typography variant="h6">Analysis Results</Typography>
        <Box>
          <Tooltip title="Toggle fullscreen">
            <IconButton onClick={handleToggleFullscreen} size="small">
              {fullscreen ? <FullscreenExitIcon /> : <FullscreenIcon />}
            </IconButton>
          </Tooltip>
          <Tooltip title="Download all results">
            <IconButton size="small">
              <FileDownloadIcon />
            </IconButton>
          </Tooltip>
        </Box>
      </Box>
      
      <Divider />
      
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs 
          value={tabValue} 
          onChange={handleTabChange} 
          aria-label="results tabs"
          variant="scrollable"
          scrollButtons="auto"
        >
          <Tab label="Summary" />
          <Tab label="Tables" />
          <Tab label="Visualizations" />
          <Tab label="Statistics" />
          <Tab label="Code" />
        </Tabs>
      </Box>
      
      <Box flexGrow={1} sx={{ overflow: 'hidden' }}>
        {/* Summary Tab */}
        <TabPanel value={tabValue} index={0}>
          <Box sx={{ p: 1 }}>
            <ReactMarkdown>
              {results.summary || "No summary information available."}
            </ReactMarkdown>
          </Box>
        </TabPanel>
        
        {/* Tables Tab */}
        <TabPanel value={tabValue} index={1}>
          {results.tables && results.tables.length > 0 ? (
            results.tables.map((table, index) => (
              <Box key={index} mt={index > 0 ? 4 : 0} id={`table-${index}`}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="subtitle1">{table.title || `Table ${index + 1}`}</Typography>
                  <Box>
                    <Tooltip title="Copy as CSV">
                      <IconButton size="small" onClick={() => handleCopyToClipboard(table.csvContent || "")}>
                        <ContentCopyIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                    <Tooltip title="Download as CSV">
                      <IconButton size="small" onClick={() => saveAs(new Blob([table.csvContent || ""], {type: "text/csv;charset=utf-8"}), `table-${index + 1}.csv`)}>
                        <SaveAltIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </Box>
                
                {table.description && (
                  <Typography variant="body2" color="text.secondary" paragraph>
                    {table.description}
                  </Typography>
                )}
                
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        {table.headers.map((header, idx) => (
                          <TableCell key={idx}>{header}</TableCell>
                        ))}
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {table.rows.map((row, rowIdx) => (
                        <TableRow key={rowIdx}>
                          {row.map((cell, cellIdx) => (
                            <TableCell key={cellIdx}>{cell}</TableCell>
                          ))}
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </Box>
            ))
          ) : (
            <Typography variant="body1" color="text.secondary" align="center">
              No tables available. Try asking for tabular data or descriptive statistics.
            </Typography>
          )}
        </TabPanel>
        
        {/* Visualizations Tab */}
        <TabPanel value={tabValue} index={2}>
          {results.visualizations && results.visualizations.length > 0 ? (
            <Grid container spacing={3}>
              {results.visualizations.map((viz, index) => (
                <Grid item xs={12} key={index}>
                  <Paper variant="outlined" sx={{ p: 2 }} id={`viz-${index}`}>
                    <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                      <Typography variant="subtitle1">{viz.title || `Figure ${index + 1}`}</Typography>
                      <Box>
                        <Tooltip title="Download as PNG">
                          <IconButton size="small" onClick={() => exportAsPNG(`viz-${index}`, `figure-${index + 1}.png`)}>
                            <SaveAltIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                        <Tooltip title="Download as PDF (300 dpi)">
                          <IconButton size="small" onClick={() => exportAsPDF(`viz-${index}`, `figure-${index + 1}.pdf`)}>
                            <FileDownloadIcon fontSize="small" />
                          </IconButton>
                        </Tooltip>
                      </Box>
                    </Box>
                    
                    {viz.description && (
                      <Typography variant="body2" color="text.secondary" paragraph>
                        {viz.description}
                      </Typography>
                    )}
                    
                    {/* The visualization content would be inserted here */}
                    <Box 
                      sx={{ 
                        minHeight: '300px', 
                        bgcolor: 'background.default',
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center'
                      }}
                      dangerouslySetInnerHTML={{ __html: viz.content || "" }}
                    />
                  </Paper>
                </Grid>
              ))}
            </Grid>
          ) : (
            <Typography variant="body1" color="text.secondary" align="center">
              No visualizations available. Try asking for a plot or chart.
            </Typography>
          )}
        </TabPanel>
        
        {/* Statistics Tab */}
        <TabPanel value={tabValue} index={3}>
          {results.statistics && results.statistics.length > 0 ? (
            results.statistics.map((stat, index) => (
              <Paper key={index} variant="outlined" sx={{ p: 2, mb: 2 }}>
                <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                  <Typography variant="subtitle1">{stat.title || `Statistical Test ${index + 1}`}</Typography>
                  <Tooltip title="Copy results">
                    <IconButton size="small" onClick={() => handleCopyToClipboard(stat.text || "")}>
                      <ContentCopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
                
                <Box sx={{ p: 1 }}>
                  <ReactMarkdown>
                    {stat.text || "No results available."}
                  </ReactMarkdown>
                </Box>
                
                {stat.interpretation && (
                  <Box mt={2} p={1} bgcolor="background.default" borderRadius={1}>
                    <Typography variant="subtitle2" gutterBottom>
                      Interpretation:
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {stat.interpretation}
                    </Typography>
                  </Box>
                )}
              </Paper>
            ))
          ) : (
            <Typography variant="body1" color="text.secondary" align="center">
              No statistical results available. Try asking for a specific statistical test or analysis.
            </Typography>
          )}
        </TabPanel>
        
        {/* Code Tab */}
        <TabPanel value={tabValue} index={4}>
          {results.code ? (
            <Box>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={1}>
                <Typography variant="subtitle1">Python Code</Typography>
                <Box>
                  <Tooltip title="Copy code">
                    <IconButton size="small" onClick={() => handleCopyToClipboard(results.code)}>
                      <ContentCopyIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                  <Tooltip title="Download as .py file">
                    <IconButton size="small" onClick={() => saveAs(new Blob([results.code], {type: "text/plain;charset=utf-8"}), "analysis_code.py")}>
                      <SaveAltIcon fontSize="small" />
                    </IconButton>
                  </Tooltip>
                </Box>
              </Box>
              
              <Paper 
                variant="outlined" 
                sx={{ 
                  p: 2, 
                  bgcolor: '#f5f5f5', 
                  fontFamily: 'monospace',
                  whiteSpace: 'pre-wrap',
                  overflow: 'auto',
                  borderRadius: 1
                }}
              >
                <code>{results.code}</code>
              </Paper>
              
              <Box mt={2} mb={1} display="flex" justifyContent="flex-end">
                <Button 
                  variant="contained" 
                  color="primary" 
                  startIcon={<FileDownloadIcon />}
                  onClick={() => saveAs(new Blob([results.code], {type: "text/plain;charset=utf-8"}), "analysis_code.py")}
                >
                  Download Complete Analysis Script
                </Button>
              </Box>
              
              <Typography variant="body2" color="text.secondary">
                This Python code reproduces all the analyses and visualizations shown in the results.
              </Typography>
            </Box>
          ) : (
            <Typography variant="body1" color="text.secondary" align="center">
              No code available. Code will be generated when you run a specific analysis.
            </Typography>
          )}
        </TabPanel>
      </Box>
    </Paper>
  );
}

export default ResultsDisplay;
