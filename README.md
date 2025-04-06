# AzzamAnalyst

[![Deploy to GitHub Pages](https://github.com/drazzam/AzzamAnalyst/actions/workflows/deploy.yml/badge.svg)](https://github.com/drazzam/AzzamAnalyst/actions/workflows/deploy.yml)

AzzamAnalyst is a revolutionary web-based biostatistical AI agent that combines advanced data analysis capabilities with an intuitive user interface. It offers a comprehensive solution for statistical analysis, visualization, and reporting, aimed at researchers and professionals in biomedical and clinical fields.

## Features

- **Intelligent Data Processing**: Automatically cleans, validates, and prepares data for analysis
- **Advanced Statistical Analysis**: Performs a wide range of statistical tests and analyses
- **Interactive Visualizations**: Creates publication-quality graphs and charts
- **AI-Powered Insights**: Leverages Google Gemini AI to interpret data and explain findings
- **Multi-format Support**: Handles various file formats (CSV, Excel, PDF, DOCX, TXT)
- **Natural Language Interface**: Allows you to describe analyses in plain English
- **Export Capabilities**: Save results as high-resolution images or PDFs

## Getting Started

### Prerequisites

- A Google Gemini API key (obtain from [Google AI Studio](https://ai.google.dev/))
- A modern web browser (Chrome, Firefox, Edge, or Safari)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/drazzam/AzzamAnalyst.git
cd AzzamAnalyst
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

4. Build for production:
```bash
npm run build
```

### Deployment

This project is configured for automatic deployment to GitHub Pages. Simply push to the main branch, and GitHub Actions will handle the deployment process.

To deploy manually:
```bash
npm run deploy
```

## Usage

1. **API Setup**: Enter your Google Gemini API key when prompted
2. **Upload Data**: Upload your data files in any supported format
3. **Natural Language Analysis**: Simply ask questions about your data
4. **View Results**: Explore statistical results and visualizations
5. **Export**: Download results in various formats

### Example Queries

- "Show me descriptive statistics for all variables"
- "Run a t-test comparing treatment and control groups"
- "Create a scatter plot of age vs. weight and add a regression line"
- "Perform a survival analysis on time to event data"
- "Check for outliers and show their distribution"
- "Analyze the correlation between all numeric variables"

## Architecture

AzzamAnalyst uses a browser-based architecture that runs entirely on GitHub Pages:

- **Frontend**: React with Material-UI for the user interface
- **Data Processing**: Python code executed in the browser using Pyodide/WebAssembly
- **AI Integration**: Google Gemini API for natural language understanding
- **Data Visualization**: Python's matplotlib, seaborn, and other libraries
- **Statistical Analysis**: Python's scipy, statsmodels, and scikit-learn libraries

## Statistical Capabilities

AzzamAnalyst supports a comprehensive range of statistical analyses:

- **Descriptive Statistics**: Mean, median, mode, standard deviation, etc.
- **Inferential Statistics**: t-tests, ANOVA, chi-square, etc.
- **Regression Analysis**: Linear, multiple, logistic regression
- **Survival Analysis**: Kaplan-Meier, Cox proportional hazards
- **Dimensional Reduction**: PCA, factor analysis
- **Clustering**: K-means, hierarchical clustering
- **Correlation Analysis**: Pearson, Spearman, partial correlations
- **Time Series Analysis**: Trends, seasonality, forecasting
- **Power Analysis**: Sample size calculations

## Data Cleaning Capabilities

AzzamAnalyst automatically performs the following data cleaning operations:

- Standardizing missing values
- Detecting and handling outliers
- Converting data types
- Normalizing text and categorical variables
- Identifying duplicate data
- Handling inconsistent formatting
- Validating data integrity

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Pyodide](https://pyodide.org/) for making Python in the browser possible
- [Google Gemini API](https://ai.google.dev/) for the AI capabilities
- [React](https://reactjs.org/) and [Material-UI](https://material-ui.com/) for the frontend framework
- [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/), and other Python libraries for data visualization
- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), and other Python libraries for data processing
- [SciPy](https://scipy.org/), [statsmodels](https://www.statsmodels.org/), and [scikit-learn](https://scikit-learn.org/) for statistical analysis
