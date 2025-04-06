/**
 * Service for interacting with the Google Gemini API
 */

/**
 * Test if the provided API key is valid
 * @param {string} apiKey - Gemini API key to test
 * @returns {Promise<boolean>} - Whether the API key is valid
 */
export const testGeminiApiKey = async (apiKey) => {
  try {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: "Hello, can you respond with just the word 'valid' to test API connectivity?"
              }
            ]
          }
        ],
        generationConfig: {
          temperature: 0,
          maxOutputTokens: 10,
        }
      })
    });

    const data = await response.json();
    
    // Check if we got a valid response
    if (data.candidates && data.candidates.length > 0) {
      return true;
    } else if (data.error) {
      console.error('API key validation error:', data.error);
      return false;
    }
    
    return false;
  } catch (error) {
    console.error('Error testing Gemini API key:', error);
    return false;
  }
};

/**
 * Generate content using the Gemini API
 * @param {string} prompt - Prompt to send to Gemini
 * @param {string} apiKey - Gemini API key
 * @param {object} options - Additional options for the API call
 * @returns {Promise<object>} - Gemini response
 */
export const generateContent = async (prompt, apiKey, options = {}) => {
  try {
    const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${apiKey}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [
          {
            parts: [
              {
                text: prompt
              }
            ]
          }
        ],
        generationConfig: {
          temperature: options.temperature || 0.2,
          maxOutputTokens: options.maxTokens || 1024,
        }
      })
    });

    const data = await response.json();
    
    if (data.error) {
      throw new Error(`Gemini API error: ${data.error.message}`);
    }
    
    if (!data.candidates || data.candidates.length === 0) {
      throw new Error('No response generated from Gemini API');
    }
    
    return data;
  } catch (error) {
    console.error('Error generating content with Gemini API:', error);
    throw error;
  }
};

/**
 * Extract text from Gemini API response
 * @param {object} geminiResponse - Response from Gemini API
 * @returns {string} - Extracted text
 */
export const extractTextFromResponse = (geminiResponse) => {
  if (!geminiResponse.candidates || geminiResponse.candidates.length === 0) {
    return '';
  }
  
  const candidate = geminiResponse.candidates[0];
  
  if (!candidate.content || !candidate.content.parts || candidate.content.parts.length === 0) {
    return '';
  }
  
  return candidate.content.parts[0].text || '';
};

/**
 * Generate a biostatistics analysis plan based on dataset and query
 * @param {string} userQuery - User's query about the data
 * @param {object} datasetInfo - Information about the dataset
 * @param {string} apiKey - Gemini API key
 * @returns {Promise<object>} - Analysis plan
 */
export const generateAnalysisPlan = async (userQuery, datasetInfo, apiKey) => {
  try {
    const prompt = `
You are a biostatistical AI assistant. Based on the user's query and the available data, determine what analysis to perform.

USER QUERY: ${userQuery}

AVAILABLE DATA:
${JSON.stringify(datasetInfo, null, 2)}

Return a structured JSON response with the following fields:
1. "analysisType": The type of analysis to perform (e.g., "descriptive_statistics", "hypothesis_test", "correlation", "regression", "visualization")
2. "dataset": The name of the primary dataset to use
3. "variables": Array of relevant variables/columns to include
4. "additionalParameters": Any specific parameters needed for the analysis
5. "explanation": Brief explanation of what analysis will be performed and why
6. "visualizationType": If visualization is needed, what type (e.g., "histogram", "scatter", "boxplot")

RESPONSE (JSON format only):
`;

    const response = await generateContent(prompt, apiKey, { temperature: 0.1 });
    const textResponse = extractTextFromResponse(response);
    
    // Parse JSON from the response
    try {
      // Try to extract JSON if it's wrapped in markdown code blocks
      const jsonMatch = textResponse.match(/```json\n([\s\S]*?)\n```/) || 
                        textResponse.match(/```\n([\s\S]*?)\n```/) || 
                        textResponse.match(/{[\s\S]*}/);
      
      if (jsonMatch) {
        return JSON.parse(jsonMatch[1] || jsonMatch[0]);
      } else {
        // Try to parse the entire response as JSON
        return JSON.parse(textResponse);
      }
    } catch (error) {
      console.error('Error parsing analysis plan JSON:', error);
      throw new Error('Failed to generate a valid analysis plan');
    }
  } catch (error) {
    console.error('Error generating analysis plan:', error);
    throw error;
  }
};

/**
 * Generate Python code for statistical analysis
 * @param {object} analysisPlan - Analysis plan from generateAnalysisPlan
 * @param {object} datasetInfo - Information about the dataset
 * @param {string} apiKey - Gemini API key
 * @returns {Promise<string>} - Python code for analysis
 */
export const generatePythonCode = async (analysisPlan, datasetInfo, apiKey) => {
  try {
    const prompt = `
You are a biostatistical AI assistant. Generate Python code for the requested analysis.

ANALYSIS PLAN:
${JSON.stringify(analysisPlan, null, 2)}

AVAILABLE DATA:
${JSON.stringify(datasetInfo, null, 2)}

Generate complete, executable Python code for performing this analysis. Use pandas, numpy, scipy, statsmodels, matplotlib, and seaborn as appropriate.

Make sure to include:
1. Importing necessary libraries
2. Loading and preprocessing the data
3. Performing the requested analysis
4. Creating appropriate visualizations
5. Interpreting the results

Return the code only, without any explanations or markdown formatting.
`;

    const response = await generateContent(prompt, apiKey, { temperature: 0.1 });
    const textResponse = extractTextFromResponse(response);
    
    // Extract code from markdown if present
    const codeMatch = textResponse.match(/```python\n([\s\S]*?)\n```/) || 
                      textResponse.match(/```\n([\s\S]*?)\n```/);
    
    if (codeMatch) {
      return codeMatch[1];
    }
    
    return textResponse;
  } catch (error) {
    console.error('Error generating Python code:', error);
    throw error;
  }
};

/**
 * Generate a summary of analysis results
 * @param {object} analysisResults - Results from the analysis
 * @param {string} userQuery - Original user query
 * @param {string} apiKey - Gemini API key
 * @returns {Promise<string>} - Textual summary
 */
export const generateResultsSummary = async (analysisResults, userQuery, apiKey) => {
  try {
    const prompt = `
You are a biostatistical AI assistant. Based on the user's query and the analysis results, provide a clear, concise summary of the findings.

USER QUERY: ${userQuery}

ANALYSIS RESULTS:
${JSON.stringify(analysisResults, null, 2)}

Create a summary that:
1. Explains what analysis was performed and why
2. Highlights the most important findings
3. Provides context and interpretation where appropriate
4. Uses clear, professional language suitable for biostatistical reporting
5. Is concise (300-500 words)

SUMMARY:
`;

    const response = await generateContent(prompt, apiKey);
    return extractTextFromResponse(response);
  } catch (error) {
    console.error('Error generating results summary:', error);
    // Provide a basic summary if API call fails
    return "Analysis complete. The results are shown in the tables and visualizations below. You can ask follow-up questions for more specific interpretations.";
  }
};
