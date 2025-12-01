
import { GoogleGenAI } from "@google/genai";
import { BacktestResults, StressTestType } from "../types";

// The API Key check is moved inside the function to avoid a module-level crash
// and provide a more graceful error handling flow.

export const analyzeResultsWithGemini = async (
  results: BacktestResults,
  totalDraws: number,
  stressTest?: StressTestType
): Promise<string> => {
  // Simulate AI failure for stress testing
  if (stressTest === 'AI_FAILURE') {
    throw new Error("Simulated Error: The AI analysis API is currently unavailable.");
  }
  // Simulate an empty but successful response for stress testing
  if (stressTest === 'AI_EMPTY_RESPONSE') {
    return "";
  }

  const API_KEY = process.env.API_KEY;

  // Robust check for API key at time of use.
  if (!API_KEY) {
    console.error("Gemini API key is not configured.");
    throw new Error("AI analysis is not available: The API key has not been configured by the developer.");
  }

  const ai = new GoogleGenAI({ apiKey: API_KEY });
  const model = "gemini-2.5-flash";

  const prompt = `
You are a statistical analyst specializing in games of chance. Your task is to analyze the results of a simulation run against historical Powerball lottery data.

The simulation generated one random ticket for each of the ${totalDraws} historical drawings. The goal of this simulation is to establish a clear performance baseline for a purely random ticket-picking strategy.

Here are the results of the simulation, showing how many times each prize tier was won:

${JSON.stringify(results, null, 2)}

Please provide a detailed analysis in the following structure, using clear text headers and paragraphs. Do not use markdown like '#' or '*'.

1. EXECUTIVE SUMMARY:
Start with a brief, high-level overview of the findings. State clearly whether the random agent performed as expected according to the laws of probability.

2. PERFORMANCE BREAKDOWN:
For each significant prize tier that had at least one win, compare the simulated frequency (e.g., "1 win in ${totalDraws} draws") with the known statistical odds of winning that prize. For example, the odds of matching just the Powerball are 1 in 38.32. Discuss any notable deviations, but attribute them to normal statistical variance expected in a random sample of this size.

3. CONCLUSION ON BASELINE:
Conclude by affirming that this simulation successfully establishes a random performance baseline. Emphasize that this baseline represents the 'luck' that any future, more intelligent agent must consistently and significantly beat to be considered truly effective and not just lucky.
`;

  try {
    const response = await ai.models.generateContent({
      model: model,
      contents: prompt,
    });

    const text = response.text;
    if (!text || text.trim() === '') {
        throw new Error("The AI model returned an empty or invalid response.");
    }
    return text;
  } catch (error: any) {
    console.error("Gemini API call failed:", error);
    // Provide a more specific and helpful error message to the user.
    throw new Error(`The AI analysis request failed. Please check your network connection or API key configuration. Details: ${error.message}`);
  }
};
