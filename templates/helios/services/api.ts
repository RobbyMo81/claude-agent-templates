
import { parseCSV, runBacktest } from './lotteryService';
import { analyzeResultsWithGemini } from './geminiService';
import { BacktestResults, HistoricalDraw, StressTestType } from '../types';

/**
 * A structured payload containing all the data from a successful analysis.
 */
export interface AnalysisPayload {
    results: BacktestResults;
    aiAnalysis: string;
    totalDraws: number;
}

/**
 * The unified API endpoint for running a full analysis.
 * This function orchestrates the entire process:
 * 1. Parsing and validating the user-provided CSV file.
 * 2. Running the backtest simulation with a random agent.
 * 3. Calling the Gemini API to get a statistical analysis of the results.
 *
 * It's designed to be robust, with clear error messages for each step.
 *
 * @param file The CSV file to analyze, typically from a file input.
 * @param stressTest An optional flag to trigger specific test scenarios.
 * @returns A promise that resolves with the `AnalysisPayload` containing all results.
 * @throws An error with a user-friendly message if any step of the process fails.
 */
export const runFullAnalysis = async (file: File, stressTest?: StressTestType): Promise<AnalysisPayload> => {
    // Simulate a slow backtest for stress testing
    if (stressTest === 'SLOW_BACKTEST') {
        await new Promise(res => setTimeout(res, 3000));
    }

    // Step 1: Parse and Validate Data
    // This can throw errors related to file format or missing columns.
    const historicalData: HistoricalDraw[] = await parseCSV(file);

    // Ensure the file contains actionable data after parsing.
    if (historicalData.length === 0) {
        throw new Error("Analysis Aborted: The CSV file is empty or contains no valid data rows.");
    }
    const totalDraws = historicalData.length;

    // Step 2: Run the local backtest simulation. This is a pure function and unlikely to fail.
    const backtestResults = runBacktest(historicalData);

    // Step 3: Get AI-powered analysis. This can throw errors related to API keys or network issues.
    const aiAnalysis = await analyzeResultsWithGemini(backtestResults, totalDraws, stressTest);

    // Step 4: Return the complete, unified payload.
    return {
        results: backtestResults,
        aiAnalysis,
        totalDraws,
    };
};
