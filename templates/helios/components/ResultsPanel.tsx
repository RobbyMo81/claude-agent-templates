
import React from 'react';
import { BacktestResults, PrizeTier } from '../types';
import Spinner from './Spinner';
import Alert from './Alert';
import { PRIZE_TIER_ORDER } from '../constants';

interface ResultsPanelProps {
  isLoading: boolean;
  error: string | null;
  results: BacktestResults | null;
  aiAnalysis: string | null;
  totalDraws: number;
}

const WelcomeMessage: React.FC = () => (
    <div className="text-center p-8 bg-gray-800/50 rounded-lg border border-gray-700">
        <h2 className="text-3xl font-bold text-blue-300 mb-2">Welcome to Helios Mission Control</h2>
        <p className="text-lg text-gray-300 mb-4">The Anomaly Detection Engine</p>
        <p className="text-gray-400 max-w-2xl mx-auto">
            To begin, please upload your historical Powerball CSV data using the sidebar and launch the baseline analysis. The system will run a simulation of a purely random agent to establish the statistical ground truth.
        </p>
    </div>
);

const AnalysisContent: React.FC<{ analysis: string }> = ({ analysis }) => {
    // Simple parser to format the AI response
    const paragraphs = analysis.split('\n').filter(p => p.trim() !== '');
    return (
        <div className="space-y-4 text-gray-300">
            {paragraphs.map((p, i) => {
                if (p.match(/^\d\.\s[A-Z\s]+:/)) { // Matches "1. EXECUTIVE SUMMARY:"
                    return <h3 key={i} className="text-xl font-semibold text-blue-300 mt-6 mb-2">{p}</h3>;
                }
                return <p key={i} className="leading-relaxed">{p}</p>;
            })}
        </div>
    );
};

const ResultsPanel: React.FC<ResultsPanelProps> = ({ isLoading, error, results, aiAnalysis, totalDraws }) => {
  const renderContent = () => {
    if (isLoading) {
      return <Spinner />;
    }
    if (error) {
      return <Alert message={error} type="error" />;
    }
    if (results) {
      const sortedResults = (Object.entries(results) as [PrizeTier, number][])
        .sort(([tierA], [tierB]) => PRIZE_TIER_ORDER.indexOf(tierA) - PRIZE_TIER_ORDER.indexOf(tierB));

      return (
        <div className="space-y-8">
            <div>
                <h2 className="text-3xl font-bold text-white mb-4">Level 0: Baseline Performance Report</h2>
                <p className="text-gray-400">Analysis of a purely random agent over {totalDraws.toLocaleString()} historical draws.</p>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
              <h3 className="text-2xl font-semibold text-blue-300 mb-4">Simulation Hit Count</h3>
              <div className="overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-700">
                      <thead className="bg-gray-800">
                          <tr>
                              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Prize Tier</th>
                              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Hit Count</th>
                              <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Observed Frequency (1 in X)</th>
                          </tr>
                      </thead>
                      <tbody className="bg-gray-900/50 divide-y divide-gray-700">
                          {sortedResults.map(([tier, count]) => (
                              <tr key={tier}>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-200">{tier}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{count.toLocaleString()}</td>
                                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{count > 0 ? (totalDraws / count).toFixed(2) : 'N/A'}</td>
                              </tr>
                          ))}
                      </tbody>
                  </table>
              </div>
            </div>

            <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
                <h3 className="text-2xl font-semibold text-blue-300 mb-4">AI-Powered Analysis</h3>
                {aiAnalysis ? <AnalysisContent analysis={aiAnalysis} /> : <p className="text-gray-400">Generating analysis...</p>}
            </div>
        </div>
      );
    }
    return <WelcomeMessage />;
  };

  return (
    <main className="flex-1 p-8 overflow-y-auto">
        <div className="max-w-7xl mx-auto">
            {renderContent()}
        </div>
    </main>
  );
};

export default ResultsPanel;
