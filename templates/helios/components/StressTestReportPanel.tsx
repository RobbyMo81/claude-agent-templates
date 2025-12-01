import React from 'react';
import { TestReportEntry } from '../types';
import Alert from './Alert';

interface StressTestReportPanelProps {
  log: TestReportEntry[];
}

const StressTestReportPanel: React.FC<StressTestReportPanelProps> = ({ log }) => {
    const totalTests = log.length;
    const passedTests = log.filter(entry => entry.status === 'PASS').length;
    const failedTests = totalTests - passedTests;

    const renderContent = () => {
        if (log.length === 0) {
            return (
                <div className="text-center p-8 bg-gray-800/50 rounded-lg border border-gray-700">
                    <h2 className="text-3xl font-bold text-yellow-300 mb-2">System Performance Report</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        No performance tests have been run yet. Use the "Developer Tools" in the sidebar to run individual or comprehensive stress tests.
                    </p>
                </div>
            );
        }

        return (
             <div className="space-y-8">
                <div>
                    <h2 className="text-3xl font-bold text-white mb-2">System Performance Report</h2>
                    <p className="text-gray-400">A log of all major system operations and their performance.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
                     <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700">
                        <h4 className="text-sm font-medium text-gray-400 uppercase">Total Tests Run</h4>
                        <p className="text-2xl font-semibold text-white">{totalTests}</p>
                    </div>
                     <div className="bg-green-800/30 p-4 rounded-lg border border-green-700">
                        <h4 className="text-sm font-medium text-green-300 uppercase">Passed</h4>
                        <p className="text-2xl font-semibold text-white">{passedTests}</p>
                    </div>
                     <div className="bg-red-800/30 p-4 rounded-lg border border-red-700">
                        <h4 className="text-sm font-medium text-red-300 uppercase">Failed</h4>
                        <p className="text-2xl font-semibold text-white">{failedTests}</p>
                    </div>
                </div>

                <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
                  <h3 className="text-2xl font-semibold text-yellow-300 mb-4">Execution Log</h3>
                  <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-700">
                          <thead className="bg-gray-800">
                              <tr>
                                  <th scope="col" className="px-4 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Status</th>
                                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Test Name</th>
                                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Duration (ms)</th>
                                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Details</th>
                                  <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Timestamp</th>
                              </tr>
                          </thead>
                          <tbody className="bg-gray-900/50 divide-y divide-gray-700">
                              {log.map((entry) => (
                                  <tr key={entry.id}>
                                      <td className="px-4 py-4 whitespace-nowrap">
                                          <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${entry.status === 'PASS' ? 'bg-green-900 text-green-200' : 'bg-red-900 text-red-200'}`}>
                                            {entry.status}
                                          </span>
                                      </td>
                                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-200">{entry.testName}</td>
                                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">{entry.durationMs.toFixed(2)}</td>
                                      <td className="px-6 py-4 text-sm text-gray-400 max-w-sm truncate" title={entry.details}>{entry.details}</td>
                                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{entry.timestamp}</td>
                                  </tr>
                              ))}
                          </tbody>
                      </table>
                  </div>
                </div>
            </div>
        );
    }

    return (
        <main className="flex-1 p-8 overflow-y-auto">
            <div className="max-w-7xl mx-auto">
                {renderContent()}
            </div>
        </main>
    );
};

export default StressTestReportPanel;
