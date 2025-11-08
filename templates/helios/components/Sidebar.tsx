import React from 'react';
import FileUploader from './FileUploader';
import { StressTestType, TestReportEntry, BackendStatus } from '../types';
import { TrainingPanel } from './TrainingPanel';
import Alert from './Alert';

interface SidebarProps {
  onFileChange: (file: File) => void;
  selectedFile: File | null;
  onLaunch: () => void;
  isLoading: boolean;
  onStressTest: (testType: StressTestType) => void;
  // Model management
  models: string[];
  selectedModel: string | null;
  onSelectModel: (modelName: string) => void;
  onTrainingComplete: () => void;
  onShowBaseline: () => void;
  onShowTrainingDashboard: () => void;
  onShowMetacognitive: () => void;
  onShowCrossModelAnalytics: () => void;
  // Enhanced model management
  onLoadModel?: (modelName: string) => void;
  onGetInfo?: (modelName: string) => void;
  onPredict?: (modelName: string) => void;
  onDeleteModel?: (modelName: string) => void;
  // Backend status and reporting
  backendStatus: BackendStatus;
  onRetryConnection: () => void;
  onRunComprehensiveTest: () => void;
  onShowReport: () => void;
  reportResult: (result: Omit<TestReportEntry, 'id' | 'timestamp'>) => void;
}

const DevButton: React.FC<{onClick: () => void, children: React.ReactNode}> = ({ onClick, children }) => (
    <button
        onClick={onClick}
        className="w-full text-left text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 rounded-md p-2 transition-colors text-center"
    >
        {children}
    </button>
);


const Sidebar: React.FC<SidebarProps> = ({
    onFileChange, selectedFile, onLaunch, isLoading, onStressTest,
    models, selectedModel, onSelectModel, onTrainingComplete, onShowBaseline, onShowTrainingDashboard, onShowMetacognitive, onShowCrossModelAnalytics,
    onLoadModel, onGetInfo, onPredict, onDeleteModel,
    backendStatus, onRetryConnection, onRunComprehensiveTest, onShowReport, reportResult,
}) => {
  return (
    <aside className="w-96 bg-gray-900/80 border-r border-gray-700/50 p-6 flex flex-col space-y-6 backdrop-blur-sm overflow-y-auto">
      <div className="flex items-center space-x-3 cursor-pointer" onClick={onShowBaseline}>
        <div className="text-4xl">🚀</div>
        <div>
          <h1 className="text-2xl font-bold text-white">Helios</h1>
          <p className="text-sm text-gray-400">Mission Control</p>
        </div>
      </div>

      <div className="flex-grow flex flex-col justify-between space-y-4">
        <div className="space-y-6">
            {/* --- Level 0 --- */}
            <div>
                <h2 className="text-lg font-semibold text-blue-300 mb-2 border-b border-blue-300/20 pb-1">Level 0: Baseline</h2>
                 <div className="bg-gray-800/50 p-4 rounded-lg space-y-3">
                    <FileUploader onFileSelect={onFileChange} selectedFile={selectedFile} />
                    <button
                      onClick={onLaunch}
                      disabled={!selectedFile || isLoading}
                      className="w-full flex items-center justify-center bg-blue-600 hover:bg-blue-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-lg transition-all"
                    >
                      {isLoading ? 'Analyzing...' : 'Launch Baseline'}
                    </button>
                 </div>
            </div>

            {/* --- Level 2 --- */}
            <div>
                <h2 className="text-lg font-semibold text-green-300 mb-2 border-b border-green-300/20 pb-1">Level 2: Agent Training</h2>
                {backendStatus === 'pending' && <div className="text-sm text-gray-400 p-4 text-center">Connecting to backend...</div>}
                {backendStatus === 'disconnected' && (
                    <div className="bg-red-900/50 p-4 rounded-lg space-y-3 border border-red-700">
                        <Alert type="error" message="Backend server is not available." />
                         <button
                          onClick={onRetryConnection}
                          className="w-full flex items-center justify-center bg-yellow-600 hover:bg-yellow-500 text-white font-bold py-2 px-4 rounded-lg transition-colors"
                        >
                          Retry Connection
                        </button>
                    </div>
                )}
                {backendStatus === 'connected' && (
                    <div className="space-y-4">
                        <div>
                            <label htmlFor="model-selector" className="block text-sm font-medium text-gray-300 mb-2">Review Trained Agent</label>
                            <select
                                id="model-selector"
                                value={selectedModel || ''}
                                onChange={(e) => onSelectModel(e.target.value)}
                                className="w-full bg-gray-700 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-green-500 focus:border-green-500"
                                disabled={models.length === 0}
                            >
                                <option value="">{models.length > 0 ? 'Select an agent...' : 'No agents trained'}</option>
                                {models.map(model => (
                                    <option key={model} value={model}>{model}</option>
                                ))}
                            </select>
                        </div>

                        {/* Enhanced Model Management Controls */}
                        {selectedModel && (
                            <div className="bg-gray-700/50 p-3 rounded-lg space-y-2">
                                <h4 className="text-sm font-medium text-green-400">Model Actions</h4>
                                <div className="grid grid-cols-2 gap-2">
                                    <button
                                        onClick={() => onLoadModel && onLoadModel(selectedModel)}
                                        className="text-xs bg-green-600 hover:bg-green-500 text-white py-1 px-2 rounded transition-colors"
                                        disabled={isLoading}
                                    >
                                        Load Model
                                    </button>
                                    <button
                                        onClick={() => onGetInfo && onGetInfo(selectedModel)}
                                        className="text-xs bg-blue-600 hover:bg-blue-500 text-white py-1 px-2 rounded transition-colors"
                                        disabled={isLoading}
                                    >
                                        Get Info
                                    </button>
                                    <button
                                        onClick={() => onPredict && onPredict(selectedModel)}
                                        className="text-xs bg-purple-600 hover:bg-purple-500 text-white py-1 px-2 rounded transition-colors"
                                        disabled={isLoading}
                                    >
                                        Predict
                                    </button>
                                    <button
                                        onClick={() => onDeleteModel && onDeleteModel(selectedModel)}
                                        className="text-xs bg-red-600 hover:bg-red-500 text-white py-1 px-2 rounded transition-colors"
                                        disabled={isLoading}
                                    >
                                        Delete
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Training Dashboard Button */}
                        <div className="border-t border-gray-600/50 pt-4">
                            <button
                                onClick={onShowTrainingDashboard}
                                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-medium py-2 px-4 rounded-lg transition-colors mb-2"
                            >
                                📊 Training Dashboard
                            </button>
                            <button
                                onClick={onShowMetacognitive}
                                className="w-full bg-purple-600 hover:bg-purple-500 text-white font-medium py-2 px-4 rounded-lg transition-colors mb-2"
                            >
                                🧠 Metacognitive AI
                            </button>
                            <button
                                onClick={onShowCrossModelAnalytics}
                                className="w-full bg-indigo-600 hover:bg-indigo-500 text-white font-medium py-2 px-4 rounded-lg transition-colors mb-4"
                            >
                                📈 Cross-Model Analytics
                            </button>
                        </div>

                        <TrainingPanel
                            onTrainingComplete={onTrainingComplete}
                            reportResult={reportResult}
                            availableModels={models}
                        />
                    </div>
                )}
            </div>

            {/* Stress Testing Section */}
            <div>
                <h2 className="text-lg font-semibold text-yellow-300 mb-2 border-b border-yellow-300/20 pb-1">Developer Tools</h2>
                <div className="bg-gray-800/50 p-4 rounded-lg space-y-3">
                     <DevButton onClick={onRunComprehensiveTest}>Run Comprehensive Stress Test</DevButton>
                     <DevButton onClick={onShowReport}>View Performance Report</DevButton>
                    <p className="text-xs text-center text-gray-500 pt-2 border-t border-gray-700">Client-Side Simulations</p>
                    <div className="grid grid-cols-2 gap-2">
                        <DevButton onClick={() => onStressTest('AI_FAILURE')}>AI Failure</DevButton>
                        <DevButton onClick={() => onStressTest('CSV_PARSE_ERROR')}>Bad CSV</DevButton>
                    </div>
                </div>
            </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
