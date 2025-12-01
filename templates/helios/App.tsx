import React, { useState, useCallback, useEffect, useRef } from 'react';
import ReactDOM from 'react-dom/client';
import Sidebar from './components/Sidebar';
import ResultsPanel from './components/ResultsPanel';
import ReflectionPanel from './components/ReflectionPanel';
import StressTestReportPanel from './components/StressTestReportPanel';
import DisconnectedPanel from './components/DisconnectedPanel';
import MetacognitiveDashboard from './components/MetacognitiveDashboard';
import CrossModelAnalytics from './components/CrossModelAnalytics';
import { runFullAnalysis, AnalysisPayload } from './services/api';
import { modelService } from './services/modelService';
import { CONNECTION_ERROR_MESSAGE } from './services/config';
import {
  BacktestResults,
  StressTestType,
  ModelJournal,
  ViewType,
  TestReportEntry,
  BackendStatus,
  ModelInfo,
  ModelManagementState
} from './types';
import { REQUIRED_COLUMNS } from './constants';
import { TrainingPanel } from './components/TrainingPanel';
import { TrainingDashboard } from './components/TrainingDashboard';


const App: React.FC = () => {
  // State for different panels and views
  const [currentView, setCurrentView] = useState<ViewType>('baseline');

  // State for Baseline Analysis (Level 0)
  const [file, setFile] = useState<File | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<BacktestResults | null>(null);
  const [aiAnalysis, setAiAnalysis] = useState<string | null>(null);
  const [totalDraws, setTotalDraws] = useState<number>(0);

  // State for Model Management (Level 2) - Enhanced
  const [modelManagement, setModelManagement] = useState<ModelManagementState>({
    availableModels: [],
    selectedModel: null,
    loadingModel: false,
    modelDetails: null,
    lastPrediction: null,
    trainingInProgress: false,
    currentTrainingJob: null
  });

  // Legacy state for backward compatibility
  const [models, setModels] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [journal, setJournal] = useState<ModelJournal | null>(null);
  const [isJournalLoading, setIsJournalLoading] = useState<boolean>(false);
  const [journalError, setJournalError] = useState<string | null>(null);

  // State for Backend Connection and Reporting
  const [backendStatus, setBackendStatus] = useState<BackendStatus>('pending');
  const [stressTestLog, setStressTestLog] = useState<TestReportEntry[]>([]);
  const logIdCounter = useRef(0);

  // --- Effects ---
  useEffect(() => {
    fetchModels();
  }, []);

  // --- Reporting ---
  const handleReportResult = (result: Omit<TestReportEntry, 'id' | 'timestamp'>) => {
    const newEntry: TestReportEntry = {
      ...result,
      id: logIdCounter.current++,
      timestamp: new Date().toLocaleTimeString(),
    };
    setStressTestLog(prevLog => [...prevLog, newEntry]);
  };

  // --- Handlers ---
  const fetchModels = useCallback(async () => {
    setBackendStatus('pending');
    const testName = "Connect to Backend";
    const startTime = performance.now();
    try {
        const availableModels = await modelService.getModels();

        // Handle both old string[] format and new ModelInfo[] format
        let modelNames: string[];
        let modelInfos: ModelInfo[];

        if (Array.isArray(availableModels) && availableModels.length > 0) {
          if (typeof availableModels[0] === 'string') {
            // Old format - array of strings
            modelNames = availableModels as string[];
            modelInfos = modelNames.map(name => ({
              name,
              architecture: 'Unknown',
              version: '1.0.0',
              created_at: new Date().toISOString(),
              training_completed: false,
              total_epochs: 0,
              best_loss: 0
            }));
          } else {
            // New format - array of ModelInfo objects
            modelInfos = availableModels as unknown as ModelInfo[];
            modelNames = modelInfos.map(model => model.name);
          }
        } else {
          modelNames = [];
          modelInfos = [];
        }

        setModels(modelNames);
        setModelManagement(prev => ({
          ...prev,
          availableModels: modelInfos
        }));
        setBackendStatus('connected');

        const durationMs = performance.now() - startTime;
        handleReportResult({
          testName,
          status: 'PASS',
          durationMs,
          details: `Successfully fetched ${modelNames.length} models.`
        });
    } catch (e: any) {
        console.error("Failed to fetch models:", e.message);
        const durationMs = performance.now() - startTime;
        handleReportResult({testName, status: 'FAIL', durationMs, details: e.message});
        if(e.message === CONNECTION_ERROR_MESSAGE) {
            setBackendStatus('disconnected');
        }
    }
  }, []);

  const handleRetryConnection = () => {
    fetchModels();
  };

  const handleFileChange = (selectedFile: File) => {
    setFile(selectedFile);
    setCurrentView('baseline');
    setError(null);
    setResults(null);
    setAiAnalysis(null);
    setTotalDraws(0);
  };

  const handleLaunchAnalysis = useCallback(async (fileToAnalyze: File, stressTest?: StressTestType) => {
    setCurrentView('baseline');
    setIsLoading(true);
    setError(null);
    setResults(null);
    setAiAnalysis(null);
    setTotalDraws(0);

    const testName = stressTest ? `Baseline Analysis (${stressTest})` : 'Baseline Analysis';
    const startTime = performance.now();

    try {
      const payload: AnalysisPayload = await runFullAnalysis(fileToAnalyze, stressTest);
      setResults(payload.results);
      setAiAnalysis(payload.aiAnalysis);
      setTotalDraws(payload.totalDraws);
      const durationMs = performance.now() - startTime;
      handleReportResult({ testName, status: 'PASS', durationMs, details: `Analyzed ${payload.totalDraws} draws.` });
    } catch (e: any) {
      setError(e.message || "An unexpected error occurred during analysis.");
      const durationMs = performance.now() - startTime;
      handleReportResult({ testName, status: 'FAIL', durationMs, details: e.message });
    } finally {
      setIsLoading(false);
    }
  }, []);

  const handleSelectModel = useCallback(async (modelName: string) => {
    if (!modelName) {
      setSelectedModel(null);
      setJournal(null);
      setCurrentView('reflection');
      return;
    }
    setSelectedModel(modelName);
    setCurrentView('reflection');
    setIsJournalLoading(true);
    setJournalError(null);
    setJournal(null);

    const testName = `Fetch Journal: ${modelName}`;
    const startTime = performance.now();

    try {
        const journalData = await modelService.getModelJournal(modelName);
        setJournal(journalData);
        const durationMs = performance.now() - startTime;
        handleReportResult({ testName, status: 'PASS', durationMs, details: `Loaded journal with ${journalData.loss_history.length} epochs.` });
    } catch (e: any) {
        setJournalError(e.message);
        const durationMs = performance.now() - startTime;
        handleReportResult({ testName, status: 'FAIL', durationMs, details: e.message });
    } finally {
        setIsJournalLoading(false);
    }
  }, []);

  const handleTrainingComplete = () => {
      fetchModels(); // Refresh the list of models after training
  };

  const handleShowBaseline = () => setCurrentView('baseline');
  const handleShowReport = () => setCurrentView('stress_report');
  const handleShowTrainingDashboard = () => setCurrentView('training_dashboard');
  const handleShowMetacognitive = () => setCurrentView('metacognitive');
  const handleShowCrossModelAnalytics = () => setCurrentView('cross_model_analytics');

  const handleClientStressTest = (testType: StressTestType) => {
    let mockFile: File;
    const validCsvContent = `${REQUIRED_COLUMNS.join(',')}\n1/1/2024,1,2,3,4,5,6\n1/2/2024,7,8,9,10,11,12`;
    const badCsvContent = `header1,header2\nvalue1,value2`;

    switch (testType) {
        case 'CSV_PARSE_ERROR':
            mockFile = new File([badCsvContent], "mock_bad.csv", { type: "text/csv" });
            break;
        default:
            mockFile = new File([validCsvContent], "mock_valid.csv", { type: "text/csv" });
            break;
    }
    setFile(mockFile);
    handleLaunchAnalysis(mockFile, testType);
  };

  const handleRunComprehensiveTest = async () => {
    setCurrentView('stress_report');
    setStressTestLog([]);
    logIdCounter.current = 0;

    // Test 1: Connect to backend (already run on load, just log it as the start)
    await fetchModels();

    // Test 2: Successful Baseline Analysis
    const validCsvContent = `${REQUIRED_COLUMNS.join(',')}\n1/1/2024,1,2,3,4,5,6`;
    const mockFileValid = new File([validCsvContent], "mock_valid.csv", { type: "text/csv" });
    await handleLaunchAnalysis(mockFileValid);

    // Test 3: Failed CSV Parse
    const badCsvContent = `header1,header2\nvalue1,value2`;
    const mockFileBad = new File([badCsvContent], "mock_bad.csv", { type: "text/csv" });
    await handleLaunchAnalysis(mockFileBad);

    // Test 4: AI Service Failure
    await handleLaunchAnalysis(mockFileValid, 'AI_FAILURE');

    // Test 5 & 6 only if backend is connected
    if (backendStatus === 'connected') {
        // Test 5: Successful Agent Training
        await new Promise<void>(resolve => {
            const panel = document.createElement('div');
            const root = ReactDOM.createRoot(panel);

            const onTestTrainingComplete = () => {
                handleTrainingComplete();
                resolve();
            };

            const onTestReportResult = (result: Omit<TestReportEntry, 'id' | 'timestamp'>) => {
                handleReportResult(result);
                // Only resolve if the test fails, as onTrainingComplete will resolve on success.
                if (result.status === 'FAIL') {
                    resolve();
                }
            };

            root.render(
                <TrainingPanel
                    onTrainingComplete={onTestTrainingComplete}
                    reportResult={onTestReportResult}
                    availableModels={modelManagement.availableModels.map(model => model.name)}
                />
            );

            // Allow React to render the component before dispatching the event
            setTimeout(() => {
                const form = panel.querySelector('form');
                if (form) {
                    form.dispatchEvent(new Event('submit', { bubbles: true, cancelable: true }));
                } else {
                    // Fail the test if the form is not found
                    handleReportResult({
                        testName: 'Agent Training Simulation',
                        status: 'FAIL',
                        durationMs: 0,
                        details: 'Could not find form element to submit for training test.'
                    });
                    resolve();
                }
            }, 0);
        });

        // Test 6: Fetch Journal (Failure)
        await handleSelectModel('non_existent_model_' + Math.random());
    }
  };

  // New model management functions
  const handleLoadModel = useCallback(async (modelName: string) => {
    setModelManagement(prev => ({ ...prev, loadingModel: true }));

    const testName = `Load Model: ${modelName}`;
    const startTime = performance.now();

    try {
      const response = await modelService.loadModel(modelName);

      if (response.status === 'success') {
        setModelManagement(prev => ({
          ...prev,
          selectedModel: modelName,
          loadingModel: false,
          modelDetails: null // Will be fetched separately if needed
        }));

        const durationMs = performance.now() - startTime;
        handleReportResult({
          testName,
          status: 'PASS',
          durationMs,
          details: response.message
        });
      } else {
        throw new Error(response.message || 'Failed to load model');
      }
    } catch (e: any) {
      setModelManagement(prev => ({ ...prev, loadingModel: false }));
      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'FAIL',
        durationMs,
        details: e.message
      });
    }
  }, []);

  const handleGetModelInfo = useCallback(async (modelName: string) => {
    const testName = `Get Model Info: ${modelName}`;
    const startTime = performance.now();

    try {
      const modelDetails = await modelService.getModelInfo(modelName);

      setModelManagement(prev => ({
        ...prev,
        modelDetails
      }));

      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'PASS',
        durationMs,
        details: `Retrieved model info for ${modelName}`
      });

      return modelDetails;
    } catch (e: any) {
      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'FAIL',
        durationMs,
        details: e.message
      });
      throw e;
    }
  }, []);

  const handleGeneratePrediction = useCallback(async (modelName: string) => {
    const testName = `Generate Prediction: ${modelName}`;
    const startTime = performance.now();

    try {
      const prediction = await modelService.generatePrediction(modelName);

      setModelManagement(prev => ({
        ...prev,
        lastPrediction: prediction
      }));

      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'PASS',
        durationMs,
        details: `Generated prediction with confidence: ${prediction.confidence?.[0] || 'N/A'}`
      });

      return prediction;
    } catch (e: any) {
      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'FAIL',
        durationMs,
        details: e.message
      });
      throw e;
    }
  }, []);

  const handleDeleteModel = useCallback(async (modelName: string) => {
    const testName = `Delete Model: ${modelName}`;
    const startTime = performance.now();

    try {
      await modelService.deleteModel(modelName);

      // Remove from local state
      setModels(prev => prev.filter(name => name !== modelName));
      setModelManagement(prev => ({
        ...prev,
        availableModels: prev.availableModels.filter(model => model.name !== modelName),
        selectedModel: prev.selectedModel === modelName ? null : prev.selectedModel,
        modelDetails: prev.selectedModel === modelName ? null : prev.modelDetails
      }));

      if (selectedModel === modelName) {
        setSelectedModel(null);
        setJournal(null);
      }

      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'PASS',
        durationMs,
        details: `Successfully deleted model: ${modelName}`
      });
    } catch (e: any) {
      const durationMs = performance.now() - startTime;
      handleReportResult({
        testName,
        status: 'FAIL',
        durationMs,
        details: e.message
      });
      throw e;
    }
  }, [selectedModel]);

  const renderMainPanel = () => {
    switch(currentView) {
        case 'reflection':
            return <ReflectionPanel journal={journal} isLoading={isJournalLoading} error={journalError} />;
        case 'stress_report':
            return <StressTestReportPanel log={stressTestLog} />;
        case 'training_dashboard':
            return <TrainingDashboard onStartNewTraining={handleShowBaseline} />;
        case 'metacognitive':
            return <MetacognitiveDashboard modelName={selectedModel || 'default'} />;
        case 'cross_model_analytics':
            return <CrossModelAnalytics />;
        case 'baseline':
        default:
            return (
                <ResultsPanel
                    isLoading={isLoading}
                    error={error}
                    results={results}
                    aiAnalysis={aiAnalysis}
                    totalDraws={totalDraws}
                />
            );
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 overflow-hidden">
      <Sidebar
        onFileChange={handleFileChange}
        selectedFile={file}
        onLaunch={() => file && handleLaunchAnalysis(file)}
        isLoading={isLoading || isJournalLoading}
        onStressTest={handleClientStressTest}
        models={models}
        selectedModel={selectedModel}
        onSelectModel={handleSelectModel}
        onTrainingComplete={handleTrainingComplete}
        onShowBaseline={handleShowBaseline}
        onShowTrainingDashboard={handleShowTrainingDashboard}
        onShowMetacognitive={handleShowMetacognitive}
        onShowCrossModelAnalytics={handleShowCrossModelAnalytics}
        onLoadModel={handleLoadModel}
        onGetInfo={handleGetModelInfo}
        onPredict={handleGeneratePrediction}
        onDeleteModel={handleDeleteModel}
        backendStatus={backendStatus}
        onRetryConnection={handleRetryConnection}
        onRunComprehensiveTest={handleRunComprehensiveTest}
        onShowReport={handleShowReport}
        reportResult={handleReportResult}
      />
      {backendStatus === 'disconnected' ? (
        <DisconnectedPanel onRetry={handleRetryConnection} />
      ) : (
        renderMainPanel()
      )}
    </div>
  );
};

export default App;
