import React, { useState, useCallback, useEffect } from 'react';
import { runSystemStressTest, StressTestSuite } from '../tests/stressTest';

interface StressTestComponentProps {
    onTestComplete?: (results: StressTestSuite) => void;
}

const StressTestComponent: React.FC<StressTestComponentProps> = ({ onTestComplete }) => {
    const [isRunning, setIsRunning] = useState(false);
    const [results, setResults] = useState<StressTestSuite | null>(null);
    const [currentTest, setCurrentTest] = useState<string>('');
    const [progress, setProgress] = useState(0);
    const [logs, setLogs] = useState<string[]>([]);

    // Capture console.log for display
    useEffect(() => {
        const originalLog = console.log;
        console.log = (...args: any[]) => {
            const message = args.map(arg => typeof arg === 'string' ? arg : JSON.stringify(arg)).join(' ');
            setLogs(prev => [...prev.slice(-100), message]); // Keep last 100 logs
            originalLog(...args);
        };

        return () => {
            console.log = originalLog;
        };
    }, []);

    const runStressTest = useCallback(async () => {
        setIsRunning(true);
        setResults(null);
        setLogs([]);
        setProgress(0);
        setCurrentTest('Initializing...');

        try {
            // Simulate progress updates
            const progressInterval = setInterval(() => {
                setProgress(prev => Math.min(prev + Math.random() * 15, 95));
            }, 1000);

            const testResults = await runSystemStressTest();

            clearInterval(progressInterval);
            setProgress(100);
            setCurrentTest('Complete');
            setResults(testResults);

            if (onTestComplete) {
                onTestComplete(testResults);
            }
        } catch (error) {
            setCurrentTest(`Failed: ${error instanceof Error ? error.message : String(error)}`);
            console.error('Stress test failed:', error);
        } finally {
            setIsRunning(false);
        }
    }, [onTestComplete]);

    const getStatusIcon = (status: 'PASS' | 'FAIL' | 'WARNING') => {
        switch (status) {
            case 'PASS': return '✅';
            case 'FAIL': return '❌';
            case 'WARNING': return '⚠️';
            default: return '❓';
        }
    };

    const getStatusColor = (status: 'PASS' | 'FAIL' | 'WARNING') => {
        switch (status) {
            case 'PASS': return 'text-green-600';
            case 'FAIL': return 'text-red-600';
            case 'WARNING': return 'text-yellow-600';
            default: return 'text-gray-600';
        }
    };

    const getOverallStatus = () => {
        if (!results) return null;

        const passRate = (results.summary.passed / results.summary.totalTests) * 100;

        if (passRate === 100) return { text: 'EXCELLENT', color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200' };
        if (passRate >= 80) return { text: 'GOOD', color: 'text-green-600', bg: 'bg-green-50', border: 'border-green-200' };
        if (passRate >= 60) return { text: 'ACCEPTABLE', color: 'text-yellow-600', bg: 'bg-yellow-50', border: 'border-yellow-200' };
        return { text: 'CRITICAL', color: 'text-red-600', bg: 'bg-red-50', border: 'border-red-200' };
    };

    return (
        <div className="w-full max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
            {/* Header */}
            <div className="mb-6">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">🚀 System Stress Test</h2>
                <p className="text-gray-600">
                    Comprehensive testing of configuration system, API endpoints, and Docker containers
                </p>
            </div>

            {/* Control Panel */}
            <div className="mb-6 p-4 bg-gray-50 rounded-lg">
                <button
                    onClick={runStressTest}
                    disabled={isRunning}
                    className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                        isRunning
                            ? 'bg-gray-400 text-gray-700 cursor-not-allowed'
                            : 'bg-blue-600 text-white hover:bg-blue-700'
                    }`}
                >
                    {isRunning ? '🔄 Running Tests...' : '▶️ Start Stress Test'}
                </button>

                {isRunning && (
                    <div className="mt-4">
                        <div className="flex justify-between text-sm text-gray-600 mb-1">
                            <span>Progress</span>
                            <span>{Math.round(progress)}%</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                            <div
                                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <p className="text-sm text-gray-600 mt-2">Current: {currentTest}</p>
                    </div>
                )}
            </div>

            {/* Results Summary */}
            {results && (
                <div className="mb-6">
                    <div className={`p-4 rounded-lg border-2 ${getOverallStatus()?.bg} ${getOverallStatus()?.border}`}>
                        <div className="flex items-center justify-between mb-3">
                            <h3 className={`text-lg font-bold ${getOverallStatus()?.color}`}>
                                🎯 Overall Status: {getOverallStatus()?.text}
                            </h3>
                            <div className="text-sm text-gray-600">
                                Duration: {(results.summary.totalDuration / 1000).toFixed(1)}s
                            </div>
                        </div>

                        <div className="grid grid-cols-4 gap-4 text-center">
                            <div className="bg-white p-3 rounded">
                                <div className="text-xl font-bold text-gray-900">{results.summary.totalTests}</div>
                                <div className="text-sm text-gray-600">Total Tests</div>
                            </div>
                            <div className="bg-white p-3 rounded">
                                <div className="text-xl font-bold text-green-600">{results.summary.passed}</div>
                                <div className="text-sm text-gray-600">Passed</div>
                            </div>
                            <div className="bg-white p-3 rounded">
                                <div className="text-xl font-bold text-yellow-600">{results.summary.warnings}</div>
                                <div className="text-sm text-gray-600">Warnings</div>
                            </div>
                            <div className="bg-white p-3 rounded">
                                <div className="text-xl font-bold text-red-600">{results.summary.failed}</div>
                                <div className="text-sm text-gray-600">Failed</div>
                            </div>
                        </div>

                        <div className="mt-3 text-center">
                            <div className="text-sm text-gray-600">
                                Success Rate: {((results.summary.passed / results.summary.totalTests) * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Detailed Results */}
            {results && (
                <div className="mb-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-3">📋 Detailed Results</h3>
                    <div className="space-y-3">
                        {results.results.map((result, index) => (
                            <div key={index} className="border rounded-lg p-4 bg-gray-50">
                                <div className="flex items-start justify-between">
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="text-lg">{getStatusIcon(result.status)}</span>
                                            <span className="font-medium text-gray-900">{result.testName}</span>
                                            <span className={`text-sm font-bold ${getStatusColor(result.status)}`}>
                                                {result.status}
                                            </span>
                                        </div>
                                        <p className="text-sm text-gray-600 mb-2">{result.details}</p>

                                        {result.metrics && (
                                            <div className="flex gap-4 text-xs text-gray-500">
                                                {Object.entries(result.metrics).map(([key, value]) => (
                                                    <span key={key}>
                                                        {key}: {typeof value === 'number' ? value.toFixed(2) : value}
                                                    </span>
                                                ))}
                                            </div>
                                        )}
                                    </div>
                                    <div className="text-sm text-gray-500 ml-4">
                                        {result.duration}ms
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Recommendations */}
            {results && (
                <div className="mb-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-3">💡 Recommendations</h3>
                    <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <ul className="space-y-2 text-sm">
                            {results.summary.failed > 0 && (
                                <>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-500">•</span>
                                        Review failed tests and address underlying issues
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-500">•</span>
                                        Check Docker container health and resource allocation
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-500">•</span>
                                        Verify network connectivity and API endpoints
                                    </li>
                                </>
                            )}
                            {results.summary.warnings > 0 && (
                                <>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-500">•</span>
                                        Monitor warning conditions in production
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-yellow-500">•</span>
                                        Consider optimizing resource usage and performance
                                    </li>
                                </>
                            )}
                            {((results.summary.passed / results.summary.totalTests) * 100) >= 80 && (
                                <>
                                    <li className="flex items-start gap-2">
                                        <span className="text-green-500">•</span>
                                        System ready for deployment
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-green-500">•</span>
                                        Consider implementing monitoring and alerting
                                    </li>
                                    <li className="flex items-start gap-2">
                                        <span className="text-green-500">•</span>
                                        Regular stress testing recommended
                                    </li>
                                </>
                            )}
                        </ul>
                    </div>
                </div>
            )}

            {/* Live Logs */}
            {logs.length > 0 && (
                <div className="mb-6">
                    <h3 className="text-lg font-bold text-gray-900 mb-3">📝 Test Logs</h3>
                    <div className="bg-black text-green-400 p-4 rounded-lg font-mono text-xs max-h-64 overflow-y-auto">
                        {logs.map((log, index) => (
                            <div key={index} className="mb-1">
                                {log}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Docker Commands */}
            <div className="bg-gray-900 text-white p-4 rounded-lg">
                <h3 className="text-lg font-bold mb-3">🐳 Docker Commands</h3>
                <div className="space-y-2 text-sm font-mono">
                    <div>Start production: <span className="text-blue-400">docker-compose up -d</span></div>
                    <div>Start development: <span className="text-blue-400">docker-compose --profile dev up -d</span></div>
                    <div>Check status: <span className="text-blue-400">docker-compose ps</span></div>
                    <div>View logs: <span className="text-blue-400">docker-compose logs -f</span></div>
                    <div>Stop services: <span className="text-blue-400">docker-compose down</span></div>
                </div>
            </div>
        </div>
    );
};

export default StressTestComponent;
