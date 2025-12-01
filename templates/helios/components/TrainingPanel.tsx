import React, { useState } from 'react';
import { modelService } from '../services/modelService';
import Alert from './Alert';
import { TrainingProgress } from './TrainingProgress';
import { ModelNameInput } from './ModelNameInput';
import { TestReportEntry } from '../types';

interface TrainingPanelProps {
    onTrainingComplete: () => void;
    reportResult: (result: Omit<TestReportEntry, 'id' | 'timestamp'>) => void;
    availableModels?: string[];
}

export const TrainingPanel: React.FC<TrainingPanelProps> = ({ onTrainingComplete, reportResult, availableModels = [] }) => {
    const [modelName, setModelName] = useState<string>('agent_v1');
    const [epochs, setEpochs] = useState<number>(50);
    const [learningRate, setLearningRate] = useState<number>(0.001);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [message, setMessage] = useState<string>('');
    const [error, setError] = useState<string>('');
    const [currentJobId, setCurrentJobId] = useState<string | null>(null);
    const [isTraining, setIsTraining] = useState<boolean>(false);

    const handleTrain = async (e: React.FormEvent) => {
        e.preventDefault();
        setIsLoading(true);
        setMessage('');
        setError('');

        const startTime = performance.now();
        const testName = `Agent Training: ${modelName}`;

        try {
            const result = await modelService.startTraining({ model_name: modelName, epochs, learning_rate: learningRate });
            const durationMs = performance.now() - startTime;

            setCurrentJobId(result.job_id);
            setIsTraining(true);
            setMessage(result.message);

            reportResult({ testName, status: 'PASS', durationMs, details: result.message });
        } catch (err: any) {
            const durationMs = performance.now() - startTime;
            setError(err.message);
            reportResult({ testName, status: 'FAIL', durationMs, details: err.message });
        } finally {
            setIsLoading(false);
        }
    };

    const handleTrainingComplete = () => {
        setIsTraining(false);
        setCurrentJobId(null);
        onTrainingComplete(); // Notify parent to refresh model list
        setMessage('Training completed successfully!');
    };

    const handleTrainingError = (errorMessage: string) => {
        setIsTraining(false);
        setCurrentJobId(null);
        setError(errorMessage);
    };

    const handleTrainingStop = () => {
        setIsTraining(false);
        setCurrentJobId(null);
        setMessage('Training stopped by user.');
    };

    return (
        <div className="space-y-4">
            {!isTraining ? (
                <form onSubmit={handleTrain} className="bg-gray-800/50 p-4 rounded-lg space-y-4">
                     <div className="space-y-2">
                        <ModelNameInput
                            value={modelName}
                            onChange={setModelName}
                            existingModels={availableModels}
                            placeholder="Enter agent name..."
                        />
                        <div>
                            <label htmlFor="epochs" className="block text-sm font-medium text-gray-300">Epochs</label>
                            <input
                                id="epochs"
                                type="number"
                                value={epochs}
                                onChange={(e) => setEpochs(Number(e.target.value))}
                                className="mt-1 block w-full bg-gray-700 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                                required
                            />
                        </div>
                        <div>
                            <label htmlFor="learningRate" className="block text-sm font-medium text-gray-300">Learning Rate</label>
                            <input
                                id="learningRate"
                                type="number"
                                step="0.0001"
                                value={learningRate}
                                onChange={(e) => setLearningRate(Number(e.target.value))}
                                className="mt-1 block w-full bg-gray-700 border border-gray-600 rounded-md shadow-sm py-2 px-3 text-white focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                                required
                            />
                        </div>
                    </div>
                    <button
                        type="submit"
                        disabled={isLoading || !modelName}
                        className="w-full flex items-center justify-center bg-green-600 hover:bg-green-500 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold py-2 px-4 rounded-lg transition-colors"
                    >
                        {isLoading ? 'Starting Training...' : '🚀 Begin Training'}
                    </button>
                    {message && <Alert message={message} type="info" />}
                    {error && <Alert message={error} type="error" />}
                </form>
            ) : (
                currentJobId && (
                    <TrainingProgress
                        jobId={currentJobId}
                        onComplete={handleTrainingComplete}
                        onError={handleTrainingError}
                        onStop={handleTrainingStop}
                    />
                )
            )}
        </div>
    );
};
