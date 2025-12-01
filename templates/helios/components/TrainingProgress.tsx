import React, { useState, useEffect, useCallback, useRef } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { modelService } from '../services/modelService';
import { TrainingStatus, TrainingProgressEntry } from '../types';
import Alert from './Alert';
import './TrainingProgress.css';

interface TrainingProgressProps {
    jobId: string;
    onComplete: () => void;
    onError: (error: string) => void;
    onStop?: () => void;
}

interface LossData {
    epoch: number;
    loss: number;
}

// Styles for recharts components
const tooltipStyle = {
    backgroundColor: '#1F2937',
    border: '1px solid #374151',
    borderRadius: '6px',
    color: '#F3F4F6'
};

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
    jobId,
    onComplete,
    onError,
    onStop
}) => {
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [lossData, setLossData] = useState<LossData[]>([]);
    const [isPolling, setIsPolling] = useState(true);
    const [error, setError] = useState<string>('');
    const [isStopping, setIsStopping] = useState(false);
    const progressBarRef = useRef<HTMLDivElement>(null);

    const pollTrainingStatus = useCallback(async () => {
        try {
            const statusData = await modelService.getTrainingStatus(jobId);
            setStatus(statusData);

            // Update loss data for chart
            if (statusData.training_progress && statusData.training_progress.length > 0) {
                const newLossData = statusData.training_progress.map((entry: TrainingProgressEntry, index: number) => ({
                    epoch: index + 1,
                    loss: entry.loss
                }));
                setLossData(newLossData);
            }

            // Handle completion
            if (statusData.status === 'completed') {
                setIsPolling(false);
                onComplete();
            } else if (statusData.status === 'failed') {
                setIsPolling(false);
                onError(statusData.error || 'Training failed');
            } else if (statusData.status === 'stopped') {
                setIsPolling(false);
                if (onStop) onStop();
            }
        } catch (err: any) {
            setError(err.message);
            setIsPolling(false);
            onError(err.message);
        }
    }, [jobId, onComplete, onError, onStop]);

    // Polling effect
    useEffect(() => {
        if (!isPolling) return;

        const interval = setInterval(pollTrainingStatus, 2000); // Poll every 2 seconds
        pollTrainingStatus(); // Initial call

        return () => clearInterval(interval);
    }, [isPolling, pollTrainingStatus]);

    // Update progress bar width via CSS custom property
    useEffect(() => {
        if (progressBarRef.current && status) {
            const progressPercentage = status.total_epochs
                ? Math.round((status.current_epoch / status.total_epochs) * 100)
                : 0;
            progressBarRef.current.style.setProperty('--progress-width', `${progressPercentage}%`);
        }
    }, [status]);

    const handleStop = async () => {
        setIsStopping(true);
        try {
            await modelService.stopTraining(jobId);
            setIsPolling(false);
            if (onStop) onStop();
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsStopping(false);
        }
    };

    const formatDuration = (seconds: number): string => {
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    const calculateETA = (): string => {
        if (!status || !status.current_epoch || !status.total_epochs || !status.start_time) {
            return 'Calculating...';
        }

        const elapsedTime = (Date.now() - new Date(status.start_time).getTime()) / 1000;
        const avgTimePerEpoch = elapsedTime / status.current_epoch;
        const remainingEpochs = status.total_epochs - status.current_epoch;
        const remainingTime = avgTimePerEpoch * remainingEpochs;

        return formatDuration(remainingTime);
    };

    if (!status) {
        return (
            <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                    <span className="text-gray-300">Loading training status...</span>
                </div>
            </div>
        );
    }

    const progressPercentage = status.total_epochs
        ? Math.round((status.current_epoch / status.total_epochs) * 100)
        : 0;

    return (
        <div className="bg-gray-800/50 p-4 rounded-lg space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Training Progress</h3>
                <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                        status.status === 'running' ? 'bg-green-100 text-green-800' :
                        status.status === 'completed' ? 'bg-blue-100 text-blue-800' :
                        status.status === 'failed' ? 'bg-red-100 text-red-800' :
                        'bg-yellow-100 text-yellow-800'
                    }`}>
                        {status.status.toUpperCase()}
                    </span>
                    {status.status === 'running' && (
                        <button
                            onClick={handleStop}
                            disabled={isStopping}
                            className="px-3 py-1 bg-red-600 hover:bg-red-500 disabled:bg-gray-600 text-white text-xs rounded transition-colors"
                        >
                            {isStopping ? 'Stopping...' : 'Stop'}
                        </button>
                    )}
                </div>
            </div>

            {/* Progress Bar */}
            <div className="space-y-2">
                <div className="flex justify-between text-sm text-gray-300">
                    <span>Epoch {status.current_epoch} of {status.total_epochs}</span>
                    <span>{progressPercentage}%</span>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                    <div
                        ref={progressBarRef}
                        className="bg-blue-600 h-2 rounded-full transition-all duration-300 training-progress-bar"
                        data-progress={progressPercentage}
                    ></div>
                </div>
            </div>

            {/* Statistics */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400">Current Loss</div>
                    <div className="text-white font-medium">
                        {status.current_loss ? status.current_loss.toFixed(6) : 'N/A'}
                    </div>
                </div>
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400">Best Loss</div>
                    <div className="text-white font-medium">
                        {status.best_loss ? status.best_loss.toFixed(6) : 'N/A'}
                    </div>
                </div>
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400">Elapsed Time</div>
                    <div className="text-white font-medium">
                        {status.start_time ? formatDuration((Date.now() - new Date(status.start_time).getTime()) / 1000) : 'N/A'}
                    </div>
                </div>
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400">ETA</div>
                    <div className="text-white font-medium">
                        {status.status === 'running' ? calculateETA() : 'N/A'}
                    </div>
                </div>
            </div>

            {/* Loss Chart */}
            {lossData.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-300">Loss Curve</h4>
                    <div className="bg-gray-700/30 p-4 rounded">
                        <ResponsiveContainer width="100%" height={200}>
                            <LineChart data={lossData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis
                                    dataKey="epoch"
                                    stroke="#9CA3AF"
                                    fontSize={12}
                                />
                                <YAxis
                                    stroke="#9CA3AF"
                                    fontSize={12}
                                    tickFormatter={(value: number) => value.toFixed(4)}
                                />
                                <Tooltip
                                    contentStyle={tooltipStyle}
                                    formatter={(value: number) => [value.toFixed(6), 'Loss']}
                                />
                                <Line
                                    type="monotone"
                                    dataKey="loss"
                                    stroke="#3B82F6"
                                    strokeWidth={2}
                                    dot={false}
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {/* Recent Logs */}
            {status.training_progress && status.training_progress.length > 0 && (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-300">Recent Progress</h4>
                    <div className="bg-gray-700/30 p-3 rounded text-xs font-mono">
                        {status.training_progress.slice(-5).map((entry: TrainingProgressEntry, index: number) => (
                            <div key={index} className="text-gray-300">
                                Epoch {entry.epoch}: Loss = {entry.loss.toFixed(6)}
                                {entry.timestamp && (
                                    <span className="text-gray-500 ml-2">
                                        [{new Date(entry.timestamp).toLocaleTimeString()}]
                                    </span>
                                )}
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {error && <Alert message={error} type="error" />}
        </div>
    );
};
