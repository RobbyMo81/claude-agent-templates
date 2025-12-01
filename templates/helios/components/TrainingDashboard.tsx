import React, { useState, useEffect } from 'react';
import { modelService } from '../services/modelService';
import { TrainingHistoryEntry } from '../types';
import Alert from './Alert';

interface TrainingDashboardProps {
    onStartNewTraining: () => void;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({ onStartNewTraining }) => {
    const [trainingHistory, setTrainingHistory] = useState<TrainingHistoryEntry[]>([]);
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string>('');

    useEffect(() => {
        loadTrainingHistory();
    }, []);

    const loadTrainingHistory = async () => {
        try {
            setIsLoading(true);
            setError('');
            const history = await modelService.getTrainingHistory();
            setTrainingHistory(history);
        } catch (err: any) {
            setError(err.message);
        } finally {
            setIsLoading(false);
        }
    };

    const formatDuration = (startTime: string, endTime: string | null): string => {
        if (!endTime) return 'N/A';
        const duration = (new Date(endTime).getTime() - new Date(startTime).getTime()) / 1000;
        const minutes = Math.floor(duration / 60);
        const seconds = Math.floor(duration % 60);
        return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    };

    const getStatusBadge = (status: string) => {
        const baseClasses = "px-2 py-1 rounded text-xs font-medium";
        switch (status) {
            case 'completed':
                return `${baseClasses} bg-green-100 text-green-800`;
            case 'failed':
                return `${baseClasses} bg-red-100 text-red-800`;
            case 'stopped':
                return `${baseClasses} bg-yellow-100 text-yellow-800`;
            default:
                return `${baseClasses} bg-gray-100 text-gray-800`;
        }
    };

    if (isLoading) {
        return (
            <div className="bg-gray-800/50 p-4 rounded-lg">
                <div className="flex items-center justify-center space-x-2">
                    <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
                    <span className="text-gray-300">Loading training history...</span>
                </div>
            </div>
        );
    }

    return (
        <div className="bg-gray-800/50 p-4 rounded-lg space-y-4">
            {/* Header */}
            <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-white">Training Dashboard</h3>
                <button
                    onClick={onStartNewTraining}
                    className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded-lg transition-colors"
                >
                    Start New Training
                </button>
            </div>

            {/* Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400 text-sm">Total Sessions</div>
                    <div className="text-2xl font-bold text-white">{trainingHistory.length}</div>
                </div>
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400 text-sm">Completed</div>
                    <div className="text-2xl font-bold text-green-400">
                        {trainingHistory.filter(h => h.status === 'completed').length}
                    </div>
                </div>
                <div className="bg-gray-700/50 p-3 rounded">
                    <div className="text-gray-400 text-sm">Failed</div>
                    <div className="text-2xl font-bold text-red-400">
                        {trainingHistory.filter(h => h.status === 'failed').length}
                    </div>
                </div>
            </div>

            {/* Training History Table */}
            {trainingHistory.length === 0 ? (
                <div className="text-center py-8">
                    <div className="text-gray-400 mb-4">No training sessions found</div>
                    <button
                        onClick={onStartNewTraining}
                        className="px-6 py-3 bg-green-600 hover:bg-green-500 text-white rounded-lg transition-colors"
                    >
                        🚀 Start Your First Training
                    </button>
                </div>
            ) : (
                <div className="space-y-2">
                    <h4 className="text-sm font-medium text-gray-300">Recent Training Sessions</h4>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm text-gray-300">
                            <thead className="text-xs text-gray-400 uppercase bg-gray-700/30">
                                <tr>
                                    <th className="px-4 py-2 text-left">Model Name</th>
                                    <th className="px-4 py-2 text-left">Status</th>
                                    <th className="px-4 py-2 text-left">Epochs</th>
                                    <th className="px-4 py-2 text-left">Final Loss</th>
                                    <th className="px-4 py-2 text-left">Duration</th>
                                    <th className="px-4 py-2 text-left">Started</th>
                                </tr>
                            </thead>
                            <tbody>
                                {trainingHistory.map((session, index) => (
                                    <tr key={session.job_id} className={index % 2 === 0 ? 'bg-gray-700/20' : ''}>
                                        <td className="px-4 py-3 font-medium">{session.model_name}</td>
                                        <td className="px-4 py-3">
                                            <span className={getStatusBadge(session.status)}>
                                                {session.status.toUpperCase()}
                                            </span>
                                        </td>
                                        <td className="px-4 py-3">{session.total_epochs}</td>
                                        <td className="px-4 py-3">
                                            {session.final_loss ? session.final_loss.toFixed(6) : 'N/A'}
                                        </td>
                                        <td className="px-4 py-3">
                                            {formatDuration(session.start_time, session.end_time)}
                                        </td>
                                        <td className="px-4 py-3">
                                            {new Date(session.start_time).toLocaleDateString()} {' '}
                                            {new Date(session.start_time).toLocaleTimeString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {error && <Alert message={error} type="error" />}
        </div>
    );
};
