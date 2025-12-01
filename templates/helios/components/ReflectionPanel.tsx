import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { ModelJournal } from '../types';
import Spinner from './Spinner';
import Alert from './Alert';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ReflectionPanelProps {
    journal: ModelJournal | null;
    isLoading: boolean;
    error: string | null;
}

const ReflectionPanel: React.FC<ReflectionPanelProps> = ({ journal, isLoading, error }) => {
    if (isLoading) {
        return (
            <div className="flex-1 p-8 overflow-y-auto flex items-center justify-center">
                <Spinner />
            </div>
        );
    }
    if (error) {
        return (
             <div className="flex-1 p-8 overflow-y-auto flex items-center justify-center">
                <Alert message={error} />
            </div>
        );
    }
    if (!journal) {
        return (
            <div className="flex-1 p-8 overflow-y-auto flex items-center justify-center">
                 <div className="text-center p-8 bg-gray-800/50 rounded-lg border border-gray-700">
                    <h2 className="text-3xl font-bold text-green-300 mb-2">Agent Reflection</h2>
                    <p className="text-gray-400 max-w-2xl mx-auto">
                        Select a trained agent from the sidebar to review its training journal and performance metrics.
                    </p>
                </div>
            </div>
        );
    }

    const chartData = {
        labels: journal.loss_history.map((_, index) => `Epoch ${index + 1}`),
        datasets: [
            {
                label: 'Training Loss',
                data: journal.loss_history,
                borderColor: 'rgb(52, 211, 153)',
                backgroundColor: 'rgba(52, 211, 153, 0.5)',
                tension: 0.1,
            },
        ],
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top' as const,
                labels: {
                  color: '#d1d5db',
                }
            },
            title: {
                display: true,
                text: `Training Loss History for ${journal.model_name}`,
                color: '#d1d5db',
                font: {
                  size: 18,
                }
            },
        },
        scales: {
            y: {
                ticks: { color: '#9ca3af' },
                grid: { color: '#4b5563' }
            },
            x: {
                ticks: { color: '#9ca3af' },
                grid: { color: '#4b5563' }
            }
        }
    };

    return (
        <main className="flex-1 p-8 overflow-y-auto">
            <div className="max-w-7xl mx-auto space-y-8">
                <div>
                    <h2 className="text-3xl font-bold text-white mb-2">Metacognition Report: <span className="text-green-400">{journal.model_name}</span></h2>
                    <p className="text-gray-400">Reviewing the agent's learning process and performance.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700 text-center">
                        <h4 className="text-sm font-medium text-gray-400 uppercase">Epochs</h4>
                        <p className="text-2xl font-semibold text-white">{journal.epochs}</p>
                    </div>
                    <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700 text-center">
                        <h4 className="text-sm font-medium text-gray-400 uppercase">Learning Rate</h4>
                        <p className="text-2xl font-semibold text-white">{journal.learning_rate}</p>
                    </div>
                     <div className="bg-gray-800/50 p-4 rounded-lg border border-gray-700 text-center">
                        <h4 className="text-sm font-medium text-gray-400 uppercase">Training Time</h4>
                        <p className="text-2xl font-semibold text-white">{journal.training_duration_seconds.toFixed(2)}s</p>
                    </div>
                </div>

                <div className="bg-gray-800/50 p-6 rounded-lg border border-gray-700">
                    <Line options={chartOptions} data={chartData} />
                </div>
            </div>
        </main>
    );
};

export default ReflectionPanel;
