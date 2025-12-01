
import React from 'react';

interface DisconnectedPanelProps {
  onRetry: () => void;
}

const DisconnectedPanel: React.FC<DisconnectedPanelProps> = ({ onRetry }) => {
  return (
    <main className="flex-1 p-8 flex items-center justify-center">
      <div className="text-center p-8 bg-gray-800/50 rounded-lg border border-red-700 max-w-2xl">
        <div className="mx-auto flex items-center justify-center h-12 w-12 rounded-full bg-red-900">
            <svg className="h-6 w-6 text-red-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth="2" stroke="currentColor" aria-hidden="true">
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
        </div>
        <h2 className="mt-4 text-3xl font-bold text-red-300">Action Required: Start Backend Server</h2>
        <p className="mt-2 text-lg text-gray-300">
          The Helios frontend is running, but it cannot find its backend "engine room".
        </p>
        <p className="mt-2 text-gray-400">
          To enable advanced features like Agent Training, you must start the Python server manually.
        </p>

        <div className="mt-6 text-left bg-gray-900/70 p-4 rounded-lg border border-gray-600">
            <h4 className="font-semibold text-lg text-yellow-300 mb-3">Resolution Steps</h4>
            <ol className="space-y-4 text-gray-300 list-none">
                <li className="flex items-start">
                    <span className="bg-gray-700 text-yellow-300 font-bold rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0">1</span>
                    <span>Open a new terminal window or tab.</span>
                </li>
                <li className="flex items-start">
                    <span className="bg-gray-700 text-yellow-300 font-bold rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0">2</span>
                    <span>Navigate to the project's root directory (the one containing `server.py`).</span>
                </li>
                <li className="flex items-start">
                    <span className="bg-gray-700 text-yellow-300 font-bold rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0">3</span>
                    <div>
                        <span>Run the following command to start the server:</span>
                        <pre className="bg-black/50 p-3 rounded-md text-sm text-yellow-200 overflow-x-auto mt-2"><code>python server.py</code></pre>
                    </div>
                </li>
                 <li className="flex items-start">
                    <span className="bg-gray-700 text-yellow-300 font-bold rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0">4</span>
                    <span>Once the server is running, return to this window and click the button below.</span>
                </li>
            </ol>
        </div>

        <div className="mt-6 text-left bg-gray-900/70 p-4 rounded-lg border border-gray-600">
            <h4 className="font-semibold text-lg text-blue-300 mb-3">Troubleshooting Tips</h4>
            <ul className="space-y-3 text-gray-400 text-sm list-disc list-inside">
                <li><strong className="text-gray-300">Check the Terminal:</strong> After running `python server.py`, look for a message like `* Running on http://127.0.0.1:5001`. If you see any errors instead, you may be missing required Python packages (e.g., `pip install Flask`).</li>
                <li><strong className="text-gray-300">Firewall:</strong> Ensure that your system's firewall is not blocking connections on port 5001.</li>
                 <li><strong className="text-gray-300">Correct Directory:</strong> Make sure you are in the same folder as the `server.py` file before running the command.</li>
            </ul>
        </div>

        <div className="mt-8">
          <button
            onClick={onRetry}
            className="w-full sm:w-auto inline-flex items-center justify-center bg-yellow-600 hover:bg-yellow-500 text-white font-bold py-3 px-6 rounded-lg transition-colors text-lg"
          >
            Retry Connection
          </button>
        </div>
      </div>
    </main>
  );
};

export default DisconnectedPanel;
