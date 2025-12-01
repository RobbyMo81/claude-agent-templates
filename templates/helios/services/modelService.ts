import { TrainingConfig, ModelJournal, TrainingJobResponse } from '../types';
import { getApiEndpoint, CONNECTION_ERROR_MESSAGE } from './config';

/**
 * A helper function to create a more user-friendly error message for failed fetch requests.
 * @param error - The error caught during the fetch operation.
 * @returns A new Error with a more descriptive message.
 */
const createFetchError = (error: any): Error => {
    if (error instanceof TypeError && error.message === 'Failed to fetch') {
        return new Error(CONNECTION_ERROR_MESSAGE);
    }
    return error;
};

/**
 * A helper function to parse an error response from the server.
 * Tries to parse JSON, but falls back to the status text if parsing fails.
 * @param response - The Response object from a failed fetch call.
 * @returns A new Error with the parsed message.
 */
const createResponseError = async (response: Response): Promise<Error> => {
    try {
        const errorData = await response.json();
        return new Error(errorData.error || `Server responded with an error: ${response.statusText}`);
    } catch (e) {
        return new Error(`Server responded with an error: ${response.status} ${response.statusText}`);
    }
}


/**
 * Service object for interacting with the Python ML backend API.
 */
export const modelService = {
    /**
     * Fetches the list of available model names from the backend.
     */
    getModels: async (): Promise<string[]> => {
        try {
            const response = await fetch(getApiEndpoint('/api/models'));
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Sends a request to the backend to start a new training job.
     */
    startTraining: async (config: TrainingConfig): Promise<TrainingJobResponse> => {
         try {
            const response = await fetch(getApiEndpoint('/api/train'), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config),
            });

            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Fetches the training journal for a specific model.
     */
    getModelJournal: async (modelName: string): Promise<ModelJournal> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/models/${modelName}/journal`));
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Load a specific model for predictions.
     */
    loadModel: async (modelName: string): Promise<{ status: string; message: string; model_info?: any }> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/models/${modelName}/load`), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Get detailed information about a specific model.
     */
    getModelInfo: async (modelName: string): Promise<any> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/models/${modelName}/info`));
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Generate predictions using a specific model.
     */
    generatePrediction: async (modelName: string, inputData?: any): Promise<any> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/models/${modelName}/predict`), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputData || {})
            });
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Delete a specific model.
     */
    deleteModel: async (modelName: string): Promise<{ status: string; message: string }> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/models/${modelName}`), {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Get training status for a specific job.
     */
    getTrainingStatus: async (jobId: string): Promise<any> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/train/status/${jobId}`));
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Stop a training job.
     */
    stopTraining: async (jobId: string): Promise<{ status: string; message: string }> => {
        try {
            const response = await fetch(getApiEndpoint(`/api/train/stop/${jobId}`), {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    },

    /**
     * Get training history for all models.
     */
    getTrainingHistory: async (): Promise<any[]> => {
        try {
            const response = await fetch(getApiEndpoint('/api/train/history'));
            if (!response.ok) {
                throw await createResponseError(response);
            }
            return response.json();
        } catch (error) {
            throw createFetchError(error);
        }
    }
};
