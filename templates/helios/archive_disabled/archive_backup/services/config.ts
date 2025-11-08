/**
 * Unified configuration service for all API endpoints and ports.
 * This centralizes all network configuration in one place for easier maintenance.
 */

export interface ApiConfig {
    baseUrl: string;
    port: number;
    protocol: 'http' | 'https';
    host: string;
}

export interface AppConfig {
    api: ApiConfig;
    development: {
        frontendPort: number;
    };
}

/**
 * Default configuration - can be overridden by environment variables
 */
const DEFAULT_CONFIG: AppConfig = {
    api: {
        protocol: 'http',
        host: 'localhost',
        port: 5001,
        baseUrl: '', // Will be constructed automatically
    },
    development: {
        frontendPort: 5173, // Vite default
    }
};

/**
 * Constructs the full API base URL from protocol, host, and port
 */
const constructBaseUrl = (config: ApiConfig): string => {
    return `${config.protocol}://${config.host}:${config.port}`;
};

/**
 * Loads configuration from environment variables or uses defaults
 */
const loadConfig = (): AppConfig => {
    const config = { ...DEFAULT_CONFIG };
    
    // Override with environment variables if available (Vite environment variables)
    // To use this, create a .env file with variables like:
    // VITE_API_PORT=5001
    // VITE_API_HOST=localhost
    // VITE_API_PROTOCOL=http
    
    try {
        // Enable environment variable support for Docker deployments
        // These variables are injected at build time by Vite
        const envApiProtocol = (globalThis as any).VITE_API_PROTOCOL;
        const envApiHost = (globalThis as any).VITE_API_HOST;
        const envApiPort = (globalThis as any).VITE_API_PORT;
        const envFrontendPort = (globalThis as any).VITE_FRONTEND_PORT;
        
        if (envApiProtocol) config.api.protocol = envApiProtocol;
        if (envApiHost) config.api.host = envApiHost;
        if (envApiPort) config.api.port = Number(envApiPort);
        if (envFrontendPort) config.development.frontendPort = Number(envFrontendPort);
        
    } catch (error) {
        // Fallback to defaults if environment variables aren't available
        console.warn('Using default configuration - environment variables not available');
    }
    
    // Auto-detect unified deployment: if frontend and backend are on same origin
    if (typeof window !== 'undefined' && window.location) {
        const currentHost = window.location.hostname;
        const currentPort = window.location.port;
        const currentProtocol = window.location.protocol.replace(':', '') as 'http' | 'https';
        
        // If we're being served from the same host/port, assume unified deployment
        if (currentHost && currentPort && 
            (currentHost !== 'localhost' || currentPort !== '5173')) {
            console.log(' Detected unified deployment - using current origin for API calls');
            config.api.protocol = currentProtocol;
            config.api.host = currentHost;
            config.api.port = Number(currentPort) || config.api.port;
        }
    }
    
    // Construct the base URL
    config.api.baseUrl = constructBaseUrl(config.api);
    
    return config;
};

/**
 * The unified configuration object - import this in other services
 */
export const appConfig = loadConfig();

/**
 * Convenience getters for common configuration values
 */
export const getApiBaseUrl = (): string => appConfig.api.baseUrl;
export const getApiPort = (): number => appConfig.api.port;
export const getFrontendPort = (): number => appConfig.development.frontendPort;

/**
 * Helper function to construct API endpoint URLs
 */
export const getApiEndpoint = (path: string): string => {
    const cleanPath = path.startsWith('/') ? path : `/${path}`;
    return `${appConfig.api.baseUrl}${cleanPath}`;
};

/**
 * Connection error message that can be used across services
 */
export const CONNECTION_ERROR_MESSAGE = `Connection to the backend server failed. Please ensure the Python server is running on ${appConfig.api.baseUrl}`;

/**
 * Runtime configuration update functions (makes config dynamic)
 */
export const updateApiPort = (newPort: number): void => {
    appConfig.api.port = newPort;
    appConfig.api.baseUrl = constructBaseUrl(appConfig.api);
};

export const updateApiHost = (newHost: string): void => {
    appConfig.api.host = newHost;
    appConfig.api.baseUrl = constructBaseUrl(appConfig.api);
};

export const updateApiProtocol = (newProtocol: 'http' | 'https'): void => {
    appConfig.api.protocol = newProtocol;
    appConfig.api.baseUrl = constructBaseUrl(appConfig.api);
};

export const updateFullApiConfig = (newConfig: Partial<ApiConfig>): void => {
    Object.assign(appConfig.api, newConfig);
    appConfig.api.baseUrl = constructBaseUrl(appConfig.api);
};

/**
 * Get current configuration (useful for debugging)
 */
export const getCurrentConfig = (): AppConfig => {
    return { ...appConfig };
};
