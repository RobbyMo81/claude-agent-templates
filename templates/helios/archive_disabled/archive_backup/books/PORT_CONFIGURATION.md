# Helios Port Configuration - Unified Strategy

## Overview
Helios uses a **unified port strategy** to ensure consistency across development, local production, and Docker deployments.

## Port Configuration Strategy

### Development Environment
- **Frontend (Vite Dev Server)**: Port `5173` (Vite default)
- **Backend (Python/Flask)**: Port `5001` (default in server.py)
- **Development URL**: http://localhost:5173 (frontend) → http://localhost:5001 (API calls)

### Local Production Simulation
- **Unified Server**: Port `5001` (default, can be overridden with `-Port` parameter)
- **Command**: `.\start-helios.ps1` or `.\start-helios.ps1 -Port 5001`
- **Access URL**: http://localhost:5001 (serves both frontend build + API)

### Docker Production
- **Container Internal**: Port `8080`
- **External Mapping**: Configurable (e.g., `docker run -p 3000:8080`)
- **Docker Compose**: Backend on `5001`, Frontend on `80`/`443`
- **Cloud Run**: Uses `PORT` environment variable (typically `8080`)

### Backend API Server
- **Port**: `5001`
- **Protocol**: `HTTP`
- **URL**: `http://localhost:5001`

### Frontend Development Server (Vite)
- **Port**: `5173` (Vite default)
- **Protocol**: `HTTP`
- **URL**: `http://localhost:5173`

## Configuration Types

### 1. Static Configuration (Current Default)
Configuration is set at build time and doesn't change during runtime:
```typescript
const DEFAULT_CONFIG: AppConfig = {
    api: {
        protocol: 'http',
        host: 'localhost',
        port: 5001, // Static value
    }
};
```

### 2. Dynamic Configuration (Available)

#### A. Environment Variables (Build-time Dynamic)
Create a `.env` file to override defaults:
```env
VITE_API_PORT=5001
VITE_API_HOST=localhost
VITE_API_PROTOCOL=http
VITE_FRONTEND_PORT=5173
```

To enable environment variable support, uncomment these lines in `config.ts`:
```typescript
// config.api.protocol = import.meta.env.VITE_API_PROTOCOL || config.api.protocol;
// config.api.host = import.meta.env.VITE_API_HOST || config.api.host;
// config.api.port = Number(import.meta.env.VITE_API_PORT) || config.api.port;
```

#### B. Runtime Dynamic Configuration
Use the provided functions to change configuration during runtime:

```typescript
import { updateApiPort, updateApiHost, updateApiProtocol } from './services/config';

// Change port at runtime
updateApiPort(5002);

// Change host at runtime  
updateApiHost('192.168.1.100');

// Change protocol at runtime
updateApiProtocol('https');

// Update multiple properties
updateFullApiConfig({
    port: 5003,
    host: 'api.myserver.com',
    protocol: 'https'
});
```

## How to Change Ports

### Backend Port
To change the backend port, edit the `DEFAULT_CONFIG` in `services/config.ts`:

```typescript
const DEFAULT_CONFIG: AppConfig = {
    api: {
        protocol: 'http',
        host: 'localhost',
        port: 5001, // Change this value
        baseUrl: '',
    },
    // ...
};
```

### Frontend Port
To change the frontend development server port, you can:

1. **Command line**: `npm run dev -- --port 3000`
2. **Add to vite.config.ts**:
   ```typescript
   export default defineConfig({
     server: {
       port: 3000
     },
     // ...
   });
   ```

## Services Using Unified Configuration

- **modelService.ts**: Uses `getApiEndpoint()` for all API calls
- **config.ts**: Central configuration management

## Benefits

1. **Single source of truth** for all network configuration
2. **Easy to change ports** in one location
3. **Consistent error messages** across services
4. **Future-ready** for environment variable support
5. **Type-safe** configuration with TypeScript interfaces

## Future Enhancements

The configuration system is designed to support environment variables in the future:

```typescript
// Example for future .env support
VITE_API_PORT=5001
VITE_API_HOST=localhost
VITE_API_PROTOCOL=http
VITE_FRONTEND_PORT=5173
```

## API Endpoints

All API endpoints are now constructed using the `getApiEndpoint()` helper:

- Models: `getApiEndpoint('/api/models')`
- Training: `getApiEndpoint('/api/train')`
- Journal: `getApiEndpoint('/api/models/{modelName}/journal')`

This ensures consistency and makes it easy to change the base URL if needed.
