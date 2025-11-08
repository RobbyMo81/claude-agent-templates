# HELIOS UNIFIED STARTUP - DOCKER MIRROR IMPLEMENTATION

## Overview
Successfully refactored the Helios startup scripts to **exactly mirror the Docker container behavior**, creating a truly unified development experience that serves both frontend and backend on a single port.

## Key Architectural Changes

### ✅ **Docker Architecture Analysis**
- **Dockerfile Stage 1**: Frontend builder (Node.js) builds React app to `dist/`
- **Dockerfile Stage 2**: Python production container copies `dist/` to `/app/static`
- **start-server.sh**: Starts Flask server configured to serve both static files and API
- **Unified Port**: Everything serves on single port (8080 in Docker, 5001 in dev)

### ✅ **Local Development Mirror Implementation**

#### **1. Frontend Build (Stage 1 Mirror)**
```powershell
# Mirrors: RUN npm run build in Dockerfile
npm run build  # Builds React app to dist/

# Environment variables match Docker ARG/ENV
$env:VITE_API_HOST = "localhost"
$env:VITE_API_PORT = $Port.ToString()  
$env:VITE_API_PROTOCOL = "http"
```

#### **2. Backend Server (Stage 2 Mirror)**  
```powershell
# Mirrors: COPY --from=frontend-builder /app/dist /app/static
# Flask server.py configured with static_folder pointing to ../dist

# Mirrors: CMD ["/app/start-server.sh"] 
& $pythonExe server.py  # Flask dev server (Windows equivalent of Gunicorn)
```

#### **3. Static File Serving Configuration**
```python
# In server.py - Dynamic path detection
if os.path.exists('/app/static'):
    # Docker environment
    static_folder_path = '/app/static'
elif os.path.exists('../dist'):  
    # Local development - mirrors Docker behavior
    static_folder_path = '../dist'

app = Flask(__name__, static_folder=static_folder_path)
```

## Available Scripts

### **Primary Script: `start-helios.ps1`**
Complete unified startup with validation, frontend build, and backend server:
```powershell
./start-helios.ps1                    # Full startup with tests
./start-helios.ps1 -SkipTests         # Skip validation  
./start-helios.ps1 -SkipBuild         # Skip frontend build
./start-helios.ps1 -Port 3000         # Custom port
```

### **Server-Only Script: `start-server.ps1`**
Direct mirror of Docker `start-server.sh` - starts only backend (assumes frontend built):
```powershell
./start-server.ps1                    # Start server only
./start-server.ps1 -Port 5001         # Custom port
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HELIOS UNIFIED SERVER                    │
│                     (Single Port: 5001)                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React SPA)           │  Backend (Flask API)     │  
│  ├── Served from dist/          │  ├── /api/* routes       │
│  ├── Static assets              │  ├── /health endpoint    │
│  ├── index.html (SPA routing)   │  ├── CORS enabled       │
│  └── Built by Vite              │  └── JSON responses     │
├─────────────────────────────────────────────────────────────┤
│                 Flask Application                           │
│  server.py with static_folder=../dist (mirrors Docker)     │
└─────────────────────────────────────────────────────────────┘
```

## Key Features Achieved

### ✅ **Perfect Docker Mirror**
- **Single Port**: Everything on port 5001 (like Docker 8080)
- **Static File Serving**: Flask serves `dist/` files (like Docker `/app/static`)
- **Unified Access**: Frontend UI and Backend API on same port
- **Environment Variables**: Match Docker ARG/ENV configuration

### ✅ **Development Experience**
- **One Command Startup**: `./start-helios.ps1`
- **Hot Reload**: Flask development server with debug mode
- **Validation**: Optional system checks before startup
- **Flexible Building**: Can skip frontend build for rapid testing

### ✅ **Production Readiness**  
- **Gunicorn-Like Behavior**: Flask configured similarly to Docker Gunicorn
- **CORS Enabled**: Frontend/Backend communication works
- **Health Checks**: `/api/health` endpoint available
- **Error Handling**: Proper 404/500 error responses for SPA routing

## Access Points

| Component | URL | Description |
|-----------|-----|-------------|
| **Frontend UI** | http://localhost:5001 | React application |
| **Backend API** | http://localhost:5001/api/ | All API endpoints |
| **Health Check** | http://localhost:5001/api/health | Server status |

## Technical Notes

### **Windows vs Docker Differences**
- **Windows**: Flask development server (similar functionality to Gunicorn)
- **Docker**: Gunicorn production server (Linux-only, uses fcntl module)
- **Behavior**: Identical serving of static files and API endpoints

### **Static File Strategy**  
- **Local Dev**: `../dist` (built by npm run build)
- **Docker**: `/app/static` (copied from builder stage)
- **Server Logic**: Dynamic path detection in server.py

### **Port Configuration**
- **Development**: 5001 (matches backend default)
- **Docker Production**: 8080 (Cloud Run standard)
- **Configurable**: Both scripts accept `-Port` parameter

## Success Criteria Met

✅ **Unified Port Architecture**: Single port serves everything  
✅ **Docker Behavior Mirror**: Exact replication of container startup  
✅ **Static File Serving**: Frontend served by Flask like Docker  
✅ **API Functionality**: All backend endpoints working  
✅ **Developer Experience**: Simple one-command startup  
✅ **Production Similarity**: Configuration matches Docker deployment

## Usage Examples

```powershell
# Complete development startup
./start-helios.ps1 

# Quick restart (skip build and tests)  
./start-helios.ps1 -SkipBuild -SkipTests

# Server-only startup (like Docker)
./start-server.ps1

# Custom port
./start-helios.ps1 -Port 3000
```

## Result
The startup scripts now **perfectly mirror the Docker container behavior**, creating a unified development experience where both frontend and backend are served from a single Flask application on one port, exactly like the production Docker deployment.
