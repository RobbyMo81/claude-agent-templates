# Full Stack Integration Guide

This guide helps you set up and test the complete Helios full stack application.

## 🚀 Quick Start

### Option 1: Manual Setup (Recommended for Development)

1. **Setup Python Virtual Environment** (one time only):
   ```bash
   npm run python:setup
   ```

2. **Start Backend Server** (Terminal 1):
   ```bash
   npm run python:dev
   ```
   This will start the Flask backend on http://localhost:5001

3. **Start Frontend Server** (Terminal 2):
   ```bash
   npm run dev
   ```
   This will start the React frontend on http://localhost:5173 or 5174

### Option 2: Docker Setup (For Production)

```bash
# Build and start all services
npm run docker:up

# View logs
npm run docker:logs

# Stop services
npm run docker:down
```

## 🔗 Testing Full Stack Integration

### Browser Console Tests

1. **Open the frontend** in your browser (http://localhost:5174)

2. **Open browser console** (F12 → Console tab)

3. **Run integration tests**:
   ```javascript
   // Quick connectivity test
   testConnection()

   // Comprehensive full stack test
   testFullStack()
   ```

### Manual API Testing

Test backend endpoints directly:

```bash
# Health check
curl http://localhost:5001/health

# Available models
curl http://localhost:5001/api/models

# Root endpoint info
curl http://localhost:5001/
```

## 📋 Configuration Overview

The system uses unified configuration in `services/config.ts`:

```typescript
// Default configuration
const config = {
  api: {
    protocol: 'http',
    host: 'localhost',
    port: 5001,
    baseUrl: 'http://localhost:5001'
  },
  development: {
    frontendPort: 5173
  }
}
```

## 🔧 Available Services

### Frontend (React/TypeScript/Vite)
- **Development**: http://localhost:5173 or 5174
- **Production**: http://localhost (via Docker)
- **Features**:
  - Unified configuration system
  - Real-time backend connectivity testing
  - Stress test UI components
  - Anomaly detection interface

### Backend (Python/Flask)
- **Development**: http://localhost:5001
- **Production**: http://localhost:5001 (via Docker)
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /` - Service info
  - `GET /api/models` - Available models
  - `POST /api/train` - Start training
  - `GET /api/models/{name}/journal` - Training logs

## 🧪 Integration Test Results

When you run `testFullStack()` in the browser console, you should see:

```
🔗 Helios Full Stack Integration Test
====================================

📋 Configuration:
   Frontend URL: http://localhost:5174
   Backend URL: http://localhost:5001

🏥 Testing Backend Health...
   ✅ Backend Health: PASSED
   📊 Response: {status: "healthy", service: "helios-backend"}

🤖 Testing Models API...
   ✅ Models API: PASSED
   📊 Available models: ["random_forest", "neural_network", ...]

🌐 Testing CORS Configuration...
   ✅ CORS: PASSED

📊 Test Results Summary
========================
Total Tests: 3
✅ Passed: 3
❌ Failed: 0
Success Rate: 100%

🎉 ALL TESTS PASSED! Full stack integration is working perfectly!
```

## 🛠️ Troubleshooting

### Backend Not Responding

1. **Check if backend is running**:
   ```bash
   # In the terminal where you started the backend
   # You should see: "Running on http://127.0.0.1:5001"
   ```

2. **Restart backend**:
   ```bash
   # Ctrl+C to stop, then restart
   npm run python:dev
   ```

3. **Check virtual environment**:
   ```bash
   # Verify virtual environment exists
   dir backend\venv  # Windows
   ls backend/venv   # Unix
   ```

### Frontend Not Loading

1. **Check if frontend is running**:
   ```bash
   # Should show: "Local: http://localhost:5173/"
   npm run dev
   ```

2. **Clear browser cache** and reload

3. **Check console for errors** (F12 → Console)

### CORS Errors

1. **Verify CORS is enabled** in `backend/server.py`:
   ```python
   from flask_cors import CORS
   CORS(app)  # Should be present
   ```

2. **Check frontend is using correct backend URL**:
   ```javascript
   // In browser console
   getCurrentConfig()
   ```

### Port Conflicts

1. **Backend port 5001 in use**:
   - Kill existing processes or change port in config

2. **Frontend port 5173/5174 in use**:
   - Vite will automatically try the next available port

## 📊 Performance and Monitoring

### Development Monitoring

```bash
# Watch backend logs
npm run python:dev

# Watch frontend build
npm run dev

# Monitor system health
npm run stress-test
```

### Production Monitoring

```bash
# Docker logs
npm run docker:logs

# Container status
docker-compose ps

# Resource usage
docker stats
```

## 🚀 Next Steps

Once integration is working:

1. **Upload CSV data** through the frontend
2. **Start model training** via the UI
3. **View results** in the analytics panels
4. **Run stress tests** to validate system performance
5. **Deploy to production** using Docker

## 📚 Related Documentation

- [Python Virtual Environment Setup](PYTHON_VENV_SETUP.md)
- [Docker Setup Guide](DOCKER_SETUP.md)
- [Port Configuration](PORT_CONFIGURATION.md)
- [Stress Testing Guide](STRESS_TEST_GUIDE.md)
- [Windows Troubleshooting](WINDOWS_TROUBLESHOOTING.md)

## 🆘 Getting Help

If you encounter issues:

1. **Check the browser console** for frontend errors
2. **Check the backend terminal** for Python errors
3. **Run the stress test**: `npm run stress-test`
4. **Test API endpoints manually** with curl or browser
5. **Verify configuration** with `getCurrentConfig()` in browser console
