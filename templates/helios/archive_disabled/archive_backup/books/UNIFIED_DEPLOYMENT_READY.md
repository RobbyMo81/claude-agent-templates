# **HELIOS UNIFIED DEPLOYMENT - READY FOR CLOUD RUN**

## **Deployment Architecture**

### **Single Container Solution** 
**Frontend + Backend Unified**: One container serves both React frontend and Python Flask API  
**Production Ready**: Optimized multi-stage Docker build  
**Cloud Run Compatible**: Dynamic port binding with `$PORT` environment variable  
**Security Hardened**: Non-root user, minimal attack surface  

---

## **What Changed**

### **Previous Setup (Frontend Only)**
- Only built and served React static files via nginx
- No Python backend included
- API calls would fail (no backend to connect to)

### **New Unified Setup**
- **Stage 1**: Builds React frontend into static files (`/app/static`)
- **Stage 2**: Python environment with Flask backend (`/app/backend`) 
- **Runtime**: Gunicorn serves both API endpoints AND static files
- **Smart Routing**: API calls go to Flask, everything else serves React SPA

---

## **Container Structure**

```
/app/
├── static/          # Built React frontend (from Stage 1)
│   ├── index.html   
│   ├── assets/      # CSS, JS, images
│   └── ...
├── backend/         # Python Flask application
│   ├── server.py    # Main Flask app (modified for static serving)
│   ├── agent.py     # ML Powerball Agent
│   ├── trainer.py   # Model training
│   ├── memory_store.py
│   ├── metacognition.py
│   ├── decision_engine.py
│   ├── cross_model_analytics.py
│   └── requirements.txt
├── data/           # Persistent storage volume mount point
└── start-server.sh # Startup script
```

---

## **Key Modifications**

### **1. Dockerfile (Completely Rewritten)**
```dockerfile
# Multi-stage build:
# Stage 1: Build React frontend with Node.js 22.13.1
# Stage 2: Python 3.11-slim (Debian-based for PyTorch compatibility)

FROM node:22.13.1-alpine3.21 AS frontend-builder
# ... builds React app to /app/dist

FROM python:3.11-slim  
# ... sets up Python with Debian base for PyTorch wheel compatibility
```

** Critical Fix**: Switched from `python:3.11-alpine3.21` to `python:3.11-slim` to resolve PyTorch installation issues. Alpine's musl libc is incompatible with PyTorch wheels, causing "No matching distribution found" errors.

### **2. Flask Server (Enhanced for Static Serving)**
- **Added imports**: `send_file`, `send_from_directory` for static serving
- **Modified app creation**: `Flask(__name__, static_folder='/app/static')`  
- **New routes**:
  - `GET /` → Serves `index.html` (React app)
  - `GET /<path>` → Serves static assets or SPA fallback
  - Enhanced 404 handler for SPA routing

### **3. Startup Script (`start-server.sh`)**
```bash
#!/bin/sh
echo " Starting Helios Unified Server..."
cd /app/backend

exec gunicorn \
    --bind 0.0.0.0:${PORT:-8080} \
    --workers 1 \
    --timeout 300 \
    server:app
```

---

## **Request Flow**

### **Frontend Requests**
- `GET /` → Flask serves `index.html` 
- `GET /assets/main.js` → Flask serves from `/app/static/assets/`
- `GET /some-spa-route` → Flask serves `index.html` (SPA routing)

### **API Requests**  
- `GET /api/models` → Flask handles via backend routes
- `POST /api/train` → Flask processes training request
- `GET /api/health` → Flask returns backend health status

---

## **Deployment Commands**

### **Build & Deploy to Cloud Run**
```bash
# Build the unified container
gcloud builds submit --tag gcr.io/YOUR-PROJECT/helios

# Deploy to Cloud Run  
gcloud run deploy helios \
  --image gcr.io/YOUR-PROJECT/helios \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --port 8080
```

### **Persistent Storage Configuration**
Your trained models and SQLite database (`helios_memory.db`) will be created in the persistent volume mounted at `/app/data/`. Configure Cloud Run with:

```bash
--add-volume name=helios-data,type=cloud-storage,bucket=your-helios-bucket
--add-volume-mount volume=helios-data,mount-path=/app/data
```

---

## **Production Readiness Checklist**

### **Architecture**
- Single container for simplified deployment
- Multi-stage build for optimal image size  
- Production-grade Gunicorn WSGI server
- Dynamic port binding for Cloud Run

### **Security**  
- Non-root user (`helios:helios`)
- Minimal Alpine Linux base images
- Security updates applied
- Proper file permissions

### **Performance**
- Static assets served efficiently by Flask
- Gunicorn optimized for single-worker deployment
- Frontend build optimized (Vite production build)
- Health checks configured

### **Reliability**
- Proper error handling for missing static files
- SPA routing fallback to `index.html`
- Graceful degradation if frontend unavailable
- Signal handling with `dumb-init`

---

## **Next Steps**

1. **Deploy**: Run the Cloud Build and Cloud Run commands above
2. **Test**: Verify both frontend loads and API endpoints work  
3. **Configure Storage**: Set up persistent volume for model/database persistence
4. **Monitor**: Use Cloud Run logs to monitor deployment health

Your Helios system is now ready for production deployment with both frontend and backend in a single, optimized container!

---
*Unified Deployment Configuration Complete*  
*Generated: July 20, 2025*
