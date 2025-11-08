# Docker Setup for Helios Project

This document provides complete instructions for running the Helios project using Docker containers.

## Architecture

The Helios project consists of two main services:

1. **Frontend**: React + Vite application (port 80/5173)
2. **Backend**: Python Flask API server (port 5001)

## Quick Start

### Prerequisites

- Docker Desktop installed
- Docker Compose installed

### 1. Production Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Access the application
# Frontend: http://localhost
# Backend API: http://localhost:5001
```

### 2. Development Mode

```bash
# Start with development profile (hot-reload enabled)
docker-compose --profile dev up -d

# Access development server
# Frontend: http://localhost:5173
# Backend API: http://localhost:5001
```

## Detailed Commands

### Building Services

```bash
# Build all services
docker-compose build

# Build specific service
docker-compose build frontend
docker-compose build backend

# Force rebuild (no cache)
docker-compose build --no-cache
```

### Managing Services

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Restart specific service
docker-compose restart frontend

# View service status
docker-compose ps

# View logs
docker-compose logs -f frontend
docker-compose logs -f backend
```

### Environment Configuration

The project supports different environment configurations:

#### Development (.env.development)
- Frontend: http://localhost:5173
- Backend: http://localhost:5001
- API Host: localhost

#### Production (.env.production)
- Frontend: http://localhost (port 80)
- Backend: http://backend:5001 (internal Docker network)
- API Host: backend (Docker service name)

### Health Checks

Both services include health checks:

```bash
# Check service health
docker-compose ps

# Manual health check
curl http://localhost/
curl http://localhost:5001/health
```

### Scaling Services

```bash
# Scale backend service
docker-compose up -d --scale backend=2

# Scale services to specific count
docker-compose up -d --scale backend=3 --scale frontend=1
```

### Data Persistence

- Backend logs are persisted in the `helios-backend-logs` volume
- Application data can be mounted as volumes if needed

### Troubleshooting

#### Common Issues

1. **Port conflicts**
   ```bash
   # Check what's using the ports
   netstat -an | grep :80
   netstat -an | grep :5001

   # Change ports in docker-compose.yml if needed
   ```

2. **Build failures**
   ```bash
   # Clean build
   docker-compose down
   docker system prune -f
   docker-compose build --no-cache
   ```

3. **Network issues**
   ```bash
   # Recreate network
   docker-compose down
   docker network rm helios-network
   docker-compose up -d
   ```

#### Debugging

```bash
# Enter container shell
docker-compose exec frontend sh
docker-compose exec backend bash

# View container resource usage
docker stats

# Inspect container configuration
docker inspect helios-frontend
docker inspect helios-backend
```

### Security Considerations

1. **Non-root users**: All containers run as non-root users
2. **Security headers**: Nginx includes security headers
3. **Environment variables**: Store sensitive data in `.env` files (not committed to git)
4. **Network isolation**: Services communicate through Docker network

### Production Deployment

For production deployment:

1. **Set environment variables**:
   ```bash
   # Copy and edit production environment
   cp .env.production .env
   # Edit .env with your actual values
   ```

2. **Use SSL/TLS**:
   - Add SSL certificates to nginx configuration
   - Update docker-compose.yml to mount certificates
   - Change ports to 443 for HTTPS

3. **Resource limits**:
   ```yaml
   # Add to docker-compose.yml services
   deploy:
     resources:
       limits:
         cpus: '1'
         memory: 512M
   ```

4. **Monitoring**:
   - Add monitoring services (Prometheus, Grafana)
   - Configure log aggregation
   - Set up alerts

### API Endpoints

The backend exposes the following endpoints:

- `GET /` - API information
- `GET /health` - Health check
- `GET /api/models` - Get available models
- `POST /api/train` - Start training job
- `GET /api/models/{model_name}/journal` - Get training journal

### Development Workflow

1. **Local development**:
   ```bash
   # Start development environment
   docker-compose --profile dev up -d

   # Code changes are reflected immediately (hot-reload)
   # Edit files in your local directory
   ```

2. **Testing changes**:
   ```bash
   # Rebuild after dependency changes
   docker-compose build frontend-dev
   docker-compose --profile dev up -d
   ```

3. **Production testing**:
   ```bash
   # Test production build locally
   docker-compose up -d
   ```

### Cleanup

```bash
# Stop and remove containers
docker-compose down

# Remove volumes (careful - this deletes data!)
docker-compose down -v

# Complete cleanup
docker-compose down -v --rmi all
docker system prune -a
```
