# Python Virtual Environment Setup for Helios Project

This guide helps you set up and manage the Python virtual environment for the Helios backend.

## Quick Setup

### Windows (Command Prompt/PowerShell)
```bash
# Navigate to project root
cd c:\Users\RobMo\OneDrive\Documents\helios

# Run setup script
setup-venv.bat
```

### Windows (PowerShell)
```powershell
# Navigate to project root
cd c:\Users\RobMo\OneDrive\Documents\helios

# Run PowerShell setup script
.\setup-venv.ps1
```

### Unix/Linux/macOS
```bash
# Navigate to project root
cd /path/to/helios

# Run setup script
bash setup-venv.sh
```

## Manual Setup

If the setup scripts don't work, you can set up the virtual environment manually:

### 1. Create Virtual Environment
```bash
# Windows
python -m venv backend/venv

# Or if you have Python 3 specifically
python3 -m venv backend/venv
```

### 2. Activate Virtual Environment

#### Windows Command Prompt
```cmd
backend\venv\Scripts\activate
```

#### Windows PowerShell
```powershell
backend\venv\Scripts\Activate.ps1
```

#### Unix/Linux/macOS
```bash
source backend/venv/bin/activate
```

### 3. Install Dependencies
```bash
# Make sure you're in the activated virtual environment
# You should see (venv) in your prompt

pip install --upgrade pip
pip install -r backend/requirements.txt
```

### 4. Verify Installation
```bash
python --version
pip list
```

## Using the Virtual Environment

### Activating the Environment

Every time you work on the Python backend, activate the virtual environment first:

#### Windows Command Prompt
```cmd
backend\venv\Scripts\activate
```

#### Windows PowerShell
```powershell
backend\venv\Scripts\Activate.ps1
```

#### Unix/Linux/macOS
```bash
source backend/venv/bin/activate
```

### Running the Backend Server

With the virtual environment activated:
```bash
# Navigate to backend directory
cd backend

# Run the Flask server
python server.py

# Or using Flask command
flask run --host=0.0.0.0 --port=5001
```

### Deactivating the Environment

When you're done working:
```bash
deactivate
```

## Package Management

### Installing New Packages
```bash
# Activate virtual environment first
backend\venv\Scripts\activate  # Windows
# source backend/venv/bin/activate  # Unix

# Install package
pip install package-name

# Update requirements.txt
pip freeze > backend/requirements.txt
```

### Updating Dependencies
```bash
# Activate virtual environment
backend\venv\Scripts\activate  # Windows

# Upgrade all packages
pip install --upgrade -r backend/requirements.txt

# Or upgrade pip itself
pip install --upgrade pip
```

## NPM Scripts Integration

The following npm scripts are available for managing the Python environment:

```bash
# Setup virtual environment and install dependencies
npm run python:setup

# Activate virtual environment and start backend server
npm run python:dev

# Install Python dependencies
npm run python:install

# Run Python backend in production mode
npm run python:start
```

## Docker Integration

The virtual environment is primarily for local development. For production deployment, use Docker:

```bash
# Build and run with Docker
npm run docker:up

# Or build backend only
docker build -t helios-backend backend/
```

## Troubleshooting

### Virtual Environment Not Activating (Windows)

If PowerShell blocks script execution:
```powershell
# Set execution policy (run as Administrator)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Or use bypass for single execution
powershell -ExecutionPolicy Bypass -File backend\venv\Scripts\Activate.ps1
```

### Python Not Found

Make sure Python is installed and in your PATH:
```bash
# Check Python installation
python --version
python3 --version

# If not found, download from https://python.org
```

### Permission Errors

On Windows, run Command Prompt or PowerShell as Administrator.

### Package Installation Fails

Update pip first:
```bash
python -m pip install --upgrade pip
```

## Environment Variables

Create a `.env` file in the backend directory for local configuration:

```env
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_APP=server.py
API_PORT=5001
API_HOST=localhost
```

## Project Structure

After setup, your project structure should look like:

```
helios/
├── backend/
│   ├── venv/              # Virtual environment (created)
│   │   ├── Scripts/       # Windows executables
│   │   ├── Lib/          # Python packages
│   │   └── ...
│   ├── server.py         # Flask application
│   ├── requirements.txt  # Python dependencies
│   ├── Dockerfile        # Docker configuration
│   └── .env             # Environment variables (optional)
├── services/             # Frontend services
├── components/           # React components
└── ...
```

## Best Practices

1. **Always activate the virtual environment** before working on Python code
2. **Update requirements.txt** when adding new packages
3. **Use the virtual environment for local development** and Docker for production
4. **Keep the virtual environment out of version control** (already in .gitignore)
5. **Document any special setup requirements** in this file

## Quick Commands Reference

```bash
# Setup (one time)
npm run python:setup

# Daily workflow
npm run python:dev        # Start backend in development
npm run dev              # Start frontend in development

# Package management
npm run python:install   # Install Python dependencies
npm install              # Install Node.js dependencies

# Production
npm run docker:up        # Full stack with Docker
```
