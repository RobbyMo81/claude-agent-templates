# Helios System Schema

## 📋 System Architecture Overview

The Helios Powerball Anomaly Detection system is a full-stack application with React/TypeScript frontend, Python/Flask backend, unified configuration management, Docker containerization, and comprehensive testing infrastructure.

## 🏗️ Project Structure

```
helios/
├── 📁 Root Configuration & Entry Points
├── 📁 Frontend (React/TypeScript/Vite)
├── 📁 Backend (Python/Flask)
├── 📁 Services (API & Configuration)
├── 📁 Components (React UI)
├── 📁 Testing & Validation
├── 📁 Docker & Deployment
├── 📁 Documentation
└── 📁 Environment & Setup
```

## 📁 Detailed File Structure

### 🔧 Root Configuration & Entry Points
```
├── index.html                    ✅ Main HTML entry point
├── index.tsx                     ✅ React application entry point
├── App.tsx                       ✅ Main React application component
├── package.json                  ✅ NPM dependencies and scripts
├── package-lock.json             ✅ NPM lock file
├── tsconfig.json                 ✅ TypeScript configuration
├── vite.config.ts                ✅ Vite build configuration
├── types.ts                      ✅ Global TypeScript type definitions
├── constants.ts                  ✅ Application constants
├── metadata.json                 ✅ Project metadata
└── .gitignore                    ✅ Git ignore rules
```

### 🎨 Frontend (React/TypeScript/Vite)
```
components/
├── Alert.tsx                     ✅ Alert notification component
├── DisconnectedPanel.tsx         ✅ Backend disconnection handling
├── FileUploader.tsx              ✅ CSV file upload component
├── ReflectionPanel.tsx           ✅ Model training reflection/journal viewer
├── ResultsPanel.tsx              ✅ Analysis results display
├── Sidebar.tsx                   ✅ Main navigation sidebar
├── Spinner.tsx                   ✅ Loading spinner component
├── StressTestComponent.tsx       ✅ Stress testing UI component
├── StressTestReportPanel.tsx     ✅ Stress test results display
└── TrainingPanel.tsx             ✅ Model training interface
```

### 🐍 Backend (Python/Flask)
```
backend/
├── server.py                     ✅ Main Flask application server
├── requirements.txt              ✅ Python dependencies
├── Dockerfile                    ✅ Docker container configuration
├── .env.example                  ✅ Environment variables template
└── venv/                         ✅ Python virtual environment (local)
    ├── Scripts/                  ✅ Windows activation scripts
    ├── bin/                      ✅ Unix activation scripts
    └── Lib/                      ✅ Python packages
```

### 🔗 Services (API & Configuration)
```
services/
├── config.ts                     ✅ Unified configuration management
├── api.ts                        ✅ Backend API communication
├── modelService.ts               ✅ Machine learning model services
├── geminiService.ts              ✅ Google Gemini AI integration
└── lotteryService.ts             ✅ Lottery data processing services
```

### 🧪 Testing & Validation
```
tests/
├── stressTest.ts                 ✅ TypeScript stress test suite
├── runStressTest.ts              ✅ TypeScript stress test runner
├── runStressTest.js              ✅ JavaScript stress test runner (ES modules)
├── fullStackTest.mjs             ✅ Full stack integration test
└── fullStackIntegrationTest.js   ✅ Browser-based integration test
```

### 🐳 Docker & Deployment
```
├── docker-compose.yml            ✅ Multi-service orchestration
├── Dockerfile                    ✅ Frontend production container
├── Dockerfile.dev                ✅ Frontend development container
├── nginx.conf                    ✅ Nginx configuration for production
├── .dockerignore                 ✅ Docker ignore rules
├── docker-manage.bat             ✅ Windows Docker management script
└── docker-manage.sh              ✅ Unix Docker management script
```

### 🌍 Environment & Setup
```
├── .env.development              ✅ Development environment variables
├── .env.production               ✅ Production environment variables
├── .env.local                    ✅ Local environment overrides
├── setup-venv.bat                ✅ Windows Python virtual environment setup
├── setup-venv.ps1                ✅ PowerShell virtual environment setup
├── setup-venv.sh                 ✅ Unix virtual environment setup
├── activate-venv.bat             ✅ Quick virtual environment activation (generated)
├── stress-test.bat               ✅ Windows stress testing script
├── stress-test.ps1               ✅ PowerShell stress testing script
└── stress-test.sh                ✅ Unix stress testing script
```

### 📚 Documentation
```
├── README.md                     ✅ Main project documentation
├── DOCKER_SETUP.md               ✅ Docker setup and deployment guide
├── FULL_STACK_INTEGRATION.md     ✅ Full stack integration guide
├── PORT_CONFIGURATION.md         ✅ Port and network configuration guide
├── PYTHON_VENV_SETUP.md          ✅ Python virtual environment guide
├── STRESS_TEST_GUIDE.md          ✅ Stress testing documentation
├── WINDOWS_TROUBLESHOOTING.md    ✅ Windows-specific troubleshooting
└── SYSTEM_SCHEMA.md              ✅ This file - system architecture overview
```

### 🛠️ Development & Debug Files
```
├── browser-test.js               ✅ Browser console testing utilities
└── NPM Enviroment Startup.txt    ✅ Environment setup notes
```

## ❌ Potentially Missing Files

### 🔍 Files That May Be Missing or Needed

#### **1. CSS/Styling Files**
```
❌ index.css                      # Global CSS styles (if not using only Tailwind)
❌ styles/                        # Additional CSS modules or styled components
   ├── components.css             # Component-specific styles
   ├── globals.css                # Global CSS reset/base styles
   └── themes.css                 # Theme configurations
```

#### **2. Backend Data Models & Utils**
```
❌ backend/models/                # Data models and schemas
   ├── __init__.py
   ├── lottery_model.py           # Lottery data models
   ├── anomaly_model.py           # Anomaly detection models
   └── user_model.py              # User/session models

❌ backend/utils/                 # Backend utilities
   ├── __init__.py
   ├── data_processor.py          # Data processing utilities
   ├── ml_helpers.py              # Machine learning helpers
   └── validators.py              # Input validation utilities

❌ backend/routes/                # API route modules (if splitting routes)
   ├── __init__.py
   ├── api_routes.py              # API endpoints
   ├── health_routes.py           # Health check endpoints
   └── model_routes.py            # Model-specific routes
```

#### **3. Configuration Files**
```
❌ .env                          # Local environment file (user-created)
❌ .env.test                     # Test environment variables
❌ config/                       # Additional configuration files
   ├── database.json             # Database configuration (if needed)
   ├── logging.json              # Logging configuration
   └── deployment.json           # Deployment-specific config
```

#### **4. Static Assets**
```
❌ public/                       # Static public assets
   ├── favicon.ico               # Website favicon
   ├── logo.png                  # Application logo
   ├── manifest.json             # PWA manifest (if needed)
   └── robots.txt                # SEO robots file

❌ assets/                       # Application assets
   ├── images/                   # Image assets
   ├── icons/                    # Icon assets
   └── fonts/                    # Custom fonts (if any)
```

#### **5. Testing Infrastructure**
```
❌ .github/                      # GitHub Actions CI/CD (if using GitHub)
   └── workflows/
      ├── test.yml               # Automated testing workflow
      ├── build.yml              # Build workflow
      └── deploy.yml             # Deployment workflow

❌ tests/unit/                   # Unit tests
   ├── components/               # Component unit tests
   ├── services/                 # Service unit tests
   └── utils/                    # Utility unit tests

❌ tests/integration/            # Integration tests
   ├── api.test.js               # API integration tests
   └── e2e.test.js               # End-to-end tests

❌ jest.config.js               # Jest testing configuration
❌ cypress.config.js            # Cypress E2E testing config
```

#### **6. Build & Deployment**
```
❌ dist/                         # Build output directory (generated)
❌ build/                        # Alternative build directory
❌ .vscode/                      # VS Code configuration
   ├── settings.json             # Editor settings
   ├── launch.json               # Debug configuration
   └── extensions.json           # Recommended extensions

❌ kubernetes/                   # Kubernetes deployment (if needed)
   ├── deployment.yaml
   ├── service.yaml
   └── ingress.yaml

❌ terraform/                    # Infrastructure as Code (if needed)
   ├── main.tf
   ├── variables.tf
   └── outputs.tf
```

#### **7. Database Files (if needed)**
```
❌ migrations/                   # Database migrations
❌ seeds/                        # Database seed data
❌ schema.sql                    # Database schema
❌ data/                         # Data files
   ├── sample_data.csv           # Sample lottery data
   ├── test_data.csv             # Test datasets
   └── training_data/            # ML training datasets
```

## 🔧 System Dependencies

### **Node.js Dependencies (package.json)**
```json
{
  "dependencies": {
    "@google/genai": "^1.9.0",      ✅ Installed
    "chart.js": "^4.4.3",           ✅ Installed
    "react": "^19.1.0",             ✅ Installed
    "react-chartjs-2": "^5.2.0",    ✅ Installed
    "react-dom": "^19.1.0"          ✅ Installed
  },
  "devDependencies": {
    "@types/node": "^22.14.0",      ✅ Installed
    "@types/react": "^19.1.8",      ✅ Installed
    "@types/react-dom": "^19.1.6",  ✅ Installed
    "typescript": "~5.7.2",         ✅ Installed
    "vite": "^6.2.0"                ✅ Installed
  }
}
```

### **Python Dependencies (requirements.txt)**
```
Flask==3.0.0                     ✅ Installed
flask-cors==4.0.0                ✅ Installed
gunicorn==21.2.0                 ✅ Installed
```

## 🚀 NPM Scripts Available

### **Development Scripts**
```bash
npm run dev                      # Start frontend development server
npm run python:dev               # Start backend development server
npm run fullstack:start          # Instructions for full stack startup
```

### **Testing Scripts**
```bash
npm run stress-test              # Run system stress tests
npm run stress-test:simple       # Simple Node.js stress test
npm run stress-test:windows      # Windows batch stress test
npm run stress-test:powershell   # PowerShell stress test
npm run stress-test:unix         # Unix shell stress test
npm run fullstack:test           # Browser integration test instructions
```

### **Python Environment Scripts**
```bash
npm run python:setup             # Setup Python virtual environment
npm run python:setup:powershell  # PowerShell venv setup
npm run python:setup:unix        # Unix venv setup
npm run python:activate          # Activate Python environment
npm run python:install           # Install Python dependencies
npm run python:start             # Start Python backend
```

### **Docker Scripts**
```bash
npm run docker:up                # Start Docker production stack
npm run docker:dev               # Start Docker development stack
npm run docker:down              # Stop Docker services
npm run docker:logs              # View Docker logs
npm run docker:build             # Build Docker containers
```

### **Build Scripts**
```bash
npm run build                    # Build frontend for production
npm run preview                  # Preview production build
```

## 📊 System Status

### ✅ **Currently Working**
- Full stack integration (Frontend ↔ Backend)
- Python virtual environment setup
- Unified configuration system
- Docker containerization
- Comprehensive stress testing
- Cross-platform compatibility (Windows/Unix)
- TypeScript type safety
- React component architecture

### 🔄 **Potentially Needs Development**
Based on the missing files analysis:

1. **CSS Styling System** - May need dedicated CSS files
2. **Backend Data Models** - Structured data handling
3. **Static Assets** - Icons, logos, favicon
4. **Unit Testing** - Component and service tests
5. **Database Integration** - If persistent storage is needed
6. **CI/CD Pipeline** - Automated testing and deployment
7. **Error Logging** - Structured logging system
8. **Performance Monitoring** - Production monitoring

## 🎯 Recommended Next Steps

1. **Verify Core Functionality**: Test file upload and model training features
2. **Add Missing Assets**: Create favicon, logo, and other static assets
3. **Implement Unit Tests**: Add Jest/testing-library tests for components
4. **Set up CI/CD**: Configure automated testing and deployment
5. **Add Error Handling**: Implement comprehensive error logging
6. **Performance Optimization**: Add monitoring and optimization
7. **Documentation**: Complete API documentation and user guides

## 🔗 Key Integration Points

- **Frontend-Backend**: `services/config.ts` → `backend/server.py`
- **API Communication**: `services/api.ts` → Flask routes
- **Configuration**: `.env.*` files → `services/config.ts`
- **Docker**: `docker-compose.yml` orchestrates all services
- **Testing**: `tests/` directory contains validation suites
- **Environment**: Virtual environment isolates Python dependencies

This schema represents a well-structured, production-ready full-stack application with comprehensive tooling and documentation.
