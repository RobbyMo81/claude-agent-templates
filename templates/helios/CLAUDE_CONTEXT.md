# Helios Project - Context Documentation

## Project Overview

**Helios** is a full-stack machine learning system for Powerball lottery anomaly detection and prediction analysis. The system combines neural network training, metacognitive AI capabilities, and cross-model analytics to establish baseline random performance and detect patterns in historical lottery data.

**Current Status:** Phase 4 Complete - Production Ready
**Architecture:** Full-stack monolithic with separated frontend/backend services

---

## Technology Stack

### Frontend
- **Framework:** React 19.1.0 + TypeScript 5.7.2
- **Build:** Vite 6.2.0
- **UI:** Material-UI (MUI), Tailwind CSS
- **Charts:** Chart.js 4.4.3, Recharts 3.1.0
- **AI Integration:** Google Generative AI SDK (@google/genai 1.9.0)

### Backend
- **Server:** Flask 3.0.0 + Gunicorn 21.2.0
- **ML Framework:** PyTorch 2.7.1
- **Data:** NumPy, Pandas, Scikit-learn
- **Database:** SQLite (helios_memory.db)
- **Testing:** pytest 8.2.2
- **Runtime:** Python 3.11+

### DevOps
- **Containers:** Docker + docker-compose
- **Web Server:** Nginx (production reverse proxy)
- **Ports:** Frontend (5173/80), Backend (5001)

---

## Project Structure

```
helios/
├── components/              # React components (17 files)
│   ├── App.tsx             # Main application container
│   ├── Sidebar.tsx         # Navigation panel
│   ├── TrainingPanel.tsx   # Model training UI
│   ├── CrossModelAnalytics.tsx  # Phase 4: Multi-model comparison
│   └── [13 other components]
├── services/               # Frontend service layer
│   ├── config.ts          # Configuration management
│   ├── api.ts             # Orchestration layer
│   ├── modelService.ts    # ML backend API client
│   ├── geminiService.ts   # Google Gemini AI integration
│   └── lotteryService.ts  # Lottery data processing
├── backend/               # Python backend
│   ├── server.py         # Flask REST API (20+ endpoints)
│   ├── agent.py          # PowerballNet neural network
│   ├── trainer.py        # Training pipeline
│   ├── memory_store.py   # SQLite persistence layer
│   ├── metacognition.py  # Self-aware AI (Phase 3)
│   ├── decision_engine.py # Autonomous decisions (Phase 3)
│   ├── cross_model_analytics.py # Multi-model analytics (Phase 4)
│   └── models/           # Trained model storage
├── tests/                # Test suites
├── types.ts              # TypeScript interfaces (270+ lines)
├── constants.ts          # Application constants
├── docker-compose.yml    # Container orchestration
└── [config files]
```

---

## Core Architecture

### Frontend Architecture

**Main Component:** `App.tsx`
- Centralized state management (React hooks)
- Manages 6 views: baseline, reflection, stress_report, training_dashboard, metacognitive, cross_model_analytics
- Backend connection monitoring
- Coordinates sidebar navigation

**Service Layer Pattern:**
```typescript
api.ts (Orchestrator)
├─> lotteryService.ts   # CSV parsing, backtest simulation
├─> modelService.ts     # Backend API calls
└─> geminiService.ts    # AI analysis

config.ts → Runtime configuration with Docker support
```

**Key Workflows:**
1. **Baseline Analysis:** CSV Upload → Parse → Backtest → Gemini Analysis → Display
2. **Model Training:** Configure → POST /api/train → Poll Status → Display Progress
3. **Cross-Model Analytics:** Select Models → Fetch History → Compare → Ensemble Recommendations

### Backend Architecture

**Flask Application:** `backend/server.py` (550+ lines)

**Critical Endpoints:**
```
GET  /health                              # Health check
GET  /api/models                          # List models
POST /api/train                           # Start training job
GET  /api/train/status/{job_id}          # Training status
GET  /api/train/history                   # Training history
POST /api/models/{name}/predict           # Generate predictions
GET  /api/analytics/compare               # Cross-model comparison
GET  /api/analytics/ensemble/recommendations  # Ensemble advice
```

**Core Modules:**

1. **agent.py - PowerballNet Neural Network**
   - Embedding layers (white balls: 69, powerball: 26)
   - Multi-head attention (8 heads)
   - Bidirectional LSTM (3 layers)
   - Multiple prediction heads:
     - White ball predictor
     - Powerball predictor
     - Frequency analysis
     - Confidence scoring
   - Positional encoding for temporal awareness

2. **trainer.py - Training Pipeline**
   - `TrainingConfig`: Epochs, learning rate, batch size
   - `DataPreprocessor`: Validation, normalization, feature engineering
   - `ModelTrainer`: Job management, training loop, checkpointing
   - Early stopping, loss computation (MSE)
   - Training journal logging

3. **memory_store.py - SQLite Persistence** (400+ lines)
   - Thread-safe database operations
   - **Tables:**
     - `enhanced_journal` - Training event logs
     - `knowledge_fragments` - Persistent learnings
     - `performance_metrics` - Model metrics
     - `training_sessions` - Job history
     - `model_metadata` - Model information
     - `predictions` - Prediction records
     - `model_versions` - Version tracking

4. **metacognition.py - Self-Aware AI (Phase 3)**
   - Confidence estimation (VERY_LOW to VERY_HIGH)
   - Learning strategies (CONSERVATIVE, BALANCED, AGGRESSIVE, ADAPTIVE, ACTIVE_LEARNING)
   - Knowledge gap identification
   - Performance pattern recognition
   - Adaptive learning rate adjustment

5. **decision_engine.py - Autonomous Decisions (Phase 3)**
   - Parameter optimization
   - Goal prioritization (CRITICAL, HIGH, MEDIUM, LOW)
   - Resource allocation
   - Training strategy adaptation
   - Multi-threaded execution

6. **cross_model_analytics.py - Multi-Model Analytics (Phase 4)**
   - Performance comparison across models
   - Ensemble recommendation algorithms (3 strategies)
   - Trend analysis
   - Performance matrix generation
   - Metrics: Stability score, efficiency score, convergence detection

---

## Key Data Models

### TypeScript Interfaces (types.ts)

```typescript
// Core lottery data
interface HistoricalDraw {
  draw_date: string;
  wb1, wb2, wb3, wb4, wb5: number;  // White balls 1-69
  pb: number;                        // Powerball 1-26
}

// Training configuration
interface TrainingConfig {
  model_name: string;
  epochs: number;
  learning_rate: number;
  batch_size?: number;
  validation_split?: number;
}

// Training status
interface TrainingStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  current_epoch: number;
  total_epochs: number;
  current_loss: number | null;
  best_loss: number | null;
  training_progress: TrainingProgressEntry[];
}

// Model information
interface ModelInfo {
  name: string;
  architecture: string;
  version: string;
  created_at: string;
  training_completed: boolean;
  total_epochs: number;
  best_loss: number;
}

// Cross-model analytics (Phase 4)
interface CrossModelComparison {
  compared_models: string[];
  performance_ranking: Array<[string, number]>;
  efficiency_ranking: Array<[string, number]>;
  ensemble_potential: number;
}

interface EnsembleRecommendation {
  recommended_models: string[];
  weights: number[];
  expected_performance: number;
  confidence_score: number;
  reasoning: string;
  risk_assessment: string;
}
```

---

## Configuration Management

### Environment Files
```
.env.development   → Dev settings (localhost:5173, localhost:5001)
.env.production    → Production (Docker network, container names)
.env.example       → Template

Loaded via: services/config.ts → Vite build-time substitution
```

### Port Configuration
```
Development:
  Frontend: http://localhost:5173  (Vite dev server)
  Backend:  http://localhost:5001  (Flask)

Production (Docker):
  Frontend: http://localhost:80    (Nginx reverse proxy)
  Backend:  http://backend:5001    (Docker network internal)
```

---

## Database Schema

**File:** `helios_memory.db` (SQLite)

**Primary Tables:**
```sql
enhanced_journal         # Training events, timestamps, status
knowledge_fragments      # Persistent model learnings
performance_metrics      # Loss, accuracy, precision, recall
training_sessions        # Job history, configuration
model_metadata          # Model info, versions, creation dates
predictions             # Generated predictions with confidence
model_versions          # Version tracking
```

**Relationships:**
- `training_sessions.job_id` → `enhanced_journal.job_id`
- `model_metadata.name` → `predictions.model_name`
- Foreign keys for data integrity

---

## Testing Infrastructure

### Test Files
```
tests/
├── stressTest.ts              # TypeScript stress test suite
├── runStressTest.ts           # Test runner
├── fullStackTest.mjs          # Full stack integration
└── fullStackIntegrationTest.js # Browser-based tests
```

### Stress Test Scenarios
- `SLOW_BACKTEST` - Slow processing simulation (3s delay)
- `AI_FAILURE` - AI service failure testing
- `AI_EMPTY_RESPONSE` - Empty API response testing
- `CSV_PARSE_ERROR` - Invalid CSV handling
- `CSV_EMPTY_FILE` - Empty file handling

### Test Coverage
- Backend connectivity
- CSV parsing (valid/invalid)
- Backtest execution
- Model training pipeline
- All API CRUD operations
- Error handling scenarios
- Gemini AI integration

---

## Development Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- Docker (optional, for production deployment)

### Quick Start

**Backend:**
```bash
# Windows
setup-venv.bat
venv\Scripts\activate
pip install -r requirements.txt
python backend/server.py

# Unix
./setup-venv.sh
source venv/bin/activate
pip install -r requirements.txt
python backend/server.py
```

**Frontend:**
```bash
npm install
npm run dev
```

**Docker (Production):**
```bash
docker-compose up --build
# With dev frontend:
docker-compose --profile dev up --build
```

---

## Design Patterns Used

1. **Service Layer Pattern** - Frontend services encapsulate domain logic
2. **Repository Pattern** - MemoryStore acts as data repository
3. **Factory Pattern** - Model instantiation with configurable parameters
4. **Strategy Pattern** - Multiple learning strategies, ensemble methods
5. **Observer Pattern** - Training status polling, event logging
6. **Adapter Pattern** - Frontend ↔ Backend configuration compatibility

---

## Feature Phases

### Phase 1: Baseline Analysis ✅
- CSV upload and validation
- Backtest simulation (random lottery agent)
- Prize tier evaluation
- Google Gemini AI statistical analysis
- Results visualization

### Phase 2: Model Training ✅
- PowerballNet neural network implementation
- Real-time training progress tracking
- Training journal with loss history
- Model persistence and checkpointing
- Job status polling API

### Phase 3: Metacognition & Autonomy ✅
- Self-aware model assessment
- Confidence scoring and knowledge gaps
- Adaptive learning strategy selection
- Autonomous parameter optimization
- Goal-based decision making

### Phase 4: Cross-Model Analytics ✅
- Multi-model performance comparison
- Ensemble recommendation engine
- Historical trend analysis
- Performance matrix visualization
- 5-tab analytics dashboard

---

## Important Files Reference

### Frontend Entry Points
- `index.html` - HTML root
- `index.tsx` - React root
- `App.tsx:1` - Main application component
- `services/api.ts:1` - API orchestration layer
- `vite.config.ts:1` - Build configuration

### Backend Entry Points
- `backend/server.py:1` - Flask application
- `backend/agent.py:1` - PowerballNet neural network
- `backend/trainer.py:1` - Training pipeline
- `backend/memory_store.py:1` - Database layer

### Configuration
- `services/config.ts:1` - Frontend configuration
- `.env.development` - Dev environment variables
- `.env.production` - Production environment variables
- `docker-compose.yml:1` - Container orchestration

### Documentation
- `IMPLEMENTATION_PLAN_v0.9.0.md` - Overall implementation plan
- `PHASE_4_IMPLEMENTATION_COMPLETE.md` - Latest phase documentation
- `COMPREHENSIVE_SYSTEM_ARCHITECTURE_REPORT.md` - Detailed architecture
- `FULL_STACK_INTEGRATION.md` - Integration guide
- `DOCKER_SETUP.md` - Docker deployment guide

---

## Common Development Tasks

### Add a New API Endpoint
1. Define route in `backend/server.py`
2. Add TypeScript interface in `types.ts`
3. Create service method in appropriate `services/*.ts` file
4. Update component to call service method

### Add a New UI Component
1. Create component in `components/`
2. Import in `App.tsx`
3. Add to sidebar navigation if needed
4. Update state management in `App.tsx` if required

### Modify Neural Network Architecture
1. Update `PowerballNet` class in `backend/agent.py`
2. Update training config in `backend/trainer.py`
3. Test with new training job
4. Update model metadata schema if needed

### Add New Analytics Feature
1. Add method to `backend/cross_model_analytics.py`
2. Create API endpoint in `backend/server.py`
3. Add TypeScript interface in `types.ts`
4. Create or update analytics component
5. Add to CrossModelAnalytics dashboard

---

## Troubleshooting

### Common Issues

**Backend Connection Failed:**
- Check Flask server is running on port 5001
- Verify environment variables in `.env.development`
- Check `services/config.ts` for correct API URL

**Docker Services Not Starting:**
- Check logs: `docker-compose logs backend`
- Verify port 5001 and 80 are not in use
- Ensure `.env.production` is configured

**Training Job Stuck:**
- Check training status: `GET /api/train/status/{job_id}`
- View enhanced_journal table in helios_memory.db
- Check backend logs for errors

**Model Prediction Errors:**
- Verify model is loaded: `GET /api/models/{name}/info`
- Check model file exists in `backend/models/`
- Ensure training completed successfully

---

## Production Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose up -d --build

# Check service health
docker-compose ps
curl http://localhost/health

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

### Health Checks
- Backend: `http://localhost:5001/health`
- Frontend: `http://localhost/` (Docker) or `http://localhost:5173` (Dev)

### Monitoring
- Training logs: `enhanced_journal` table
- Application logs: `logs/` directory or Docker logs
- Model checkpoints: `backend/models/` directory

---

## Security Considerations

1. **API Key Management:**
   - Gemini API key injected at build time via Vite
   - Never commit `.env` files to version control
   - Use `.env.example` as template

2. **CORS Configuration:**
   - Configured in `backend/server.py`
   - Allows localhost origins in development
   - Restrict in production deployment

3. **Input Validation:**
   - CSV data validated in `lotteryService.ts`
   - Training config validated in `backend/trainer.py`
   - Model names sanitized before file operations

4. **Database Security:**
   - SQLite file permissions should be restricted
   - No SQL injection risk (parameterized queries)
   - Thread-safe operations in `memory_store.py`

---

## Performance Optimization

### Current Optimizations
- Multi-head attention for parallel processing
- Batch training for efficient GPU utilization
- Early stopping to prevent overtraining
- Model checkpointing to save best models
- Connection pooling in database operations

### Scalability Considerations
- SQLite adequate for current data volume
- Consider PostgreSQL for production scale
- Docker allows horizontal scaling of frontend
- Backend can be load-balanced behind Nginx

---

## Known Limitations

1. **Single Backend Instance:** No built-in support for multiple training jobs simultaneously
2. **Database:** SQLite not ideal for high-concurrency scenarios
3. **Monitoring:** Basic health checks only, no metrics collection
4. **CI/CD:** No automated pipeline configured
5. **Error Recovery:** Manual intervention required for failed training jobs

---

## Future Enhancements

### Potential Improvements
- Add unit test coverage (Jest for frontend, pytest for backend)
- Implement OpenAPI/Swagger documentation
- Add metrics collection (Prometheus/Grafana)
- Setup CI/CD pipeline (GitHub Actions)
- Migrate to PostgreSQL for production
- Add database migration tooling (Alembic)
- Implement WebSocket for real-time training updates
- Add authentication and user management

---

## Resources

### Internal Documentation
- `/docs` - Additional documentation (if exists)
- `README.md` - Project README (if exists)
- Phase completion reports (PHASE_1_COMPLETE.md through PHASE_4_IMPLEMENTATION_COMPLETE.md)

### External References
- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [React Documentation](https://react.dev/)
- [Vite Documentation](https://vite.dev/)
- [Google Generative AI](https://ai.google.dev/)

---

## Contact & Support

For issues and feature requests, refer to the project documentation in the repository root or consult the comprehensive architecture reports.

**Last Updated:** Phase 4 Implementation Complete
**Document Version:** 1.0
**Architecture Status:** Production Ready
