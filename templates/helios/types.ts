export type PrizeTier =
  | "Grand Prize"
  | "$1 Million"
  | "$50,000"
  | "$100 (4+0)"
  | "$100 (3+1)"
  | "$7 (3+0)"
  | "$7 (2+1)"
  | "$4 (1+1)"
  | "$4 (0+1)"
  | "No Win";

export type Ticket = [number[], number];

export interface HistoricalDraw {
  draw_date: string;
  wb1: number;
  wb2: number;
  wb3: number;
  wb4: number;
  wb5: number;
  pb: number;
}

export type BacktestResults = Record<PrizeTier, number>;

export type StressTestType =
  | 'SLOW_BACKTEST'
  | 'AI_FAILURE'
  | 'AI_EMPTY_RESPONSE'
  | 'CSV_PARSE_ERROR'
  | 'CSV_EMPTY_FILE';

export interface TrainingConfig {
  model_name: string;
  epochs: number;
  learning_rate: number;
}

export interface ModelJournal {
  model_name: string;
  epochs: number;
  learning_rate: number;
  training_duration_seconds: number;
  loss_history: number[];
}

// New types for the stress testing framework
export type BackendStatus = 'pending' | 'connected' | 'disconnected';

export type ViewType = 'baseline' | 'reflection' | 'stress_report' | 'training_dashboard' | 'metacognitive' | 'cross_model_analytics';

export interface TestReportEntry {
  id: number;
  timestamp: string;
  testName: string;
  status: 'PASS' | 'FAIL';
  durationMs: number;
  details: string;
}

// Model Management Types
export interface ModelInfo {
  name: string;
  architecture: string;
  version: string;
  created_at: string;
  training_completed: boolean;
  total_epochs: number;
  best_loss: number;
}

export interface ModelMetadata {
  created_at: string;
  version: string;
  architecture: string;
  sequence_length: number;
  device: string;
  training_completed: boolean;
  total_epochs: number;
  best_loss: number;
}

export interface ModelDetails {
  name: string;
  architecture: string;
  version: string;
  created_at: string;
  updated_at: string;
  metadata: ModelMetadata;
  recent_predictions: PredictionRecord[];
  file_path: string;
}

export interface TrainingJobResponse {
  status: 'started' | 'completed' | 'failed';
  message: string;
  job_id: string;
  estimated_duration?: string;
}

export interface TrainingLogEntry {
  timestamp: string;
  epoch: number;
  loss: number;
  accuracy?: number;
}

export interface PredictionData {
  white_balls: {
    numbers: number[][];
    probabilities: number[][];
  };
  powerball: {
    numbers: number[][];
    probabilities: number[][];
  };
  confidence: number[];
  model_features: number[][];
}

export interface PredictionRecord {
  id: number;
  model_name: string;
  prediction_data: PredictionData;
  confidence: number;
  prediction_timestamp: string;
  draw_date?: string;
  actual_outcome?: any;
  is_correct?: boolean;
}

export interface ModelLoadResponse {
  status: 'success' | 'error';
  message: string;
  model_info?: {
    metadata: ModelMetadata;
    training_history: TrainingLogEntry[];
    model_parameters: number;
    device: string;
    memory_usage: number;
  };
}

// Enhanced Training Config
export interface ExtendedTrainingConfig {
  modelName: string;
  epochs: number;
  learningRate: number;
  batchSize?: number;
  sequenceLength?: number;
  dataSource?: string;
}

// Model Management State
export interface ModelManagementState {
  availableModels: ModelInfo[];
  selectedModel: string | null;
  loadingModel: boolean;
  modelDetails: ModelDetails | null;
  lastPrediction: PredictionData | null;
  trainingInProgress: boolean;
  currentTrainingJob: string | null;
}

export interface TrainingProgressEntry {
  epoch: number;
  loss: number;
  timestamp?: string;
  accuracy?: number;
}

export interface TrainingStatus {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'stopped';
  current_epoch: number;
  total_epochs: number;
  current_loss: number | null;
  best_loss: number | null;
  start_time: string | null;
  end_time: string | null;
  error: string | null;
  training_progress: TrainingProgressEntry[];
}

export interface TrainingHistoryEntry {
  job_id: string;
  model_name: string;
  status: 'completed' | 'failed' | 'stopped';
  start_time: string;
  end_time: string | null;
  total_epochs: number;
  final_loss: number | null;
  config: TrainingConfig;
}

// =============================================================================
// PHASE 4: CROSS-MODEL ANALYTICS TYPES
// =============================================================================

export interface ModelPerformanceMetrics {
  model_name: string;
  training_time: number;
  final_loss: number;
  best_loss: number;
  total_epochs: number;
  convergence_epoch: number | null;
  stability_score: number;
  efficiency_score: number;
  last_updated: string;
}

export interface CrossModelComparison {
  compared_models: string[];
  performance_ranking: Array<[string, number]>;
  efficiency_ranking: Array<[string, number]>;
  convergence_analysis: Record<string, number>;
  recommendation_score: Record<string, number>;
  ensemble_potential: number;
  analysis_timestamp: string;
}

export interface EnsembleRecommendation {
  recommended_models: string[];
  weights: number[];
  expected_performance: number;
  confidence_score: number;
  reasoning: string;
  risk_assessment: string;
}

export interface PerformanceMatrix {
  models: string[];
  metrics: string[];
  data: (number | null)[][];
  rankings: Record<string, Array<[string, number]>>;
  correlations: Record<string, any>;
  generated_at: string;
}

export interface TrendAnalysis {
  time_period: string;
  active_models: number;
  model_trends: Record<string, ModelTrendData>;
  overall_trends: OverallTrends;
  insights: string[];
}

export interface ModelTrendData {
  trend_direction: 'improving' | 'declining' | 'stable' | 'insufficient_data' | 'no_valid_data';
  improvement_rate: number;
  consistency: number;
  data_points?: number;
}

export interface OverallTrends {
  trend_distribution: Record<string, number>;
  average_improvement_rate: number;
  average_consistency: number;
  total_models_analyzed: number;
}

export interface AnalyticsSummary {
  active_models: string[];
  total_models: number;
  time_period: string;
  last_updated: string;
}
