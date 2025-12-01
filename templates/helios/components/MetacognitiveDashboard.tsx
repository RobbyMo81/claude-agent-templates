import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Alert,
  Tab,
  Tabs,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Tooltip,
  IconButton,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Psychology,
  AutoAwesome,
  TrendingUp,
  Settings,
  Memory,
  Assessment,
  SmartToy,
  Refresh,
} from '@mui/icons-material';
import { XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, BarChart, Bar } from 'recharts';
import { getApiEndpoint } from '../services/config';

interface MetacognitiveAssessment {
  confidence_score: number;
  predicted_performance: number;
  uncertainty_estimate: number;
  knowledge_gaps: string[];
  recommended_strategy: string;
  assessment_timestamp: string;
  context: Record<string, any>;
}

interface PerformancePattern {
  pattern_type: string;
  pattern_strength: number;
  conditions: Record<string, any>;
  impact_score: number;
  frequency: number;
  last_observed: string;
}

interface Decision {
  decision_id: string;
  decision_type: string;
  priority: number;
  rationale: string;
  expected_impact: number;
  confidence: number;
  status: string;
  created_at: string;
  result?: Record<string, any>;
}

interface Goal {
  goal_id: string;
  name: string;
  progress: number;
  priority: number;
  active: boolean;
  target_metric: string;
  target_value: number;
  current_value: number;
}

interface MetacognitiveDashboardProps {
  modelName: string;
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`metacognitive-tabpanel-${index}`}
      aria-labelledby={`metacognitive-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const MetacognitiveDashboard: React.FC<MetacognitiveDashboardProps> = ({ modelName }) => {
  const [tabValue, setTabValue] = useState(0);
  const [assessment, setAssessment] = useState<MetacognitiveAssessment | null>(null);
  const [patterns, setPatterns] = useState<PerformancePattern[]>([]);
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [goals, setGoals] = useState<Goal[]>([]);
  const [autonomousMode, setAutonomousMode] = useState(false);
  const [systemStatus, setSystemStatus] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch metacognitive data
  const fetchMetacognitiveData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch current assessment
      const assessmentResponse = await fetch(getApiEndpoint(`/api/metacognitive/assessment?model_name=${modelName}`));
      if (assessmentResponse.ok) {
        const assessmentData = await assessmentResponse.json();
        setAssessment(assessmentData);
      }

      // Fetch performance patterns
      const patternsResponse = await fetch(getApiEndpoint(`/api/metacognitive/patterns?model_name=${modelName}`));
      if (patternsResponse.ok) {
        const patternsData = await patternsResponse.json();
        setPatterns(patternsData);
      }

      // Fetch recent decisions
      const decisionsResponse = await fetch(getApiEndpoint(`/api/decisions/history?days=7`));
      if (decisionsResponse.ok) {
        const decisionsData = await decisionsResponse.json();
        setDecisions(decisionsData);
      }

      // Fetch system status
      const statusResponse = await fetch(getApiEndpoint('/api/decisions/status'));
      if (statusResponse.ok) {
        const statusData = await statusResponse.json();
        setSystemStatus(statusData);
        setAutonomousMode(statusData.autonomous_mode);
        setGoals(statusData.goal_status?.goals || []);
      }

    } catch (err) {
      console.error('Error fetching metacognitive data:', err);
      setError('Failed to load metacognitive data');
    } finally {
      setLoading(false);
    }
  };

  // Toggle autonomous mode
  const toggleAutonomousMode = async () => {
    try {
      const action = autonomousMode ? 'stop' : 'start';
      const response = await fetch(getApiEndpoint(`/api/decisions/autonomous/${action}`), {
        method: 'POST',
      });

      if (response.ok) {
        setAutonomousMode(!autonomousMode);
        fetchMetacognitiveData(); // Refresh data
      }
    } catch (err) {
      console.error('Error toggling autonomous mode:', err);
    }
  };

  useEffect(() => {
    fetchMetacognitiveData();

    // Set up polling for real-time updates
    const interval = setInterval(fetchMetacognitiveData, 10000); // Every 10 seconds

    return () => clearInterval(interval);
  }, [modelName]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return '#4caf50';
    if (confidence >= 0.6) return '#ff9800';
    if (confidence >= 0.4) return '#f44336';
    return '#9e9e9e';
  };

  const getStrategyColor = (strategy: string) => {
    switch (strategy.toLowerCase()) {
      case 'conservative': return '#2196f3';
      case 'balanced': return '#4caf50';
      case 'aggressive': return '#f44336';
      case 'adaptive': return '#9c27b0';
      default: return '#757575';
    }
  };

  const getPriorityColor = (priority: number) => {
    switch (priority) {
      case 1: return '#f44336';
      case 2: return '#ff9800';
      case 3: return '#ffc107';
      case 4: return '#4caf50';
      default: return '#757575';
    }
  };

  // Prepare radar chart data for assessment
  const radarData = assessment ? [
    {
      subject: 'Confidence',
      A: assessment.confidence_score * 100,
      fullMark: 100,
    },
    {
      subject: 'Predicted Performance',
      A: assessment.predicted_performance * 100,
      fullMark: 100,
    },
    {
      subject: 'Certainty',
      A: (1 - assessment.uncertainty_estimate) * 100,
      fullMark: 100,
    },
    {
      subject: 'Knowledge Coverage',
      A: Math.max(0, 100 - (assessment.knowledge_gaps.length * 20)),
      fullMark: 100,
    },
  ] : [];

  // Prepare pattern strength chart data
  const patternChartData = patterns.map(pattern => ({
    name: pattern.pattern_type.replace(/_/g, ' '),
    strength: pattern.pattern_strength * 100,
    impact: pattern.impact_score * 100,
    frequency: pattern.frequency,
  }));

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <Typography>Loading metacognitive data...</Typography>
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ margin: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Psychology color="primary" sx={{ fontSize: 32 }} />
          <Typography variant="h4" component="h1">
            Metacognitive Dashboard
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <FormControlLabel
            control={
              <Switch
                checked={autonomousMode}
                onChange={toggleAutonomousMode}
                color="primary"
              />
            }
            label="Autonomous Mode"
          />
          <IconButton onClick={fetchMetacognitiveData} title="Refresh Data">
            <Refresh />
          </IconButton>
        </Box>
      </Box>

      {/* Tabs */}
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="metacognitive dashboard tabs">
          <Tab label="Self-Assessment" icon={<Assessment />} />
          <Tab label="Performance Patterns" icon={<TrendingUp />} />
          <Tab label="Autonomous Decisions" icon={<AutoAwesome />} />
          <Tab label="Goals & Strategy" icon={<SmartToy />} />
        </Tabs>
      </Box>

      {/* Self-Assessment Tab */}
      <TabPanel value={tabValue} index={0}>
        {assessment ? (
          <Grid container spacing={3}>
            {/* Current Assessment Overview */}
            <Grid size={{ xs: 12, md: 6 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Assessment sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Current Self-Assessment
                  </Typography>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Confidence Score
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={assessment.confidence_score * 100}
                      sx={{
                        height: 8,
                        borderRadius: 4,
                        backgroundColor: '#e0e0e0',
                        '& .MuiLinearProgress-bar': {
                          backgroundColor: getConfidenceColor(assessment.confidence_score)
                        }
                      }}
                    />
                    <Typography variant="h6" sx={{ mt: 1 }}>
                      {(assessment.confidence_score * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Predicted Performance
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={assessment.predicted_performance * 100}
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="h6" sx={{ mt: 1 }}>
                      {(assessment.predicted_performance * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Typography variant="body2" color="text.secondary">
                      Uncertainty Level
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={assessment.uncertainty_estimate * 100}
                      color="warning"
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="h6" sx={{ mt: 1 }}>
                      {(assessment.uncertainty_estimate * 100).toFixed(1)}%
                    </Typography>
                  </Box>

                  <Box>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Recommended Strategy
                    </Typography>
                    <Chip
                      label={assessment.recommended_strategy}
                      sx={{
                        backgroundColor: getStrategyColor(assessment.recommended_strategy),
                        color: 'white',
                        fontWeight: 'bold'
                      }}
                    />
                  </Box>
                </CardContent>
              </Card>
            </Grid>

            {/* Radar Chart */}
            <Grid size={{ xs: 12, md: 6 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    Assessment Radar
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <RadarChart data={radarData}>
                      <PolarGrid />
                      <PolarAngleAxis dataKey="subject" />
                      <PolarRadiusAxis angle={90} domain={[0, 100]} />
                      <Radar
                        name="Current State"
                        dataKey="A"
                        stroke="#8884d8"
                        fill="#8884d8"
                        fillOpacity={0.3}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </Grid>

            {/* Knowledge Gaps */}
            <Grid size={{ xs: 12 }}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    <Memory sx={{ mr: 1, verticalAlign: 'middle' }} />
                    Identified Knowledge Gaps
                  </Typography>
                  {assessment.knowledge_gaps.length > 0 ? (
                    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                      {assessment.knowledge_gaps.map((gap, index) => (
                        <Chip
                          key={index}
                          label={gap}
                          color="warning"
                          variant="outlined"
                          size="small"
                        />
                      ))}
                    </Box>
                  ) : (
                    <Typography color="text.secondary">
                      No significant knowledge gaps identified
                    </Typography>
                  )}
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        ) : (
          <Alert severity="info">
            No self-assessment data available. Assessment will be generated during training.
          </Alert>
        )}
      </TabPanel>

      {/* Performance Patterns Tab */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          {/* Patterns Chart */}
          <Grid size={{ xs: 12, md: 8 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Performance Pattern Analysis
                </Typography>
                {patternChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={patternChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Bar dataKey="strength" fill="#8884d8" name="Pattern Strength %" />
                      <Bar dataKey="impact" fill="#82ca9d" name="Impact Score %" />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <Typography color="text.secondary">
                    No performance patterns detected yet
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Pattern Details */}
          <Grid size={{ xs: 12, md: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Pattern Summary
                </Typography>
                {patterns.length > 0 ? (
                  <Box>
                    {patterns.slice(0, 5).map((pattern, index) => (
                      <Box key={index} sx={{ mb: 2, p: 1, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Typography variant="body2" fontWeight="bold">
                          {pattern.pattern_type.replace(/_/g, ' ')}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          Strength: {(pattern.pattern_strength * 100).toFixed(1)}%
                        </Typography>
                        <br />
                        <Typography variant="caption" color="text.secondary">
                          Frequency: {pattern.frequency} occurrences
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Typography color="text.secondary">
                    Pattern analysis will appear after sufficient training data
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Autonomous Decisions Tab */}
      <TabPanel value={tabValue} index={2}>
        <Grid container spacing={3}>
          {/* System Status */}
          <Grid size={{ xs: 12, md: 4 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <Settings sx={{ mr: 1, verticalAlign: 'middle' }} />
                  System Status
                </Typography>
                {systemStatus && (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Autonomous Mode:</Typography>
                      <Chip
                        label={systemStatus.autonomous_mode ? 'Active' : 'Inactive'}
                        color={systemStatus.autonomous_mode ? 'success' : 'default'}
                        size="small"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Pending Decisions:</Typography>
                      <Typography variant="body2">{systemStatus.pending_decisions}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2">Recent Decisions (24h):</Typography>
                      <Typography variant="body2">{systemStatus.recent_decisions}</Typography>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Decisions */}
          <Grid size={{ xs: 12, md: 8 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Recent Autonomous Decisions
                </Typography>
                <TableContainer component={Paper} variant="outlined">
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell>Type</TableCell>
                        <TableCell>Priority</TableCell>
                        <TableCell>Rationale</TableCell>
                        <TableCell>Status</TableCell>
                        <TableCell>Impact</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {decisions.slice(0, 10).map((decision) => (
                        <TableRow key={decision.decision_id}>
                          <TableCell>
                            <Typography variant="body2">
                              {decision.decision_type.replace(/_/g, ' ')}
                            </Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={`P${decision.priority}`}
                              size="small"
                              sx={{
                                backgroundColor: getPriorityColor(decision.priority),
                                color: 'white'
                              }}
                            />
                          </TableCell>
                          <TableCell>
                            <Tooltip title={decision.rationale}>
                              <Typography variant="body2" noWrap sx={{ maxWidth: 200 }}>
                                {decision.rationale}
                              </Typography>
                            </Tooltip>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={decision.status}
                              size="small"
                              color={decision.status === 'completed' ? 'success' :
                                     decision.status === 'failed' ? 'error' : 'default'}
                            />
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2">
                              {(decision.expected_impact * 100).toFixed(1)}%
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>

      {/* Goals & Strategy Tab */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          {/* Active Goals */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  <SmartToy sx={{ mr: 1, verticalAlign: 'middle' }} />
                  Active Goals
                </Typography>
                {goals.filter(g => g.active).length > 0 ? (
                  <Box>
                    {goals.filter(g => g.active).map((goal) => (
                      <Box key={goal.goal_id} sx={{ mb: 2, p: 2, border: '1px solid #e0e0e0', borderRadius: 1 }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                          <Typography variant="body1" fontWeight="bold">
                            {goal.name}
                          </Typography>
                          <Chip
                            label={`P${goal.priority}`}
                            size="small"
                            sx={{
                              backgroundColor: getPriorityColor(goal.priority),
                              color: 'white'
                            }}
                          />
                        </Box>
                        <Box sx={{ mb: 1 }}>
                          <Typography variant="body2" color="text.secondary">
                            Progress: {(goal.progress * 100).toFixed(1)}%
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={goal.progress * 100}
                            sx={{ height: 6, borderRadius: 3, mt: 0.5 }}
                          />
                        </Box>
                        <Typography variant="caption" color="text.secondary">
                          {goal.target_metric}: {goal.current_value.toFixed(3)} / {goal.target_value.toFixed(3)}
                        </Typography>
                      </Box>
                    ))}
                  </Box>
                ) : (
                  <Typography color="text.secondary">
                    No active goals defined
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Goal Statistics */}
          <Grid size={{ xs: 12, md: 6 }}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Goal Statistics
                </Typography>
                {systemStatus?.goal_status && (
                  <Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">Active Goals:</Typography>
                      <Typography variant="h6">{systemStatus.goal_status.active_goals}</Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                      <Typography variant="body2">Completed Goals:</Typography>
                      <Typography variant="h6">{systemStatus.goal_status.completed_goals}</Typography>
                    </Box>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="body2" color="text.secondary">
                        Overall Progress
                      </Typography>
                      <LinearProgress
                        variant="determinate"
                        value={systemStatus.goal_status.total_progress * 100}
                        sx={{ height: 8, borderRadius: 4, mt: 1 }}
                      />
                      <Typography variant="h6" sx={{ mt: 1 }}>
                        {(systemStatus.goal_status.total_progress * 100).toFixed(1)}%
                      </Typography>
                    </Box>
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </TabPanel>
    </Box>
  );
};

export default MetacognitiveDashboard;
