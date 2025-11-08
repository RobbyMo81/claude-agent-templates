import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Tabs,
  Tab,
  Button,
  Chip,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Divider
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Timeline,
  Compare,
  GroupWork,
  Analytics,
  Speed,
  Star,
  Warning
} from '@mui/icons-material';
import { getApiEndpoint } from '../services/config';
import {
  CrossModelComparison,
  EnsembleRecommendation,
  PerformanceMatrix,
  TrendAnalysis,
  AnalyticsSummary
} from '../types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

const CrossModelAnalytics: React.FC = () => {
  const [tabValue, setTabValue] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Data states
  const [analyticsSummary, setAnalyticsSummary] = useState<AnalyticsSummary | null>(null);
  const [modelComparison, setModelComparison] = useState<CrossModelComparison | null>(null);
  const [ensembleRecommendations, setEnsembleRecommendations] = useState<EnsembleRecommendation[]>([]);
  const [performanceMatrix, setPerformanceMatrix] = useState<PerformanceMatrix | null>(null);
  const [trendAnalysis, setTrendAnalysis] = useState<TrendAnalysis | null>(null);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);

  // UI states
  const [daysBack, setDaysBack] = useState(30);
  const [comparisonType, setComparisonType] = useState('comprehensive');

  useEffect(() => {
    loadAnalyticsSummary();
  }, [daysBack]);

  const loadAnalyticsSummary = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(getApiEndpoint(`/api/analytics/models?days=${daysBack}`));
      if (!response.ok) throw new Error('Failed to load analytics summary');

      const data = await response.json();
      setAnalyticsSummary(data);

      // Auto-select first few models for comparison
      if (data.active_models.length > 0) {
        setSelectedModels(data.active_models.slice(0, Math.min(3, data.active_models.length)));
      }

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const loadModelComparison = async () => {
    if (selectedModels.length < 2) {
      setError('At least 2 models required for comparison');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/analytics/compare', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          models: selectedModels,
          type: comparisonType
        })
      });

      if (!response.ok) throw new Error('Failed to compare models');

      const data = await response.json();
      setModelComparison(data);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const loadEnsembleRecommendations = async () => {
    if (selectedModels.length < 2) {
      setError('At least 2 models required for ensemble recommendations');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/analytics/ensemble/recommendations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          models: selectedModels,
          target_metric: 'loss'
        })
      });

      if (!response.ok) throw new Error('Failed to generate ensemble recommendations');

      const data = await response.json();
      setEnsembleRecommendations(data.recommendations || []);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const loadPerformanceMatrix = async () => {
    if (selectedModels.length < 2) {
      setError('At least 2 models required for performance matrix');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await fetch('/api/analytics/performance-matrix', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ models: selectedModels })
      });

      if (!response.ok) throw new Error('Failed to generate performance matrix');

      const data = await response.json();
      setPerformanceMatrix(data);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const loadTrendAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(getApiEndpoint(`/api/analytics/trends?days=${daysBack}`));
      if (!response.ok) throw new Error('Failed to load trend analysis');

      const data = await response.json();
      setTrendAnalysis(data);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
    setError(null);

    // Load data for the selected tab
    switch (newValue) {
      case 1:
        if (selectedModels.length >= 2) {
          loadModelComparison();
        }
        break;
      case 2:
        if (selectedModels.length >= 2) {
          loadEnsembleRecommendations();
        }
        break;
      case 3:
        if (selectedModels.length >= 2) {
          loadPerformanceMatrix();
        }
        break;
      case 4:
        loadTrendAnalysis();
        break;
    }
  };

  const formatScore = (score: number): string => {
    return (score * 100).toFixed(1) + '%';
  };

  const formatLoss = (loss: number): string => {
    if (loss === Infinity || loss > 1000) return 'N/A';
    return loss.toFixed(4);
  };

  const getTrendIcon = (direction: string) => {
    switch (direction) {
      case 'improving':
        return <TrendingUp color="success" />;
      case 'declining':
        return <TrendingDown color="error" />;
      case 'stable':
        return <Timeline color="info" />;
      default:
        return <Warning color="warning" />;
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk.toLowerCase()) {
      case 'very low risk - consistent performers':
      case 'low risk - proven performers':
        return 'success';
      case 'medium risk - experimental combination':
        return 'warning';
      default:
        return 'error';
    }
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Header */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Box display="flex" alignItems="center" mb={2}>
            <Analytics sx={{ mr: 2, fontSize: 32, color: 'primary.main' }} />
            <Typography variant="h4" component="h1">
              Cross-Model Analytics
            </Typography>
          </Box>
          <Typography variant="body1" color="text.secondary">
            Advanced analytics and comparison across multiple models with ensemble recommendations
          </Typography>

          {/* Controls */}
          <Grid container spacing={2} sx={{ mt: 2 }}>
            <Grid size={{ xs: 12, md: 3 }}>
              <Typography variant="subtitle2" gutterBottom>
                Analysis Period
              </Typography>
              <Slider
                value={daysBack}
                onChange={(_: Event, value: number | number[]) => setDaysBack(value as number)}
                min={7}
                max={180}
                marks={[
                  { value: 7, label: '7d' },
                  { value: 30, label: '30d' },
                  { value: 90, label: '90d' },
                  { value: 180, label: '180d' }
                ]}
                valueLabelDisplay="auto"
                valueLabelFormat={(value: number) => `${value} days`}
              />
            </Grid>

            <Grid size={{ xs: 12, md: 4 }}>
              <FormControl fullWidth>
                <InputLabel>Comparison Type</InputLabel>
                <Select
                  value={comparisonType}
                  onChange={(e: any) => setComparisonType(e.target.value)}
                  label="Comparison Type"
                >
                  <MenuItem value="comprehensive">Comprehensive</MenuItem>
                  <MenuItem value="performance">Performance Only</MenuItem>
                  <MenuItem value="efficiency">Efficiency Only</MenuItem>
                </Select>
              </FormControl>
            </Grid>

            <Grid size={{ xs: 12, md: 5 }}>
              {analyticsSummary && (
                <Box>
                  <Typography variant="subtitle2" gutterBottom>
                    Available Models ({analyticsSummary.total_models})
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={1}>
                    {analyticsSummary.active_models.map((model) => (
                      <Chip
                        key={model}
                        label={model}
                        color={selectedModels.includes(model) ? 'primary' : 'default'}
                        onClick={() => {
                          if (selectedModels.includes(model)) {
                            setSelectedModels(selectedModels.filter(m => m !== model));
                          } else {
                            setSelectedModels([...selectedModels, model]);
                          }
                        }}
                        variant={selectedModels.includes(model) ? 'filled' : 'outlined'}
                      />
                    ))}
                  </Box>
                </Box>
              )}
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Error Display */}
      {error && (
        <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Main Analytics Tabs */}
      <Card>
        <Tabs value={tabValue} onChange={handleTabChange} variant="scrollable" scrollButtons="auto">
          <Tab icon={<Analytics />} label="Overview" />
          <Tab icon={<Compare />} label="Model Comparison" />
          <Tab icon={<GroupWork />} label="Ensemble Recommendations" />
          <Tab icon={<Speed />} label="Performance Matrix" />
          <Tab icon={<Timeline />} label="Trend Analysis" />
        </Tabs>

        {/* Overview Tab */}
        <TabPanel value={tabValue} index={0}>
          {loading ? (
            <Box display="flex" justifyContent="center" p={4}>
              <CircularProgress />
            </Box>
          ) : analyticsSummary ? (
            <Grid container spacing={3}>
              <Grid size={{ xs: 12, md: 6 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Analytics Summary
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Active models in {analyticsSummary.time_period}
                    </Typography>

                    <Box sx={{ mt: 2 }}>
                      <Typography variant="h3" color="primary">
                        {analyticsSummary.total_models}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Active Models
                      </Typography>
                    </Box>

                    <Typography variant="body2" sx={{ mt: 2 }}>
                      Last updated: {new Date(analyticsSummary.last_updated).toLocaleString()}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid size={{ xs: 12, md: 6 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Selected Models
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      Choose models for analysis
                    </Typography>

                    <Box sx={{ mt: 2 }}>
                      <Typography variant="h3" color="primary">
                        {selectedModels.length}
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Models Selected
                      </Typography>
                    </Box>

                    {selectedModels.length >= 2 && (
                      <Alert severity="success" sx={{ mt: 2 }}>
                        Ready for cross-model analysis
                      </Alert>
                    )}

                    {selectedModels.length < 2 && (
                      <Alert severity="info" sx={{ mt: 2 }}>
                        Select at least 2 models for comparison
                      </Alert>
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          ) : (
            <Typography>No analytics data available</Typography>
          )}
        </TabPanel>

        {/* Model Comparison Tab */}
        <TabPanel value={tabValue} index={1}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Model Comparison</Typography>
            <Button
              variant="contained"
              onClick={loadModelComparison}
              disabled={loading || selectedModels.length < 2}
              startIcon={loading ? <CircularProgress size={20} /> : <Compare />}
            >
              {loading ? 'Comparing...' : 'Compare Models'}
            </Button>
          </Box>

          {modelComparison && (
            <Grid container spacing={3}>
              {/* Performance Ranking */}
              <Grid size={{ xs: 12, md: 6 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Performance Ranking (by Best Loss)
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Rank</TableCell>
                            <TableCell>Model</TableCell>
                            <TableCell align="right">Loss</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {modelComparison.performance_ranking.map(([model, loss], index) => (
                            <TableRow key={model}>
                              <TableCell>
                                <Box display="flex" alignItems="center">
                                  {index === 0 && <Star color="warning" sx={{ mr: 1 }} />}
                                  {index + 1}
                                </Box>
                              </TableCell>
                              <TableCell>{model}</TableCell>
                              <TableCell align="right">{formatLoss(loss)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Efficiency Ranking */}
              <Grid size={{ xs: 12, md: 6 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Efficiency Ranking
                    </Typography>
                    <TableContainer>
                      <Table size="small">
                        <TableHead>
                          <TableRow>
                            <TableCell>Rank</TableCell>
                            <TableCell>Model</TableCell>
                            <TableCell align="right">Efficiency</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {modelComparison.efficiency_ranking.map(([model, efficiency], index) => (
                            <TableRow key={model}>
                              <TableCell>
                                <Box display="flex" alignItems="center">
                                  {index === 0 && <Speed color="success" sx={{ mr: 1 }} />}
                                  {index + 1}
                                </Box>
                              </TableCell>
                              <TableCell>{model}</TableCell>
                              <TableCell align="right">{formatScore(efficiency)}</TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Ensemble Potential */}
              <Grid size={{ xs: 12 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Analysis Summary
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid size={{ xs: 12, md: 4 }}>
                        <Box textAlign="center">
                          <Typography variant="h4" color="primary">
                            {formatScore(modelComparison.ensemble_potential)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Ensemble Potential
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid size={{ xs: 12, md: 8 }}>
                        <Typography variant="body2" color="text.secondary">
                          Compared Models: {modelComparison.compared_models.join(', ')}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Analysis completed: {new Date(modelComparison.analysis_timestamp).toLocaleString()}
                        </Typography>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}
        </TabPanel>

        {/* Ensemble Recommendations Tab */}
        <TabPanel value={tabValue} index={2}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Ensemble Recommendations</Typography>
            <Button
              variant="contained"
              onClick={loadEnsembleRecommendations}
              disabled={loading || selectedModels.length < 2}
              startIcon={loading ? <CircularProgress size={20} /> : <GroupWork />}
            >
              {loading ? 'Generating...' : 'Generate Recommendations'}
            </Button>
          </Box>

          {ensembleRecommendations.length > 0 && (
            <Grid container spacing={3}>
              {ensembleRecommendations.map((recommendation, index) => (
                <Grid size={{ xs: 12 }} key={index}>
                  <Card>
                    <CardContent>
                      <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
                        <Typography variant="h6">
                          Strategy {index + 1}
                        </Typography>
                        <Chip
                          label={`${formatScore(recommendation.confidence_score)} Confidence`}
                          color={recommendation.confidence_score > 0.8 ? 'success' :
                                 recommendation.confidence_score > 0.6 ? 'warning' : 'default'}
                        />
                      </Box>

                      <Grid container spacing={2}>
                        <Grid size={{ xs: 12, md: 6 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Recommended Models & Weights
                          </Typography>
                          {recommendation.recommended_models.map((model, modelIndex) => (
                            <Box key={model} display="flex" alignItems="center" mb={1}>
                              <Typography variant="body2" sx={{ minWidth: 120 }}>
                                {model}
                              </Typography>
                              <Box sx={{ flexGrow: 1, mx: 2 }}>
                                <Box
                                  sx={{
                                    width: `${recommendation.weights[modelIndex] * 100}%`,
                                    height: 8,
                                    backgroundColor: 'primary.main',
                                    borderRadius: 1
                                  }}
                                />
                              </Box>
                              <Typography variant="body2" sx={{ minWidth: 50 }}>
                                {formatScore(recommendation.weights[modelIndex])}
                              </Typography>
                            </Box>
                          ))}
                        </Grid>

                        <Grid size={{ xs: 12, md: 6 }}>
                          <Typography variant="subtitle2" gutterBottom>
                            Expected Performance
                          </Typography>
                          <Typography variant="h4" color="primary" gutterBottom>
                            {formatLoss(recommendation.expected_performance)}
                          </Typography>

                          <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                            Risk Assessment
                          </Typography>
                          <Chip
                            label={recommendation.risk_assessment}
                            color={getRiskColor(recommendation.risk_assessment)}
                            size="small"
                          />
                        </Grid>

                        <Grid size={{ xs: 12 }}>
                          <Divider sx={{ my: 1 }} />
                          <Typography variant="body2" color="text.secondary">
                            <strong>Reasoning:</strong> {recommendation.reasoning}
                          </Typography>
                        </Grid>
                      </Grid>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </TabPanel>

        {/* Performance Matrix Tab */}
        <TabPanel value={tabValue} index={3}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Performance Matrix</Typography>
            <Button
              variant="contained"
              onClick={loadPerformanceMatrix}
              disabled={loading || selectedModels.length < 2}
              startIcon={loading ? <CircularProgress size={20} /> : <Speed />}
            >
              {loading ? 'Generating...' : 'Generate Matrix'}
            </Button>
          </Box>

          {performanceMatrix && (
            <TableContainer component={Paper}>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>Model</TableCell>
                    {performanceMatrix.metrics.map((metric) => (
                      <TableCell key={metric} align="right">
                        {metric.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </TableCell>
                    ))}
                  </TableRow>
                </TableHead>
                <TableBody>
                  {performanceMatrix.models.map((model, modelIndex) => (
                    <TableRow key={model}>
                      <TableCell component="th" scope="row">
                        <strong>{model}</strong>
                      </TableCell>
                      {performanceMatrix.metrics.map((metric, metricIndex) => {
                        const value = performanceMatrix.data[metricIndex][modelIndex];
                        return (
                          <TableCell key={metric} align="right">
                            {value !== null ?
                              (metric.includes('loss') ? formatLoss(value) : formatScore(value))
                              : 'N/A'
                            }
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </TabPanel>

        {/* Trend Analysis Tab */}
        <TabPanel value={tabValue} index={4}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">Trend Analysis</Typography>
            <Button
              variant="contained"
              onClick={loadTrendAnalysis}
              disabled={loading}
              startIcon={loading ? <CircularProgress size={20} /> : <Timeline />}
            >
              {loading ? 'Analyzing...' : 'Analyze Trends'}
            </Button>
          </Box>

          {trendAnalysis && (
            <Grid container spacing={3}>
              {/* Overall Trends */}
              <Grid size={{ xs: 12 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Overall Trends ({trendAnalysis.time_period})
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid size={{ xs: 12, md: 4 }}>
                        <Box textAlign="center">
                          <Typography variant="h4" color="primary">
                            {trendAnalysis.active_models}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Active Models
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid size={{ xs: 12, md: 4 }}>
                        <Box textAlign="center">
                          <Typography variant="h4" color={
                            trendAnalysis.overall_trends.average_improvement_rate > 0 ? 'success.main' : 'error.main'
                          }>
                            {(trendAnalysis.overall_trends.average_improvement_rate * 100).toFixed(1)}%
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Avg Improvement Rate
                          </Typography>
                        </Box>
                      </Grid>
                      <Grid size={{ xs: 12, md: 4 }}>
                        <Box textAlign="center">
                          <Typography variant="h4" color="info.main">
                            {formatScore(trendAnalysis.overall_trends.average_consistency)}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Avg Consistency
                          </Typography>
                        </Box>
                      </Grid>
                    </Grid>
                  </CardContent>
                </Card>
              </Grid>

              {/* Model Trends */}
              <Grid size={{ xs: 12 }}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Individual Model Trends
                    </Typography>
                    <TableContainer>
                      <Table>
                        <TableHead>
                          <TableRow>
                            <TableCell>Model</TableCell>
                            <TableCell align="center">Trend</TableCell>
                            <TableCell align="right">Improvement Rate</TableCell>
                            <TableCell align="right">Consistency</TableCell>
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {Object.entries(trendAnalysis.model_trends).map(([model, trend]) => (
                            <TableRow key={model}>
                              <TableCell>{model}</TableCell>
                              <TableCell align="center">
                                <Box display="flex" alignItems="center" justifyContent="center">
                                  {getTrendIcon(trend.trend_direction)}
                                  <Typography variant="body2" sx={{ ml: 1, textTransform: 'capitalize' }}>
                                    {trend.trend_direction.replace('_', ' ')}
                                  </Typography>
                                </Box>
                              </TableCell>
                              <TableCell align="right">
                                <Typography
                                  color={trend.improvement_rate > 0 ? 'success.main' :
                                         trend.improvement_rate < 0 ? 'error.main' : 'text.primary'}
                                >
                                  {(trend.improvement_rate * 100).toFixed(1)}%
                                </Typography>
                              </TableCell>
                              <TableCell align="right">
                                {formatScore(trend.consistency)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                  </CardContent>
                </Card>
              </Grid>

              {/* Insights */}
              {trendAnalysis.insights.length > 0 && (
                <Grid size={{ xs: 12 }}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6" gutterBottom>
                        Key Insights
                      </Typography>
                      {trendAnalysis.insights.map((insight, index) => (
                        <Alert key={index} severity="info" sx={{ mb: 1 }}>
                          {insight}
                        </Alert>
                      ))}
                    </CardContent>
                  </Card>
                </Grid>
              )}
            </Grid>
          )}
        </TabPanel>
      </Card>
    </Box>
  );
};

export default CrossModelAnalytics;
