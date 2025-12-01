"""
Decision Engine for Helios AI System
Phase 3 Implementation: Autonomous decision-making and parameter optimization

This module provides autonomous decision-making capabilities including:
- Automatic parameter optimization
- Goal management and prioritization  
- Resource allocation decisions
- Training strategy adaptation
- Performance-based decision making
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
import threading

# Use absolute imports instead of relative imports
from memory_store import MemoryStore
from metacognition import MetacognitiveEngine, MetacognitiveAssessment, LearningStrategy

logger = logging.getLogger(__name__)

class DecisionType(Enum):
    """Types of decisions the engine can make"""
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    STRATEGY_CHANGE = "strategy_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    GOAL_PRIORITIZATION = "goal_prioritization"
    TRAINING_SCHEDULE = "training_schedule"
    EVALUATION_TRIGGER = "evaluation_trigger"

class DecisionPriority(Enum):
    """Priority levels for decisions"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

class DecisionStatus(Enum):
    """Status of decision execution"""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Goal:
    """Represents a training or performance goal"""
    goal_id: str
    name: str
    target_metric: str
    target_value: float
    current_value: float
    priority: int
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    progress: float = 0.0
    active: bool = True
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Decision:
    """Represents an autonomous decision"""
    decision_id: str
    decision_type: DecisionType
    priority: DecisionPriority
    context: Dict[str, Any]
    parameters: Dict[str, Any]
    rationale: str
    expected_impact: float
    confidence: float
    status: DecisionStatus = DecisionStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None

@dataclass
class ResourceAllocation:
    """Represents resource allocation decision"""
    compute_priority: float
    memory_allocation: float
    io_bandwidth: float
    evaluation_frequency: int
    parallel_jobs: int

class DecisionEngine:
    """
    Autonomous decision-making engine for the Helios AI system
    """
    
    def __init__(self, memory_store: MemoryStore, metacognitive_engine: MetacognitiveEngine):
        """
        Initialize the decision engine
        
        Args:
            memory_store: Instance of MemoryStore for persistent data
            metacognitive_engine: Instance of MetacognitiveEngine for self-assessment
        """
        self.memory_store = memory_store
        self.metacognitive_engine = metacognitive_engine
        
        # Decision management
        self.pending_decisions: List[Decision] = []
        self.active_goals: Dict[str, Goal] = {}
        self.decision_history: List[Decision] = []
        
        # Decision parameters
        self.max_concurrent_decisions = 3
        self.decision_confidence_threshold = 0.6
        self.goal_progress_threshold = 0.1
        
        # Resource management
        self.base_resources = ResourceAllocation(
            compute_priority=1.0,
            memory_allocation=1.0,
            io_bandwidth=1.0,
            evaluation_frequency=100,
            parallel_jobs=1
        )
        
        # Threading for autonomous operation
        self.decision_lock = threading.Lock()
        self.autonomous_mode = False
        self.decision_executor = ThreadPoolExecutor(max_workers=2)
        
        logger.info("Decision engine initialized")
    
    def start_autonomous_mode(self):
        """Start autonomous decision-making mode"""
        self.autonomous_mode = True
        logger.info("Autonomous decision-making mode activated")
    
    def stop_autonomous_mode(self):
        """Stop autonomous decision-making mode"""
        self.autonomous_mode = False
        logger.info("Autonomous decision-making mode deactivated")
    
    def add_goal(self, goal: Goal) -> bool:
        """
        Add a new goal to the system
        
        Args:
            goal: Goal to add
            
        Returns:
            True if goal was added successfully
        """
        with self.decision_lock:
            if goal.goal_id in self.active_goals:
                logger.warning(f"Goal {goal.goal_id} already exists")
                return False
            
            self.active_goals[goal.goal_id] = goal
            
            # Store goal in memory
            goal_data = {
                'goal_id': goal.goal_id,
                'name': goal.name,
                'target_metric': goal.target_metric,
                'target_value': goal.target_value,
                'current_value': goal.current_value,
                'priority': goal.priority,
                'deadline': goal.deadline.isoformat() if goal.deadline else None,
                'dependencies': goal.dependencies
            }
            
            self.memory_store.store_enhanced_journal_entry(
                model_name="system",
                session_id="goal_management",
                event_type="goal_added",
                event_data=goal_data
            )
            
            logger.info(f"Goal added: {goal.name} (ID: {goal.goal_id})")
            return True
    
    def update_goal_progress(self, goal_id: str, current_value: float) -> bool:
        """
        Update progress on a goal
        
        Args:
            goal_id: ID of the goal to update
            current_value: Current value of the goal metric
            
        Returns:
            True if goal was updated successfully
        """
        with self.decision_lock:
            if goal_id not in self.active_goals:
                logger.warning(f"Goal {goal_id} not found")
                return False
            
            goal = self.active_goals[goal_id]
            old_value = goal.current_value
            goal.current_value = current_value
            
            # Calculate progress
            if goal.target_value != old_value:
                goal.progress = min(1.0, max(0.0, 
                    (current_value - old_value) / (goal.target_value - old_value)))
            
            # Check if goal is completed
            if goal.progress >= 1.0:
                goal.active = False
                logger.info(f"Goal completed: {goal.name}")
                
                # Trigger decision for goal completion
                self._trigger_goal_completion_decisions(goal)
            
            # Store progress update
            progress_data = {
                'goal_id': goal_id,
                'old_value': old_value,
                'new_value': current_value,
                'progress': goal.progress
            }
            
            self.memory_store.store_enhanced_journal_entry(
                model_name="system",
                session_id="goal_management",
                event_type="goal_progress_update",
                event_data=progress_data
            )
            
            return True
    
    def make_autonomous_decision(self,
                               model_name: str,
                               current_metrics: Dict[str, float],
                               recent_performance: List[float],
                               context: Optional[Dict[str, Any]] = None) -> List[Decision]:
        """
        Make autonomous decisions based on current state
        
        Args:
            model_name: Name of the model
            current_metrics: Current performance metrics
            recent_performance: Recent performance history
            context: Additional context information
            
        Returns:
            List of decisions made
        """
        if not self.autonomous_mode:
            return []
        
        logger.info(f"Making autonomous decisions for model {model_name}")
        
        # Get metacognitive assessment
        assessment = self.metacognitive_engine.assess_current_state(
            model_name, current_metrics, recent_performance, context
        )
        
        decisions = []
        
        # Parameter optimization decisions
        param_decisions = self._make_parameter_decisions(model_name, assessment, current_metrics)
        decisions.extend(param_decisions)
        
        # Strategy adaptation decisions
        strategy_decisions = self._make_strategy_decisions(model_name, assessment, recent_performance)
        decisions.extend(strategy_decisions)
        
        # Resource allocation decisions
        resource_decisions = self._make_resource_decisions(model_name, assessment, current_metrics)
        decisions.extend(resource_decisions)
        
        # Goal management decisions
        goal_decisions = self._make_goal_decisions(model_name, assessment, current_metrics)
        decisions.extend(goal_decisions)
        
        # Add decisions to pending queue
        with self.decision_lock:
            self.pending_decisions.extend(decisions)
        
        # Execute high-priority decisions immediately
        self._execute_pending_decisions()
        
        logger.info(f"Made {len(decisions)} autonomous decisions")
        return decisions
    
    def _make_parameter_decisions(self,
                                model_name: str,
                                assessment: MetacognitiveAssessment,
                                current_metrics: Dict[str, float]) -> List[Decision]:
        """Make decisions about parameter adjustments"""
        decisions = []
        
        # Learning rate adjustment
        if assessment.confidence_score < 0.4 or assessment.uncertainty_estimate > 0.7:
            # Recommend learning rate change
            lr_recommendations = self.metacognitive_engine.get_learning_recommendations(
                model_name, assessment
            )['learning_rate_adjustment']
            
            decision = Decision(
                decision_id=f"lr_adjust_{datetime.now().timestamp()}",
                decision_type=DecisionType.PARAMETER_ADJUSTMENT,
                priority=DecisionPriority.HIGH,
                context={'model_name': model_name, 'assessment': assessment.__dict__},
                parameters={'learning_rate_adjustment': lr_recommendations},
                rationale=f"Low confidence ({assessment.confidence_score:.3f}) or high uncertainty "
                         f"({assessment.uncertainty_estimate:.3f}) detected",
                expected_impact=0.3,
                confidence=assessment.confidence_score
            )
            decisions.append(decision)
        
        # Batch size optimization
        if 'training_stability' in current_metrics:
            stability = current_metrics['training_stability']
            if stability < 0.5:
                decision = Decision(
                    decision_id=f"batch_adjust_{datetime.now().timestamp()}",
                    decision_type=DecisionType.PARAMETER_ADJUSTMENT,
                    priority=DecisionPriority.MEDIUM,
                    context={'model_name': model_name, 'stability': stability},
                    parameters={'batch_size_multiplier': 1.5 if stability < 0.3 else 1.2},
                    rationale=f"Training instability detected: {stability:.3f}",
                    expected_impact=0.2,
                    confidence=0.7
                )
                decisions.append(decision)
        
        return decisions
    
    def _make_strategy_decisions(self,
                               model_name: str,
                               assessment: MetacognitiveAssessment,
                               recent_performance: List[float]) -> List[Decision]:
        """Make decisions about training strategy changes"""
        decisions = []
        
        # Check for performance plateau
        if len(recent_performance) >= 10:
            recent_trend = np.polyfit(range(len(recent_performance[-10:])), 
                                    recent_performance[-10:], 1)[0]
            
            if abs(recent_trend) < 0.001:  # Plateau detected
                decision = Decision(
                    decision_id=f"strategy_change_{datetime.now().timestamp()}",
                    decision_type=DecisionType.STRATEGY_CHANGE,
                    priority=DecisionPriority.HIGH,
                    context={'model_name': model_name, 'plateau_detected': True},
                    parameters={'new_strategy': 'aggressive_exploration'},
                    rationale="Performance plateau detected, switching to aggressive exploration",
                    expected_impact=0.4,
                    confidence=0.8
                )
                decisions.append(decision)
        
        # Strategy alignment with metacognitive recommendation
        if assessment.recommended_strategy != LearningStrategy.BALANCED:
            decision = Decision(
                decision_id=f"strategy_align_{datetime.now().timestamp()}",
                decision_type=DecisionType.STRATEGY_CHANGE,
                priority=DecisionPriority.MEDIUM,
                context={'model_name': model_name, 'assessment': assessment.__dict__},
                parameters={'recommended_strategy': assessment.recommended_strategy.value},
                rationale=f"Aligning with metacognitive recommendation: {assessment.recommended_strategy.value}",
                expected_impact=0.25,
                confidence=assessment.confidence_score
            )
            decisions.append(decision)
        
        return decisions
    
    def _make_resource_decisions(self,
                               model_name: str,
                               assessment: MetacognitiveAssessment,
                               current_metrics: Dict[str, float]) -> List[Decision]:
        """Make decisions about resource allocation"""
        decisions = []
        
        # Evaluation frequency adjustment
        current_eval_freq = self.base_resources.evaluation_frequency
        
        if assessment.uncertainty_estimate > 0.7:
            # Need more frequent evaluation when uncertain
            new_freq = max(50, current_eval_freq // 2)
            decision = Decision(
                decision_id=f"eval_freq_{datetime.now().timestamp()}",
                decision_type=DecisionType.RESOURCE_ALLOCATION,
                priority=DecisionPriority.MEDIUM,
                context={'model_name': model_name, 'uncertainty': assessment.uncertainty_estimate},
                parameters={'evaluation_frequency': new_freq},
                rationale=f"High uncertainty ({assessment.uncertainty_estimate:.3f}) requires more frequent evaluation",
                expected_impact=0.15,
                confidence=0.7
            )
            decisions.append(decision)
        
        elif assessment.confidence_score > 0.8:
            # Can reduce evaluation frequency when confident
            new_freq = min(200, current_eval_freq * 2)
            decision = Decision(
                decision_id=f"eval_freq_{datetime.now().timestamp()}",
                decision_type=DecisionType.RESOURCE_ALLOCATION,
                priority=DecisionPriority.LOW,
                context={'model_name': model_name, 'confidence': assessment.confidence_score},
                parameters={'evaluation_frequency': new_freq},
                rationale=f"High confidence ({assessment.confidence_score:.3f}) allows less frequent evaluation",
                expected_impact=0.1,
                confidence=0.8
            )
            decisions.append(decision)
        
        return decisions
    
    def _make_goal_decisions(self,
                           model_name: str,
                           assessment: MetacognitiveAssessment,
                           current_metrics: Dict[str, float]) -> List[Decision]:
        """Make decisions about goal management and prioritization"""
        decisions = []
        
        with self.decision_lock:
            active_goals = [g for g in self.active_goals.values() if g.active]
        
        if not active_goals:
            return decisions
        
        # Re-prioritize goals based on current progress and deadlines
        goal_priorities = []
        for goal in active_goals:
            # Update current value if metric is available
            if goal.target_metric in current_metrics:
                self.update_goal_progress(goal.goal_id, current_metrics[goal.target_metric])
            
            # Calculate priority score
            urgency = 1.0
            if goal.deadline:
                days_left = (goal.deadline - datetime.now()).days
                urgency = max(0.1, 1.0 / max(1, days_left))
            
            priority_score = (
                goal.priority * 0.3 +
                (1.0 - goal.progress) * 0.4 +  # More urgent if less progress
                urgency * 0.3
            )
            
            goal_priorities.append((goal, priority_score))
        
        # Sort by priority score
        goal_priorities.sort(key=lambda x: x[1], reverse=True)
        
        # Create goal prioritization decision
        if len(goal_priorities) > 1:
            reordered_goals = [g[0].goal_id for g in goal_priorities]
            decision = Decision(
                decision_id=f"goal_priority_{datetime.now().timestamp()}",
                decision_type=DecisionType.GOAL_PRIORITIZATION,
                priority=DecisionPriority.MEDIUM,
                context={'model_name': model_name, 'active_goals': len(active_goals)},
                parameters={'goal_order': reordered_goals},
                rationale="Re-prioritizing goals based on progress, deadlines, and current performance",
                expected_impact=0.2,
                confidence=0.6
            )
            decisions.append(decision)
        
        return decisions
    
    def _trigger_goal_completion_decisions(self, completed_goal: Goal):
        """Trigger decisions when a goal is completed"""
        # Create new goals based on completed goal
        if completed_goal.dependencies:
            for dep_goal_id in completed_goal.dependencies:
                if dep_goal_id in self.active_goals:
                    dependent_goal = self.active_goals[dep_goal_id]
                    dependent_goal.priority = max(1, dependent_goal.priority - 1)  # Increase priority
        
        # Log completion
        completion_data = {
            'goal_id': completed_goal.goal_id,
            'name': completed_goal.name,
            'final_value': completed_goal.current_value,
            'target_value': completed_goal.target_value,
            'completion_time': datetime.now().isoformat()
        }
        
        self.memory_store.store_enhanced_journal_entry(
            model_name="system",
            session_id="goal_management",
            event_type="goal_completed",
            event_data=completion_data
        )
    
    def _execute_pending_decisions(self):
        """Execute pending decisions based on priority"""
        with self.decision_lock:
            # Sort by priority
            self.pending_decisions.sort(key=lambda d: d.priority.value)
            
            # Execute high-priority decisions immediately
            high_priority_decisions = [
                d for d in self.pending_decisions 
                if d.priority in [DecisionPriority.CRITICAL, DecisionPriority.HIGH]
            ]
            
            for decision in high_priority_decisions[:self.max_concurrent_decisions]:
                if decision.confidence >= self.decision_confidence_threshold:
                    self._execute_decision(decision)
                    self.pending_decisions.remove(decision)
    
    def _execute_decision(self, decision: Decision):
        """Execute a specific decision"""
        logger.info(f"Executing decision: {decision.decision_id} ({decision.decision_type.value})")
        
        decision.status = DecisionStatus.EXECUTING
        decision.executed_at = datetime.now()
        
        try:
            # Execute based on decision type
            if decision.decision_type == DecisionType.PARAMETER_ADJUSTMENT:
                result = self._execute_parameter_adjustment(decision)
            elif decision.decision_type == DecisionType.STRATEGY_CHANGE:
                result = self._execute_strategy_change(decision)
            elif decision.decision_type == DecisionType.RESOURCE_ALLOCATION:
                result = self._execute_resource_allocation(decision)
            elif decision.decision_type == DecisionType.GOAL_PRIORITIZATION:
                result = self._execute_goal_prioritization(decision)
            else:
                result = {'status': 'not_implemented'}
            
            decision.result = result
            decision.status = DecisionStatus.COMPLETED
            decision.completed_at = datetime.now()
            
            logger.info(f"Decision executed successfully: {decision.decision_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute decision {decision.decision_id}: {str(e)}")
            decision.status = DecisionStatus.FAILED
            decision.result = {'error': str(e)}
        
        # Store decision execution result
        self._store_decision_result(decision)
        
        # Add to history
        self.decision_history.append(decision)
    
    def _execute_parameter_adjustment(self, decision: Decision) -> Dict[str, Any]:
        """Execute parameter adjustment decision"""
        # This would interface with the training system to adjust parameters
        # For now, we'll just log the decision
        
        params = decision.parameters
        logger.info(f"Parameter adjustment: {params}")
        
        # Store the parameter change as a performance metric
        for param_name, param_value in params.items():
            self.memory_store.store_performance_metric(
                model_name=decision.context.get('model_name', 'unknown'),
                metric_name=f"decision_param_{param_name}",
                metric_value=float(param_value) if isinstance(param_value, (int, float)) else 1.0,
                context=f"Autonomous decision: {decision.decision_id}"
            )
        
        return {'status': 'applied', 'parameters': params}
    
    def _execute_strategy_change(self, decision: Decision) -> Dict[str, Any]:
        """Execute strategy change decision"""
        strategy = decision.parameters.get('new_strategy') or decision.parameters.get('recommended_strategy')
        
        logger.info(f"Strategy change: {strategy}")
        
        # Store strategy change
        self.memory_store.store_enhanced_journal_entry(
            model_name=decision.context.get('model_name', 'unknown'),
            session_id="strategy_management",
            event_type="strategy_change",
            event_data={'new_strategy': strategy, 'decision_id': decision.decision_id}
        )
        
        return {'status': 'applied', 'new_strategy': strategy}
    
    def _execute_resource_allocation(self, decision: Decision) -> Dict[str, Any]:
        """Execute resource allocation decision"""
        allocation = decision.parameters
        
        # Update base resources
        for key, value in allocation.items():
            if hasattr(self.base_resources, key):
                setattr(self.base_resources, key, value)
        
        logger.info(f"Resource allocation updated: {allocation}")
        
        return {'status': 'applied', 'allocation': allocation}
    
    def _execute_goal_prioritization(self, decision: Decision) -> Dict[str, Any]:
        """Execute goal prioritization decision"""
        goal_order = decision.parameters.get('goal_order', [])
        
        # Update goal priorities based on order
        with self.decision_lock:
            for i, goal_id in enumerate(goal_order):
                if goal_id in self.active_goals:
                    self.active_goals[goal_id].priority = i + 1
        
        logger.info(f"Goal priorities updated: {goal_order}")
        
        return {'status': 'applied', 'new_order': goal_order}
    
    def _store_decision_result(self, decision: Decision):
        """Store decision execution result in memory"""
        decision_data = {
            'decision_id': decision.decision_id,
            'decision_type': decision.decision_type.value,
            'priority': decision.priority.value,
            'parameters': decision.parameters,
            'rationale': decision.rationale,
            'expected_impact': decision.expected_impact,
            'confidence': decision.confidence,
            'status': decision.status.value,
            'result': decision.result,
            'execution_time': (decision.completed_at - decision.executed_at).total_seconds() 
                            if decision.completed_at and decision.executed_at else None
        }
        
        self.memory_store.store_enhanced_journal_entry(
            model_name=decision.context.get('model_name', 'system'),
            session_id="decision_execution",
            event_type="decision_result",
            event_data=decision_data,
            confidence_score=decision.confidence
        )
    
    def get_decision_history(self,
                           days: int = 7,
                           decision_type: Optional[DecisionType] = None) -> List[Decision]:
        """Get recent decision history"""
        cutoff = datetime.now() - timedelta(days=days)
        
        filtered_decisions = [
            d for d in self.decision_history 
            if d.created_at >= cutoff
        ]
        
        if decision_type:
            filtered_decisions = [
                d for d in filtered_decisions 
                if d.decision_type == decision_type
            ]
        
        return sorted(filtered_decisions, key=lambda d: d.created_at, reverse=True)
    
    def get_goal_status(self) -> Dict[str, Any]:
        """Get current status of all goals"""
        with self.decision_lock:
            active_goals = [g for g in self.active_goals.values() if g.active]
            completed_goals = [g for g in self.active_goals.values() if not g.active]
        
        return {
            'active_goals': len(active_goals),
            'completed_goals': len(completed_goals),
            'total_progress': sum(g.progress for g in active_goals) / max(1, len(active_goals)),
            'goals': [
                {
                    'goal_id': g.goal_id,
                    'name': g.name,
                    'progress': g.progress,
                    'priority': g.priority,
                    'active': g.active
                }
                for g in self.active_goals.values()
            ]
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'pending_decisions': len(self.pending_decisions),
            'recent_decisions': len(self.get_decision_history(days=1)),
            'goal_status': self.get_goal_status(),
            'resource_allocation': {
                'compute_priority': self.base_resources.compute_priority,
                'memory_allocation': self.base_resources.memory_allocation,
                'io_bandwidth': self.base_resources.io_bandwidth,
                'evaluation_frequency': self.base_resources.evaluation_frequency,
                'parallel_jobs': self.base_resources.parallel_jobs
            }
        }
