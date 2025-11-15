"""
Base Agent implementation with self-organization capabilities.
"""

import asyncio
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class AgentCapability(Enum):
    """Capabilities that agents can possess."""

    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    TESTING = "testing"
    SCHEMA_MANAGEMENT = "schema_management"
    API_INTEGRATION = "api_integration"
    DEPLOYMENT = "deployment"
    DOCUMENTATION = "documentation"
    COORDINATION = "coordination"


class AgentStatus(Enum):
    """Current status of an agent."""

    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class Task:
    """Represents a task that can be executed by an agent."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    required_capabilities: Set[AgentCapability] = field(default_factory=set)
    data: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)  # Task IDs
    result: Optional[Any] = None
    error: Optional[str] = None
    completed: bool = False


@dataclass
class AgentMessage:
    """Message for inter-agent communication."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    recipient_id: Optional[str] = None  # None for broadcast
    message_type: str = ""  # e.g., "task_request", "task_offer", "status_update"
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class Agent(ABC):
    """
    Base class for all agents in the system.

    Agents can self-organize into workflows by:
    1. Broadcasting their capabilities
    2. Responding to task requests
    3. Coordinating with other agents
    4. Executing tasks independently
    """

    def __init__(
        self,
        name: str,
        capabilities: Set[AgentCapability],
        agent_id: Optional[str] = None,
    ):
        self.id = agent_id or str(uuid.uuid4())
        self.name = name
        self.capabilities = capabilities
        self.status = AgentStatus.IDLE
        self.current_task: Optional[Task] = None
        self.message_queue: asyncio.Queue[AgentMessage] = asyncio.Queue()
        self.task_history: List[Task] = []
        self.peers: Dict[str, "Agent"] = {}  # Known other agents
        self._running = False

    def register_peer(self, agent: "Agent") -> None:
        """Register another agent as a peer for coordination."""
        self.peers[agent.id] = agent

    def unregister_peer(self, agent_id: str) -> None:
        """Unregister a peer agent."""
        self.peers.pop(agent_id, None)

    async def send_message(self, message: AgentMessage) -> None:
        """Send a message to another agent or broadcast."""
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.peers:
                await self.peers[message.recipient_id].receive_message(message)
        else:
            # Broadcast to all peers
            for peer in self.peers.values():
                await peer.receive_message(message)

    async def receive_message(self, message: AgentMessage) -> None:
        """Receive a message from another agent."""
        await self.message_queue.put(message)

    async def broadcast_capability(self) -> None:
        """Broadcast capabilities to all peer agents."""
        message = AgentMessage(
            sender_id=self.id,
            message_type="capability_announcement",
            payload={
                "name": self.name,
                "capabilities": [cap.value for cap in self.capabilities],
                "status": self.status.value,
            },
        )
        await self.send_message(message)

    def can_handle_task(self, task: Task) -> bool:
        """Check if this agent can handle the given task."""
        return task.required_capabilities.issubset(self.capabilities)

    async def offer_task_assistance(self, task: Task, coordinator_id: str) -> None:
        """Offer to help with a task."""
        message = AgentMessage(
            sender_id=self.id,
            recipient_id=coordinator_id,
            message_type="task_offer",
            payload={"task_id": task.id, "confidence": self._assess_task_confidence(task)},
        )
        await self.send_message(message)

    def _assess_task_confidence(self, task: Task) -> float:
        """Assess confidence in handling a task (0.0 to 1.0)."""
        if not self.can_handle_task(task):
            return 0.0

        # Higher confidence if all capabilities match exactly
        if task.required_capabilities == self.capabilities:
            return 1.0

        # Partial confidence based on capability overlap
        return len(task.required_capabilities) / max(len(self.capabilities), 1)

    async def execute_task(self, task: Task) -> Task:
        """
        Execute a task. This is the main entry point for task execution.
        """
        if self.status != AgentStatus.IDLE:
            task.error = f"Agent {self.name} is not idle"
            return task

        if not self.can_handle_task(task):
            task.error = f"Agent {self.name} cannot handle task requirements"
            return task

        self.status = AgentStatus.BUSY
        self.current_task = task

        try:
            # Notify peers that we're starting work
            await self.send_message(
                AgentMessage(
                    sender_id=self.id,
                    message_type="task_started",
                    payload={"task_id": task.id},
                )
            )

            # Execute the actual task logic
            task.result = await self._execute_task_logic(task)
            task.completed = True

            # Notify completion
            await self.send_message(
                AgentMessage(
                    sender_id=self.id,
                    message_type="task_completed",
                    payload={"task_id": task.id, "result": task.result},
                )
            )

        except Exception as e:
            task.error = str(e)
            self.status = AgentStatus.ERROR
            await self.send_message(
                AgentMessage(
                    sender_id=self.id,
                    message_type="task_failed",
                    payload={"task_id": task.id, "error": str(e)},
                )
            )
        finally:
            self.task_history.append(task)
            self.current_task = None
            if self.status != AgentStatus.ERROR:
                self.status = AgentStatus.IDLE

        return task

    @abstractmethod
    async def _execute_task_logic(self, task: Task) -> Any:
        """
        Implement the actual task execution logic.
        Must be overridden by subclasses.
        """
        pass

    async def run(self) -> None:
        """
        Main agent loop - processes messages and coordinates with other agents.
        """
        self._running = True
        while self._running:
            try:
                # Process messages with timeout to allow periodic checks
                message = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Agent {self.name} error processing message: {e}")

    async def _process_message(self, message: AgentMessage) -> None:
        """Process incoming messages. Can be overridden for custom behavior."""
        if message.message_type == "task_request":
            task_data = message.payload.get("task")
            if task_data:
                task = Task(**task_data)
                if self.can_handle_task(task) and self.status == AgentStatus.IDLE:
                    await self.offer_task_assistance(task, message.sender_id)

    def stop(self) -> None:
        """Stop the agent's run loop."""
        self._running = False

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status."""
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "current_task": self.current_task.id if self.current_task else None,
            "tasks_completed": len([t for t in self.task_history if t.completed]),
            "tasks_failed": len([t for t in self.task_history if t.error]),
        }
