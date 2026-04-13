"""
agent/ — Módulo de agente autónomo de AION-C
=============================================

Componentes:
  AgentLoop     (loop.py)     — bucle razonamiento → acción
  AgentSession  (session.py)  — historial de la sesión
  MemoryBridge  (memory_bridge.py) — interfaz de memoria
  Tools         (tools.py)    — herramientas ejecutables

Uso rápido:
    from agent import AgentLoop, MockMotor, MotorAction

    motor  = MockMotor([MotorAction("DONE", {}, "done")])
    loop   = AgentLoop(motor=motor, max_turns=5)
    result = loop.run(task="Test task")
    assert result.succeeded
"""

from .loop   import AgentLoop, MockMotor, MotorAction, LoopResult, DONE_SIGNAL, FAIL_SIGNAL
from .session import AgentSession, ActionEntry
from .memory_bridge import MemoryBridge
from .tools  import (
    ToolResult, BaseTool,
    BashTool, GrepTool, FindTool, CatTool, PytestTool,
    WebSearchTool, WebFetchTool, FileReadTool,
    WriteFileTool, EditFileTool, RunCodeTool, CallApiTool,
    SearchMemTool, StoreMemTool,
    DEFAULT_TOOLS, build_tool_registry,
)
from .tool_executor import (
    ToolCall, ToolExecutionRecord, ToolExecutor,
    parse_tool_calls, format_result,
)
from .planner import (
    Planner, Plan, PlanStep, StepResult, default_decompose,
    STATUS_PENDING, STATUS_IN_PROGRESS, STATUS_COMPLETED,
    STATUS_FAILED, STATUS_SKIPPED,
    PLAN_STATUS_DRAFT, PLAN_STATUS_RUNNING, PLAN_STATUS_COMPLETED,
    PLAN_STATUS_FAILED, PLAN_STATUS_TIMED_OUT,
)
from .skills import Skill, SkillsLoader, SKILL_DOMAIN
from .self_check import (
    ConfidenceLevel, SelfCheckResult, SelfChecker,
    ErrorRecord, ErrorLog,
    confidence_from_probs, classify_confidence, policy_for_confidence,
    HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD, ERROR_LOG_DOMAIN,
)
from .reasoning_levels import (
    ReasoningLevel, LevelDecision, LevelDecider,
    INSTANT_TRIGGERS, DEEP_TRIGGERS, COMPUTE_DOMAINS,
)
from .lifecycle import (
    SystemState, StateTransition, InvalidTransition,
    LifecycleManager, ALLOWED_TRANSITIONS,
)
from .goals import (
    GoalStatus, GoalSource, Goal, Task, Mission,
    GoalsManager, PERMANENT_MISSION,
)

__all__ = [
    "AgentLoop",
    "MockMotor",
    "MotorAction",
    "LoopResult",
    "DONE_SIGNAL",
    "FAIL_SIGNAL",
    "AgentSession",
    "ActionEntry",
    "MemoryBridge",
    "ToolResult",
    "BaseTool",
    "BashTool",
    "GrepTool",
    "FindTool",
    "CatTool",
    "PytestTool",
    "WebSearchTool",
    "WebFetchTool",
    "FileReadTool",
    "WriteFileTool",
    "EditFileTool",
    "RunCodeTool",
    "CallApiTool",
    "SearchMemTool",
    "StoreMemTool",
    "DEFAULT_TOOLS",
    "build_tool_registry",
    "ToolCall",
    "ToolExecutionRecord",
    "ToolExecutor",
    "parse_tool_calls",
    "format_result",
    "Planner",
    "Plan",
    "PlanStep",
    "StepResult",
    "default_decompose",
    "STATUS_PENDING",
    "STATUS_IN_PROGRESS",
    "STATUS_COMPLETED",
    "STATUS_FAILED",
    "STATUS_SKIPPED",
    "PLAN_STATUS_DRAFT",
    "PLAN_STATUS_RUNNING",
    "PLAN_STATUS_COMPLETED",
    "PLAN_STATUS_FAILED",
    "PLAN_STATUS_TIMED_OUT",
    "Skill",
    "SkillsLoader",
    "SKILL_DOMAIN",
    "ConfidenceLevel",
    "SelfCheckResult",
    "SelfChecker",
    "ErrorRecord",
    "ErrorLog",
    "confidence_from_probs",
    "classify_confidence",
    "policy_for_confidence",
    "HIGH_CONFIDENCE_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    "ERROR_LOG_DOMAIN",
    "ReasoningLevel",
    "LevelDecision",
    "LevelDecider",
    "INSTANT_TRIGGERS",
    "DEEP_TRIGGERS",
    "COMPUTE_DOMAINS",
    "SystemState",
    "StateTransition",
    "InvalidTransition",
    "LifecycleManager",
    "ALLOWED_TRANSITIONS",
    "GoalStatus", "GoalSource", "Goal", "Task", "Mission",
    "GoalsManager", "PERMANENT_MISSION",
]
