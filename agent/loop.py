"""
agent/loop.py — AgentLoop: bucle de razonamiento y acción del agente
====================================================================

El AgentLoop implementa el ciclo:

    while not done and turn <= max_turns:
        1. Obtener acción del motor (tool_name, args, reasoning)
        2. Ejecutar la herramienta
        3. Registrar en session
        4. Detectar señales de terminación: DONE o FAIL

El motor es un callable que recibe (task, history_text) y retorna una
acción estructurada (dict con "tool", "args", "reasoning").

MockMotor: motor predefinido que retorna acciones en secuencia, útil
para tests sin dependencias externas.

IMPORTANTE: El AgentLoop NO instancia ningún modelo real (ni MoSE, ni GPT,
ni ningún transformador). El motor real se conecta externamente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .session import ActionEntry, AgentSession
from .tools import BaseTool, ToolResult, build_tool_registry


# ─────────────────────────────────────────────────────────────────────────────
# SEÑALES DE TERMINACIÓN
# ─────────────────────────────────────────────────────────────────────────────

DONE_SIGNAL = "DONE"    # El agente completó la tarea
FAIL_SIGNAL = "FAIL"    # El agente falló y no puede continuar


# ─────────────────────────────────────────────────────────────────────────────
# ACCIÓN DEL MOTOR
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MotorAction:
    """
    Acción que el motor decide ejecutar en un turno.

    tool:      Nombre de la herramienta a usar, o DONE_SIGNAL / FAIL_SIGNAL.
    args:      Argumentos para la herramienta.
    reasoning: Razonamiento del motor (texto libre, para logging).
    """
    tool:      str
    args:      Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""

    @property
    def is_done(self) -> bool:
        return self.tool == DONE_SIGNAL

    @property
    def is_fail(self) -> bool:
        return self.tool == FAIL_SIGNAL

    @property
    def is_terminal(self) -> bool:
        return self.is_done or self.is_fail


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADO DE UNA EJECUCIÓN DEL LOOP
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopResult:
    """
    Resultado de ejecutar el AgentLoop hasta terminación.

    status:   "done" | "fail" | "max_turns"
    session:  AgentSession con el historial completo
    turns_used: número de turnos ejecutados
    """
    status:     str
    session:    AgentSession
    turns_used: int

    @property
    def succeeded(self) -> bool:
        return self.status == "done"

    @property
    def failed(self) -> bool:
        return self.status in ("fail", "max_turns")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK MOTOR (para tests)
# ─────────────────────────────────────────────────────────────────────────────

class MockMotor:
    """
    Motor predefinido que retorna acciones en secuencia.

    Útil para tests sin depender de ningún modelo real.

    Uso:
        motor = MockMotor([
            MotorAction("bash", {"command": "ls"}, "Listing files"),
            MotorAction("cat",  {"path": "README.md"}, "Reading README"),
            MotorAction("DONE", {}, "Task complete"),
        ])
        loop = AgentLoop(motor=motor, ...)
    """

    def __init__(self, actions: List[MotorAction]) -> None:
        self._actions = list(actions)
        self._idx     = 0

    def __call__(self, task: str, history_text: str) -> MotorAction:
        """Retorna la siguiente acción de la secuencia."""
        if self._idx >= len(self._actions):
            # Si se acaban las acciones, señal de fallo
            return MotorAction(FAIL_SIGNAL, {}, "No more actions in sequence")
        action = self._actions[self._idx]
        self._idx += 1
        return action

    def reset(self) -> None:
        """Reinicia al inicio de la secuencia."""
        self._idx = 0


# ─────────────────────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────────────────────

class AgentLoop:
    """
    Bucle de razonamiento y acción del agente AION-C.

    Ejecuta el ciclo:
        motor(task, history) → MotorAction
        tools[action.tool].run(action.args) → ToolResult
        session.record_action(...)

    Hasta que el motor emite DONE/FAIL o se alcanza max_turns.

    Args:
        motor:     Callable(task, history_text) → MotorAction.
                   Puede ser MockMotor para tests, o cualquier LLM wrapeado.
        tools:     Dict[name → BaseTool]. None → herramientas por defecto.
        max_turns: Límite de turnos (evita loops infinitos). Default: 20.
        memory:    MemoryBridge opcional para contexto de memoria.
        verbose:   Si True, imprime cada turno.

    Uso:
        from agent.loop import AgentLoop, MockMotor, MotorAction
        from agent.tools import build_tool_registry

        motor = MockMotor([
            MotorAction("bash", {"command": "echo hello"}, "Test"),
            MotorAction("DONE", {}, "All done"),
        ])
        loop = AgentLoop(motor=motor, max_turns=10)
        result = loop.run(task="Say hello")

        assert result.succeeded
        assert result.turns_used == 2
    """

    def __init__(
        self,
        motor:     Callable,
        tools:     Optional[Dict[str, BaseTool]] = None,
        max_turns: int = 20,
        memory:    Optional[Any] = None,
        verbose:   bool = False,
        max_history_tokens: int = 4096,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.motor     = motor
        self.tools     = tools if tools is not None else build_tool_registry()
        self.max_turns = max_turns
        self.memory    = memory
        self.verbose   = verbose
        self.max_history_tokens = max_history_tokens
        self.system_prompt = system_prompt
        # Conversation history across multiple run() calls
        self._conversation_history: List[Dict[str, str]] = []

    # ── Métodos de orquestación ───────────────────────────────────────────────

    def run(self, task: str) -> LoopResult:
        """
        Ejecuta el loop hasta terminación o max_turns.

        Args:
            task: Descripción de la tarea para el agente.

        Returns:
            LoopResult con status, session y turns_used.
        """
        session = AgentSession(task=task)

        for turn in range(1, self.max_turns + 1):
            # Construir el texto de historial para el motor
            history_text = self._build_history(session)

            # Obtener la siguiente acción del motor
            action = self.motor(task, history_text)

            if self.verbose:
                print(f"  [turn {turn}] tool={action.tool!r} "
                      f"reasoning={action.reasoning[:60]!r}")

            # Señal de terminación
            if action.is_terminal:
                result = ToolResult("", "", 0, action.tool)
                session.record_action(
                    turn            = turn,
                    tool_name       = action.tool,
                    args            = action.args,
                    result          = result,
                    motor_reasoning = action.reasoning,
                )
                status = "done" if action.is_done else "fail"
                if action.is_fail:
                    session.record_error(f"Motor FAIL at turn {turn}: {action.reasoning}")
                return LoopResult(status=status, session=session, turns_used=turn)

            # Ejecutar la herramienta
            tool_result = self._execute_tool(action, session, turn)
            session.record_action(
                turn            = turn,
                tool_name       = action.tool,
                args            = action.args,
                result          = tool_result,
                motor_reasoning = action.reasoning,
            )

            # Auto-detectar archivos vistos en CatTool
            if action.tool == "cat" and "path" in action.args:
                session.record_file_seen(action.args["path"])

            if self.verbose:
                out = tool_result.stdout[:100] if tool_result else ""
                print(f"           → exit={tool_result.exit_code} out={out!r}")

        # Se agotaron los turnos
        session.record_error(f"max_turns={self.max_turns} reached without DONE/FAIL")
        return LoopResult(status="max_turns", session=session, turns_used=self.max_turns)

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _execute_tool(
        self,
        action:  MotorAction,
        session: AgentSession,
        turn:    int,
    ) -> ToolResult:
        """Ejecuta la herramienta indicada en la acción."""
        tool_name = action.tool

        if tool_name not in self.tools:
            err = f"Unknown tool: {tool_name!r}. Available: {list(self.tools.keys())}"
            session.record_error(err)
            return ToolResult("", err, 1, tool_name)

        try:
            return self.tools[tool_name].run(action.args)
        except Exception as exc:
            err = f"Tool {tool_name!r} raised: {exc}"
            session.record_error(err)
            return ToolResult("", err, 1, tool_name)

    def add_user_message(self, message: str) -> None:
        """Agrega un mensaje del usuario al historial de conversación."""
        self._conversation_history.append({"role": "user", "content": message})
        self._truncate_history()

    def add_assistant_message(self, message: str) -> None:
        """Agrega un mensaje del asistente al historial de conversación."""
        self._conversation_history.append({"role": "assistant", "content": message})
        self._truncate_history()

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Retorna el historial de conversación actual."""
        return list(self._conversation_history)

    def clear_history(self) -> None:
        """Limpia el historial de conversación."""
        self._conversation_history.clear()

    def _truncate_history(self) -> None:
        """Trunca el historial si excede max_history_tokens."""
        total = sum(len(m["content"]) // 4 for m in self._conversation_history)
        while total > self.max_history_tokens and len(self._conversation_history) > 1:
            self._conversation_history.pop(0)
            total = sum(len(m["content"]) // 4 for m in self._conversation_history)

    def _build_history(self, session: AgentSession) -> str:
        """
        Serializa el historial de la sesión como texto para el motor.

        El motor recibe esto como contexto de conversación.
        Incluye el system prompt si está configurado.
        """
        parts: List[str] = []

        # System prompt
        if self.system_prompt:
            parts.append(f"[system] {self.system_prompt}")

        # Conversation history (multi-turn)
        for msg in self._conversation_history:
            parts.append(f"[{msg['role']}] {msg['content'][:300]}")

        # Session actions
        if not session.actions and not parts:
            return "(no history)"

        for a in session.actions:
            parts.append(f"[turn {a.turn}] {a.tool_name}")
            if a.motor_reasoning:
                parts.append(f"  reasoning: {a.motor_reasoning}")
            if isinstance(a.result, ToolResult):
                out = a.result.as_text()[:200]
                parts.append(f"  result: {out}")
        return "\n".join(parts)
