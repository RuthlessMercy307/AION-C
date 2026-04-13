"""
soma/interface.py — Interfaz SOMA (Parte 10.1 del MEGA-PROMPT)
================================================================

AION-C debe poder emitir comandos que SOMA (cuando exista) entendería.
Esta interfaz define la estructura del comando y un backend mockeable.

Tipos de comando (Parte 10.1):
  PRIMITIVE  — primitivos motores: stroke(dir, presión), rotate(eje, grados)
  HIGH_LEVEL — comandos de alto nivel: open_file, click, type
  GOAL       — objetivos: "esculpe forma orgánica en zona Z"

Estructura del JSON tool call (compatible con ToolExecutor):
  [TOOL: {"action": "soma_command",
          "input": {"type": "high_level",
                    "command": "open_file",
                    "args": {"path": "app.py"}}}]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SomaCommandType(str, Enum):
    PRIMITIVE  = "primitive"
    HIGH_LEVEL = "high_level"
    GOAL       = "goal"


# Vocabulario reconocido — listas extensibles, no se valida estrictamente
KNOWN_PRIMITIVES = ("stroke", "rotate", "press", "release", "move")
KNOWN_HIGH_LEVEL = (
    "open_file", "save_file", "click", "type", "scroll",
    "navigate", "screenshot", "wait",
)


@dataclass
class SomaCommand:
    """Un comando para SOMA."""
    type:    SomaCommandType
    command: str
    args:    Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":    self.type.value,
            "command": self.command,
            "args":    dict(self.args),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SomaCommand":
        ctype = data.get("type", "high_level")
        if isinstance(ctype, SomaCommandType):
            ct = ctype
        else:
            try:
                ct = SomaCommandType(ctype)
            except ValueError:
                raise ValueError(f"unknown soma command type: {ctype}")
        cmd = data.get("command", "")
        if not cmd:
            raise ValueError("soma command 'command' field is required")
        return cls(type=ct, command=cmd, args=dict(data.get("args", {})))


@dataclass
class SomaResult:
    """Resultado de ejecutar un comando SOMA."""
    success: bool
    output:  str = ""
    error:   str = ""

    def to_text(self) -> str:
        if self.success:
            return self.output or "ok"
        return f"[soma error] {self.error}"


# ─────────────────────────────────────────────────────────────────────────────
# Backend
# ─────────────────────────────────────────────────────────────────────────────


class MockSomaBackend:
    """
    Backend de SOMA por defecto. NO ejecuta nada real — solo registra
    los comandos recibidos y devuelve un SomaResult de éxito.

    Útil hasta que SOMA exista de verdad. Tests usan este backend.
    """

    def __init__(self) -> None:
        self.history: List[SomaCommand] = []

    def execute(self, command: SomaCommand) -> SomaResult:
        self.history.append(command)
        return SomaResult(
            success=True,
            output=f"mock executed: {command.type.value}/{command.command}",
        )


class SomaInterface:
    """
    Interfaz pública. AION-C llama .execute(command_dict) y la interfaz
    valida + delega al backend.

    Args:
        backend: cualquier objeto con .execute(SomaCommand) → SomaResult
    """

    def __init__(self, backend: Optional[Any] = None) -> None:
        self.backend = backend or MockSomaBackend()

    def execute(self, payload: Dict[str, Any]) -> SomaResult:
        """
        Acepta el payload tal como llega del modelo (un dict con type, command, args)
        y delega al backend.
        """
        try:
            command = SomaCommand.from_dict(payload)
        except Exception as exc:
            return SomaResult(success=False, error=f"invalid soma command: {exc}")
        try:
            return self.backend.execute(command)
        except Exception as exc:
            return SomaResult(success=False, error=f"backend raised: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# SomaCommandTool — wrapper para el Tool system
# ─────────────────────────────────────────────────────────────────────────────


class SomaCommandTool:
    """
    Tool para el ToolExecutor que delega a un SomaInterface.

    Se registra como `soma_command` en el registry. La acción del modelo es:
        [TOOL: {"action":"soma_command",
                "input":{"type":"high_level","command":"open_file","args":{...}}}]
    """

    name        = "soma_command"
    description = "Emite un comando SOMA (primitive/high_level/goal)."

    def __init__(self, interface: Optional[SomaInterface] = None) -> None:
        self.interface = interface or SomaInterface()

    def run(self, args: Dict[str, Any]):
        # Importación local para evitar circularidad con agent.tools
        from agent.tools import ToolResult
        if not isinstance(args, dict) or not args:
            return ToolResult("", "empty soma payload", 1, self.name)
        result = self.interface.execute(args)
        if result.success:
            return ToolResult(result.output, "", 0, self.name)
        return ToolResult("", result.error, 1, self.name)


__all__ = [
    "SomaCommandType",
    "SomaCommand",
    "SomaResult",
    "MockSomaBackend",
    "SomaInterface",
    "SomaCommandTool",
    "KNOWN_PRIMITIVES",
    "KNOWN_HIGH_LEVEL",
]
