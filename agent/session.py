"""
agent/session.py — AgentSession: historial de una sesión de agente
==================================================================

AgentSession registra:
  - Acciones ejecutadas en orden (tool_name + args + result)
  - Archivos vistos / leídos
  - Parches intentados (patches)
  - Errores encontrados

El historial es inmutable una vez añadido (append-only).
AgentSession NO depende del motor ni de herramientas: es un contenedor de datos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# ACTION ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ActionEntry:
    """
    Registro de una acción ejecutada durante la sesión.

    turn:      Número de turno (empieza en 1).
    tool_name: Nombre de la herramienta usada.
    args:      Argumentos pasados a la herramienta.
    result:    Resultado de la ejecución (ToolResult o cualquier objeto).
    motor_reasoning: Texto de razonamiento del motor (si disponible).
    """
    turn:             int
    tool_name:        str
    args:             Dict[str, Any]
    result:           Any
    motor_reasoning:  str = ""


# ─────────────────────────────────────────────────────────────────────────────
# SESSION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentSession:
    """
    Historial completo de una sesión del agente.

    Atributos:
        task:            Tarea original del usuario.
        actions:         Lista ordenada de ActionEntry (append-only).
        seen_files:      Set de rutas de archivos leídos durante la sesión.
        attempted_patches: Lista de parches (strings) intentados.
        errors:          Lista de mensajes de error encontrados.
        metadata:        Diccionario libre para datos adicionales.

    Uso:
        session = AgentSession(task="Fix the bug in training_utils.py")
        session.record_action(turn=1, tool_name="cat", args={"path": "..."}, result=r)
        session.record_file_seen("training_utils.py")
    """

    task:               str
    actions:            List[ActionEntry]     = field(default_factory=list)
    seen_files:         List[str]             = field(default_factory=list)
    attempted_patches:  List[str]             = field(default_factory=list)
    errors:             List[str]             = field(default_factory=list)
    metadata:           Dict[str, Any]        = field(default_factory=dict)

    # ── Registro de acciones ──────────────────────────────────────────────────

    def record_action(
        self,
        turn:            int,
        tool_name:       str,
        args:            Dict[str, Any],
        result:          Any,
        motor_reasoning: str = "",
    ) -> ActionEntry:
        """
        Registra una acción en el historial.

        Returns:
            El ActionEntry creado.
        """
        entry = ActionEntry(
            turn            = turn,
            tool_name       = tool_name,
            args            = args,
            result          = result,
            motor_reasoning = motor_reasoning,
        )
        self.actions.append(entry)
        return entry

    def record_file_seen(self, path: str) -> None:
        """Registra un archivo como visto (sin duplicados)."""
        if path not in self.seen_files:
            self.seen_files.append(path)

    def record_patch(self, patch: str) -> None:
        """Registra un parche intentado."""
        self.attempted_patches.append(patch)

    def record_error(self, error: str) -> None:
        """Registra un mensaje de error."""
        self.errors.append(error)

    # ── Consultas ─────────────────────────────────────────────────────────────

    @property
    def n_turns(self) -> int:
        """Número de acciones registradas."""
        return len(self.actions)

    def actions_for_turn(self, turn: int) -> List[ActionEntry]:
        """Todas las acciones del turno dado."""
        return [a for a in self.actions if a.turn == turn]

    def last_action(self) -> Optional[ActionEntry]:
        """La acción más reciente, o None si no hay ninguna."""
        return self.actions[-1] if self.actions else None

    def last_result(self) -> Optional[Any]:
        """El resultado de la acción más reciente, o None."""
        action = self.last_action()
        return action.result if action else None

    def tool_calls_count(self) -> Dict[str, int]:
        """Cuántas veces se llamó a cada herramienta."""
        counts: Dict[str, int] = {}
        for a in self.actions:
            counts[a.tool_name] = counts.get(a.tool_name, 0) + 1
        return counts

    def has_errors(self) -> bool:
        """True si se registraron errores."""
        return len(self.errors) > 0

    # ── Representación ────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Resumen compacto de la sesión."""
        return (
            f"AgentSession(task={self.task!r:.40s}, "
            f"turns={self.n_turns}, "
            f"files={len(self.seen_files)}, "
            f"errors={len(self.errors)})"
        )

    def __repr__(self) -> str:
        return self.summary()
