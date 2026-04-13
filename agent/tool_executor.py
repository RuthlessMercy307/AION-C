"""
agent/tool_executor.py — Tool execution pipeline (Parte 4.2 del MEGA-PROMPT)
=============================================================================

Implementa el pipeline 4.2 del plan:

  1. El usuario hace un query.
  2. AION-C decide si necesita una tool (o responde directo).
  3. Si necesita: genera JSON de tool call con formato:
        [TOOL: {"action": "...", "input": {...}}]
  4. ToolExecutor parsea el bloque, valida la action contra el registry,
     verifica sandbox/whitelist y ejecuta la herramienta.
  5. El resultado se inyecta como contexto:
        [RESULT: <salida de la herramienta>]
  6. AION-C genera la respuesta final con el resultado disponible.

Diseño:
  - El parser tolera múltiples bloques [TOOL: ...] en una misma respuesta y
    extrae el JSON de cada uno usando un balanceo de llaves para soportar
    objetos anidados (cuerpos con dicts, headers con dicts, etc).
  - El executor mantiene una lista negra de actions desconocidas y registra
    cada ejecución en una traza (`history`) para inspección y debugging.
  - El sandbox vive en `agent/tools.py` (`_validate_write_path`) y en cada
    tool individual (CallApiTool con whitelist, RunCodeTool con timeout).
    Aquí solo orquestamos.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .tools import BaseTool, ToolResult


# ─────────────────────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCall:
    """Una invocación de tool extraída del output del modelo."""
    action: str
    input:  Any  # típicamente dict, pero los tools aceptan strings simples también
    raw:    str = ""  # texto bruto del bloque [TOOL: ...]

    def to_args(self) -> Dict[str, Any]:
        """Convierte `input` a la forma `args: dict` que esperan las tools."""
        if isinstance(self.input, dict):
            return dict(self.input)
        # Algunas tools aceptan strings: read_file/search_web reciben "input": "ruta"
        # Mapeamos a la convención más común para cada action
        if isinstance(self.input, str):
            simple_string_keys = {
                "read_file":   "path",
                "file_read":   "path",
                "search_web":  "query",
                "web_search":  "query",
                "search_mem":  "query",
            }
            key = simple_string_keys.get(self.action, "input")
            return {key: self.input}
        return {"input": self.input}


@dataclass
class ToolExecutionRecord:
    """Una entrada del historial de ejecución."""
    call:    ToolCall
    result:  ToolResult
    success: bool


# ─────────────────────────────────────────────────────────────────────────────
# PARSER
# ─────────────────────────────────────────────────────────────────────────────


_TOOL_TAG_RE = re.compile(r"\[TOOL\s*:\s*", re.IGNORECASE)


def _extract_balanced_json(text: str, start: int) -> Tuple[Optional[str], int]:
    """
    Desde `text[start]` (que debe ser '{'), encuentra la llave de cierre
    correspondiente respetando strings y escapes. Devuelve (json_str, end_idx)
    o (None, start) si está mal formado.
    """
    if start >= len(text) or text[start] != "{":
        return None, start
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1], i + 1
    return None, start


def parse_tool_calls(text: str) -> List[ToolCall]:
    """
    Extrae todos los bloques [TOOL: {...}] del texto del modelo.

    Tolerante a:
      - Mayúsculas/minúsculas en `[TOOL:`
      - Espacios variables alrededor de `:` y dentro del JSON
      - JSON con objetos anidados (input: {path: ..., content: ...})
      - Bloques múltiples en la misma respuesta
      - Texto basura entre bloques (se ignora)

    Devuelve una lista de ToolCall (vacía si no hay bloques válidos).
    """
    calls: List[ToolCall] = []
    if not text:
        return calls

    pos = 0
    while True:
        match = _TOOL_TAG_RE.search(text, pos)
        if not match:
            break
        # Saltar espacios después de `[TOOL:`
        json_start = match.end()
        while json_start < len(text) and text[json_start].isspace():
            json_start += 1
        if json_start >= len(text) or text[json_start] != "{":
            pos = match.end()
            continue
        json_str, json_end = _extract_balanced_json(text, json_start)
        if json_str is None:
            pos = match.end()
            continue
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            pos = json_end
            continue
        if not isinstance(data, dict) or "action" not in data:
            pos = json_end
            continue
        # Buscar el ']' de cierre opcional
        end_bracket = json_end
        while end_bracket < len(text) and text[end_bracket].isspace():
            end_bracket += 1
        if end_bracket < len(text) and text[end_bracket] == "]":
            end_bracket += 1
        raw = text[match.start():end_bracket]
        calls.append(ToolCall(
            action=str(data["action"]),
            input=data.get("input"),
            raw=raw,
        ))
        pos = end_bracket
    return calls


def format_result(result: ToolResult) -> str:
    """Devuelve el bloque `[RESULT: ...]` correspondiente a un ToolResult."""
    return f"[RESULT: {result.as_text()}]"


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTOR
# ─────────────────────────────────────────────────────────────────────────────


class ToolExecutor:
    """
    Orquesta el pipeline parse → dispatch → execute → record.

    Uso:
        from agent.tools import build_tool_registry
        from agent.tool_executor import ToolExecutor

        registry = build_tool_registry(output_root=Path("/output"))
        executor = ToolExecutor(registry)

        records = executor.run_from_text(model_output)
        context = executor.format_context(records)
        # 'context' se inyecta como `[RESULT: ...]` para la siguiente generación

    Atributos:
        registry: dict {name: BaseTool}
        history:  lista de ToolExecutionRecord (todas las ejecuciones)
    """

    def __init__(self, registry: Dict[str, BaseTool]) -> None:
        self.registry: Dict[str, BaseTool] = dict(registry)
        self.history: List[ToolExecutionRecord] = []

    # ── ejecución individual ────────────────────────────────────────────

    def execute(self, call: ToolCall) -> ToolResult:
        """Ejecuta una sola tool call. Devuelve el ToolResult."""
        tool = self.registry.get(call.action)
        if tool is None:
            result = ToolResult(
                stdout="",
                stderr=f"unknown action: {call.action}",
                exit_code=1,
                tool_name=call.action,
            )
        else:
            try:
                result = tool.run(call.to_args())
            except Exception as exc:
                result = ToolResult(
                    stdout="",
                    stderr=f"tool raised: {exc}",
                    exit_code=1,
                    tool_name=call.action,
                )
        record = ToolExecutionRecord(call=call, result=result, success=result.ok)
        self.history.append(record)
        return result

    # ── ejecución desde texto del modelo ────────────────────────────────

    def run_from_text(self, text: str) -> List[ToolExecutionRecord]:
        """
        Parsea bloques [TOOL: ...] del texto y los ejecuta en orden.
        Devuelve los records de esta llamada (también se acumulan en history).
        """
        calls = parse_tool_calls(text)
        records: List[ToolExecutionRecord] = []
        for call in calls:
            self.execute(call)
            records.append(self.history[-1])
        return records

    # ── formato de contexto para inyectar en la siguiente generación ────

    def format_context(self, records: List[ToolExecutionRecord]) -> str:
        """Concatena los `[RESULT: ...]` de una lista de records."""
        return "\n".join(format_result(r.result) for r in records)

    def reset(self) -> None:
        """Vacía el historial."""
        self.history.clear()


__all__ = [
    "ToolCall",
    "ToolExecutionRecord",
    "ToolExecutor",
    "parse_tool_calls",
    "format_result",
]
