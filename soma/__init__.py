"""
soma/ — Interfaz SOMA (Parte 10 del MEGA-PROMPT)
=================================================

SOMA = Simultaneous Observation, Memory & Action.

SOMA es el "cuerpo" futuro de AION-C. Esta carpeta NO implementa SOMA
todavía — implementa la INTERFAZ que AION-C usa para emitir comandos
que SOMA entendería.

API expuesta:
    SomaCommand           — un comando estructurado
    SomaCommandType       — primitive | high_level | goal
    SomaInterface         — backend mockeable que ejecuta comandos
    SomaCommandTool       — wrapper como tool para el ToolExecutor
"""

from .interface import (
    SomaCommand, SomaCommandType, SomaResult,
    SomaInterface, MockSomaBackend, SomaCommandTool,
)

__all__ = [
    "SomaCommand",
    "SomaCommandType",
    "SomaResult",
    "SomaInterface",
    "MockSomaBackend",
    "SomaCommandTool",
]
