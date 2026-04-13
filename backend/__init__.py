"""
backend/ — FastAPI + WebSocket backend para AION-C (Fase D)
=============================================================

Componentes:
  app_fastapi.py — la app FastAPI con HTTP + WebSocket endpoints
  static/        — assets servidos directamente (index.html con React)

Endpoints:
  GET  /                 — sirve frontend React (CDN, sin build step)
  GET  /api/info         — model info, params, motors, version
  POST /api/session      — crea nueva sesión, devuelve session_id
  GET  /api/sessions     — lista sesiones
  GET  /api/session/{id} — historial de una sesión
  GET  /api/mem          — entradas de MEM
  POST /api/upload       — sube un archivo al sandbox /output/
  GET  /api/download/{p} — descarga un archivo del sandbox /output/
  WS   /ws/chat/{sid}    — chat streaming token-by-token

El módulo NO importa torch/modelo en su nivel superior — el modelo
se inyecta vía AppState al hacer create_app(model=...) para que los
tests puedan usar mocks.
"""

from .app_fastapi import (
    create_app, AppState,
    ChatStreamMessage, RoutingScores,
)

__all__ = [
    "create_app",
    "AppState",
    "ChatStreamMessage",
    "RoutingScores",
]
