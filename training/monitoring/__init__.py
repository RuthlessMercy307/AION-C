"""
training/monitoring/ — Sistema de monitoring para training sequential.

Componentes:
    MetricsLogger  — append-only JSONL con métricas por step
    ControlFile    — bidireccional entre training y agente externo
    Watchdog       — thread bg que detecta condiciones críticas
    MonitoringContext — agrega los tres, se pasa al trainer

Uso típico:

    ctx = create_monitoring_context(
        log_dir=Path("training/logs/sequential_20260411_100000"),
        enable_watchdog=True,
    )

    trainer = SequentialTrainer(pipeline, cfg, monitoring=ctx)
    trainer.run_phase_1_backbone(data_fn, n_steps=1500)

    ctx.close()
"""

from training.monitoring.logger import MetricsLogger, MetricEntry
from training.monitoring.control import ControlFile, ControlAction
from training.monitoring.watchdog import Watchdog, WatchdogConfig
from training.monitoring.context import MonitoringContext, create_monitoring_context

__all__ = [
    "MetricsLogger",
    "MetricEntry",
    "ControlFile",
    "ControlAction",
    "Watchdog",
    "WatchdogConfig",
    "MonitoringContext",
    "create_monitoring_context",
]
