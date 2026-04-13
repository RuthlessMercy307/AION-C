"""
Genera experiments/train_production.ipynb
"""
import json, uuid
from pathlib import Path

def _id():
    return uuid.uuid4().hex[:8]

def md(source):
    return {"cell_type":"markdown","id":_id(),"metadata":{},"source":source}

def code(source):
    source = source.lstrip("\n")
    lines = source.split("\n")
    arr = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type":"code","execution_count":None,"id":_id(),
            "metadata":{"cellView":"form"},"outputs":[],"source":arr}

HEADER = (
    "# AION-C \u2014 Training Notebook\n\n"
    "Entrena el modelo AION-C (Mixture-of-Specialists Encoder) en las 4 fases.\n\n"
    "**Flujo rapido (smoke test en T4, < 30 min):**\n"
    "1. Sube `AION-C.zip` (y opcionalmente `DataSet-Generator-Claude-Opus.zip`) al panel de archivos de Colab\n"
    "2. Ejecuta **Celda 1** (Setup)\n"
    "3. Ejecuta **Celdas 2-5** en orden con `CONFIG = \"tiny\"`\n"
    "4. Revisa el **Resumen** en Celda 6\n\n"
    "**Entrenamiento real en H200/Vast.ai:**\n"
    "- Selecciona `CONFIG = \"production\"` en cada celda\n"
    "- Monta Google Drive (`MOUNT_DRIVE = True`) para guardar checkpoints\n\n---\n"
)
