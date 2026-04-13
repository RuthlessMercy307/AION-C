"""
reward/reward.py — Reward probabilístico con 3 fuentes (Parte 25).

Tres componentes, combinados por fórmula ponderada:

    reward_mean = α·R_explicit + β·R_implicit + γ·R_intrinsic

Cada componente aporta una media y una varianza; la final es media
ponderada (y varianza combinada asumiendo independencia entre componentes).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════════════════
# Explicit (thumbs / correcciones)
# ════════════════════════════════════════════════════════════════════════════

class ExplicitSignal(str, Enum):
    UP = "up"            # thumbs up / confirmación
    DOWN = "down"        # thumbs down
    CORRECTION = "correction"  # corrección directa
    NONE = "none"        # sin señal


def _explicit_value(signal: ExplicitSignal) -> Tuple[float, float]:
    """Devuelve (mean, std) para una señal explícita."""
    if signal == ExplicitSignal.UP:
        return 1.0, 0.05
    if signal == ExplicitSignal.DOWN:
        return 0.0, 0.05
    if signal == ExplicitSignal.CORRECTION:
        return 0.1, 0.10
    return 0.5, 0.35  # NONE — alta varianza, poco informativo


# ════════════════════════════════════════════════════════════════════════════
# Implícitas
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ImplicitSignals:
    """Señales implícitas inferidas del flujo de conversación.

    no_correction_continue:  True si el usuario siguió avanzando sin corregir
                             (señal positiva FUERTE — más que 'gracias')
    thanks:                  "gracias" / "thanks" en el siguiente turno
    re_asked_similar:        re-pregunta reformulada (señal NEGATIVA)
    code_copied:             evidencia de que copiaron código/texto (pos fuerte)
    abandoned:               no hubo próximo turno en un tiempo razonable
    """
    no_correction_continue: bool = False
    thanks: bool = False
    re_asked_similar: bool = False
    code_copied: bool = False
    abandoned: bool = False


# ════════════════════════════════════════════════════════════════════════════
# Intrínsecas
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class IntrinsicSignals:
    """Señales de confianza interna del modelo.

    token_entropy_mean: [0, +inf) — menor es mejor. Se mapea a [0,1] invertida.
    symbolic_consistent: True si la salida no contradice AXIOM/FORGE-C/CORA.
    unifier_agreement: [0, 1] — acuerdo entre los motores en el unifier.
    """
    token_entropy_mean: float = 0.0
    symbolic_consistent: bool = True
    unifier_agreement: float = 1.0

    def to_mean_std(self) -> Tuple[float, float]:
        # entropy: escalada por log(5) (5 motores de MoSE como referencia)
        ent_scaled = max(0.0, min(1.0, 1.0 - self.token_entropy_mean / math.log(5)))
        sym = 1.0 if self.symbolic_consistent else 0.0
        ua = max(0.0, min(1.0, self.unifier_agreement))
        mean = 0.35 * ent_scaled + 0.4 * sym + 0.25 * ua
        # Varianza fija — esta señal es siempre más ruidosa que la explícita.
        return mean, 0.15


# ════════════════════════════════════════════════════════════════════════════
# Combinación
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class RewardSignals:
    explicit: ExplicitSignal = ExplicitSignal.NONE
    implicit: ImplicitSignals = field(default_factory=ImplicitSignals)
    intrinsic: IntrinsicSignals = field(default_factory=IntrinsicSignals)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explicit": self.explicit.value,
            "implicit": asdict(self.implicit),
            "intrinsic": asdict(self.intrinsic),
        }


@dataclass
class RewardConfig:
    """Pesos de las 3 componentes.

    alpha es el más alto porque las señales explícitas son las más fiables.
    beta es moderado — hay volumen pero ruido.
    gamma es bajo — regularizador intrínseco.
    """
    alpha_explicit: float = 0.55
    beta_implicit: float = 0.30
    gamma_intrinsic: float = 0.15

    # Pesos internos para las señales implícitas. Nótese:
    #   no_correction_continue pesa MÁS que thanks (por diseño del MEGA-PROMPT)
    #   re_asked_similar es FUERTEMENTE negativo
    w_no_correction: float = 0.40
    w_thanks: float = 0.15
    w_re_asked: float = 0.35
    w_code_copied: float = 0.30
    w_abandoned: float = 0.10

    # Cuán conservadora debe ser la decisión final: conservative_reward = mean - k·std
    k_std: float = 1.0

    def __post_init__(self) -> None:
        total = self.alpha_explicit + self.beta_implicit + self.gamma_intrinsic
        if total <= 0:
            raise ValueError("weights must sum to > 0")
        if self.k_std < 0:
            raise ValueError("k_std must be >= 0")


@dataclass
class RewardEstimate:
    mean: float
    std: float
    conservative: float
    components: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RewardEstimator:
    """Convierte RewardSignals → RewardEstimate (prob [0,1] con varianza)."""

    def __init__(self, config: Optional[RewardConfig] = None) -> None:
        self.config = config or RewardConfig()

    def compute(self, signals: RewardSignals) -> RewardEstimate:
        c = self.config
        e_mean, e_std = _explicit_value(signals.explicit)
        i_mean, i_std = self._implicit_score(signals.implicit)
        x_mean, x_std = signals.intrinsic.to_mean_std()

        total_w = c.alpha_explicit + c.beta_implicit + c.gamma_intrinsic
        mean = (
            c.alpha_explicit * e_mean +
            c.beta_implicit * i_mean +
            c.gamma_intrinsic * x_mean
        ) / total_w
        # Varianza combinada con pesos normalizados, independencia asumida.
        a = c.alpha_explicit / total_w
        b = c.beta_implicit / total_w
        g = c.gamma_intrinsic / total_w
        var = (a * a) * (e_std ** 2) + (b * b) * (i_std ** 2) + (g * g) * (x_std ** 2)
        std = math.sqrt(max(var, 0.0))
        conservative = max(0.0, min(1.0, mean - c.k_std * std))
        return RewardEstimate(
            mean=max(0.0, min(1.0, mean)),
            std=std,
            conservative=conservative,
            components={
                "explicit": e_mean,
                "implicit": i_mean,
                "intrinsic": x_mean,
            },
        )

    def _implicit_score(self, s: ImplicitSignals) -> Tuple[float, float]:
        c = self.config
        score = 0.5  # baseline neutral
        if s.no_correction_continue:
            score += c.w_no_correction
        if s.thanks:
            score += c.w_thanks
        if s.re_asked_similar:
            score -= c.w_re_asked
        if s.code_copied:
            score += c.w_code_copied
        if s.abandoned:
            score -= c.w_abandoned
        score = max(0.0, min(1.0, score))
        # Varianza alta cuando no hay NINGUNA señal; baja cuando hay varias.
        any_signal = any([
            s.no_correction_continue, s.thanks,
            s.re_asked_similar, s.code_copied, s.abandoned,
        ])
        std = 0.10 if any_signal else 0.30
        return score, std


# ════════════════════════════════════════════════════════════════════════════
# ImplicitDetector — heurísticas de detección
# ════════════════════════════════════════════════════════════════════════════

_THANKS_RE = re.compile(
    r"\b(gracias|thanks|thank you|thx|ty|genial|perfecto|excelente)\b",
    re.IGNORECASE,
)
_CODE_BLOCK_RE = re.compile(r"```[\w]*\n[\s\S]+?```")


class ImplicitDetector:
    """Detecta señales implícitas entre un turno del asistente y el siguiente.

    API:
        detect(assistant_response, next_user_text, previous_user_text,
               time_to_next_turn_sec) → ImplicitSignals
    """

    def __init__(
        self,
        abandon_threshold_sec: float = 300.0,
        similarity_threshold: float = 0.6,
    ) -> None:
        self.abandon_threshold_sec = abandon_threshold_sec
        self.similarity_threshold = similarity_threshold

    def detect(
        self,
        assistant_response: str,
        next_user_text: Optional[str],
        previous_user_text: str,
        time_to_next_turn_sec: Optional[float] = None,
    ) -> ImplicitSignals:
        # Abandonment: no next turn or took too long
        if next_user_text is None:
            abandoned = time_to_next_turn_sec is None or time_to_next_turn_sec > self.abandon_threshold_sec
            return ImplicitSignals(abandoned=abandoned)

        nxt_low = next_user_text.lower()
        thanks = bool(_THANKS_RE.search(nxt_low))
        # Re-asked: similarity entre previous_user_text y next_user_text alto
        re_asked = self._similarity(previous_user_text, next_user_text) >= self.similarity_threshold
        # No correction continue: el próximo turno NO contiene palabras de
        # corrección ni re-ask.
        correction_markers = ("no,", "incorrect", "equivocado", "mal ", "está mal", "wrong", "actually")
        has_correction = any(m in nxt_low for m in correction_markers)
        no_correction = not has_correction and not re_asked
        # Code copied: heurística imperfecta — si el assistant emitió un
        # bloque de código y el user en el próximo turno NO dice nada de
        # error, contamos code_copied como plausible.
        has_code = bool(_CODE_BLOCK_RE.search(assistant_response))
        code_copied = has_code and not has_correction
        return ImplicitSignals(
            no_correction_continue=no_correction,
            thanks=thanks,
            re_asked_similar=re_asked,
            code_copied=code_copied,
            abandoned=False,
        )

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Jaccard de tokens — suficiente como heurística de re-pregunta."""
        ta = set(a.lower().split())
        tb = set(b.lower().split())
        if not ta or not tb:
            return 0.0
        return len(ta & tb) / len(ta | tb)


# ════════════════════════════════════════════════════════════════════════════
# RewardLedger — stats por (motor, adapter)
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class _LedgerEntry:
    sum_mean: float = 0.0
    sum_var: float = 0.0
    n: int = 0

    def update(self, estimate: RewardEstimate) -> None:
        self.sum_mean += estimate.mean
        self.sum_var += estimate.std ** 2
        self.n += 1

    def mean(self) -> float:
        return self.sum_mean / max(self.n, 1)

    def std(self) -> float:
        if self.n <= 0:
            return 0.0
        return math.sqrt(self.sum_var / self.n)


class RewardLedger:
    """Acumula reward por clave (motor o motor:adapter)."""

    def __init__(self) -> None:
        self._entries: Dict[str, _LedgerEntry] = {}

    def add(self, key: str, estimate: RewardEstimate) -> None:
        e = self._entries.setdefault(key, _LedgerEntry())
        e.update(estimate)

    def mean_for(self, key: str) -> float:
        e = self._entries.get(key)
        return e.mean() if e is not None else 0.0

    def count_for(self, key: str) -> int:
        e = self._entries.get(key)
        return e.n if e is not None else 0

    def snapshot(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {"mean": e.mean(), "std": e.std(), "n": e.n}
            for k, e in self._entries.items()
        }

    def keys(self) -> List[str]:
        return list(self._entries.keys())

    # ── Persistencia JSONL ────────────────────────────────────────────────
    def save_jsonl(self, path) -> None:
        """Serializa el ledger a un archivo JSONL (una línea por key).

        Formato:
            {"key": "forge_c", "sum_mean": 3.4, "sum_var": 0.02, "n": 5}
        """
        import json
        from pathlib import Path as _Path
        p = _Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for key, entry in self._entries.items():
            lines.append(json.dumps({
                "key": key,
                "sum_mean": entry.sum_mean,
                "sum_var": entry.sum_var,
                "n": entry.n,
            }))
        p.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def load_jsonl(self, path) -> None:
        """Reemplaza el estado actual con el contenido de un JSONL.

        Es no-op si el archivo no existe.
        """
        import json
        from pathlib import Path as _Path
        p = _Path(path)
        if not p.exists():
            return
        self._entries.clear()
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            self._entries[d["key"]] = _LedgerEntry(
                sum_mean=float(d["sum_mean"]),
                sum_var=float(d["sum_var"]),
                n=int(d["n"]),
            )


# ════════════════════════════════════════════════════════════════════════════
# Sleep cycle adaptador
# ════════════════════════════════════════════════════════════════════════════

def sleep_reward_hook(
    estimator: Optional[RewardEstimator] = None,
    detector: Optional[ImplicitDetector] = None,
):
    """Devuelve un reward_hook compatible con SleepCycle (Parte 23).

    Convierte cada episodio en RewardSignals y deja los scores en el
    formato esperado por el stub de SleepCycle (dict index→float).
    """
    est = estimator or RewardEstimator()
    det = detector or ImplicitDetector()

    def _hook(episodes):
        scores: Dict[int, float] = {}
        detailed: Dict[int, Dict[str, Any]] = {}
        for i, ep in enumerate(episodes):
            explicit = ExplicitSignal.NONE
            if ep.user_feedback == "up":
                explicit = ExplicitSignal.UP
            elif ep.user_feedback == "down":
                explicit = ExplicitSignal.DOWN
            elif ep.user_feedback == "correction":
                explicit = ExplicitSignal.CORRECTION

            # Señales implícitas: sin siguiente turno aquí — pasamos ImplicitSignals() vacío.
            signals = RewardSignals(
                explicit=explicit,
                implicit=ImplicitSignals(),
                intrinsic=IntrinsicSignals(),
            )
            estimate = est.compute(signals)
            scores[i] = estimate.mean
            detailed[i] = estimate.to_dict()
        mean = (sum(scores.values()) / len(scores)) if scores else 0.0
        return {
            "scores": scores,
            "mean": mean,
            "detailed": detailed,
            "source": "reward_estimator",
        }

    return _hook
