"""
agent/self_check.py — Self-check, confidence y error log (Parte 7 del MEGA-PROMPT)
=====================================================================================

7.1  SELF-CHECK
     Antes de responder, una serie de checks declarativos verifican:
       - Coherencia con el query (longitud razonable, no vacío, no eco del prompt)
       - Sintaxis del código (compile() en Python; brackets balanceados en otros)
       - Consistencia numérica (no contradice cifras del query)
     Si alguno falla → la API SelfCheckResult.passed es False y devuelve los
     issues. El llamador decide si re-genera (max 2 intentos por convención).

7.2  CONFIDENCE SCORE
     - Calculado a partir de probabilidades token-by-token (mean de los
       primeros K tokens generados, normalmente K=5).
     - 3 niveles: HIGH (>=0.8), MEDIUM (0.5-0.8), LOW (<0.5).
     - Política asociada (`policy_for_confidence`):
         HIGH   → respond_directly
         MEDIUM → respond_with_disclaimer
         LOW    → search_then_respond  (busca en MEM/internet o pregunta)

7.3  ERROR LOG EN MEM
     ErrorLog.record(error, cause, prevention, domain) escribe una entrada
     estructurada en MEM con domain="error_log". Búsqueda por dominio recupera
     errores previos del mismo motor para no repetirlos.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE
# ─────────────────────────────────────────────────────────────────────────────


HIGH_CONFIDENCE_THRESHOLD = 0.80
LOW_CONFIDENCE_THRESHOLD  = 0.50


class ConfidenceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


def confidence_from_probs(probs: Sequence[float], window: int = 5) -> float:
    """
    Calcula confidence como la media de las primeras `window` probabilidades.
    Cada elemento de `probs` debe ser un float en [0, 1] (típicamente el
    softmax max del paso de generación).
    """
    if not probs:
        return 0.0
    head = list(probs)[:window]
    return sum(head) / len(head)


def classify_confidence(score: float) -> ConfidenceLevel:
    if score >= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    if score >= LOW_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW


def policy_for_confidence(level: ConfidenceLevel) -> str:
    """Devuelve la acción recomendada según el nivel de confianza."""
    return {
        ConfidenceLevel.HIGH:   "respond_directly",
        ConfidenceLevel.MEDIUM: "respond_with_disclaimer",
        ConfidenceLevel.LOW:    "search_then_respond",
    }[level]


# ─────────────────────────────────────────────────────────────────────────────
# SELF-CHECK
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SelfCheckResult:
    """Resultado de un self-check."""
    passed: bool
    issues: List[str] = field(default_factory=list)
    confidence:       Optional[float] = None
    confidence_level: Optional[ConfidenceLevel] = None

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False


# patrones para detectar números en query
_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _extract_numbers(text: str) -> List[float]:
    out: List[float] = []
    for m in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", ".")))
        except ValueError:
            continue
    return out


def _looks_like_python_code(text: str) -> bool:
    if not text:
        return False
    markers = ("def ", "class ", "import ", "from ", "return ", "print(", "lambda ", "with ")
    return any(m in text for m in markers)


def _check_python_syntax(code: str) -> Optional[str]:
    """Devuelve un mensaje de error si la sintaxis es inválida; None si OK."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as exc:
        return f"python syntax: {exc.msg}"


def _check_brackets_balanced(text: str) -> bool:
    """True si paréntesis, brackets y llaves están balanceados."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: List[str] = []
    in_str: Optional[str] = None
    esc = False
    for c in text:
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == in_str:
                in_str = None
            continue
        if c in ("'", '"'):
            in_str = c
            continue
        if c in "([{":
            stack.append(c)
        elif c in ")]}":
            if not stack or stack[-1] != pairs[c]:
                return False
            stack.pop()
    return not stack and in_str is None


class SelfChecker:
    """
    Ejecuta una batería de checks sobre una respuesta antes de devolverla.

    Uso:
        checker = SelfChecker()
        result = checker.check(query, response, probs=[0.9, 0.8, 0.7])
        if not result.passed:
            ... regenerate or escalate ...
    """

    def __init__(
        self,
        min_response_length: int = 1,
        max_response_length: int = 4000,
        check_code_syntax:  bool = True,
        check_numerics:     bool = True,
    ) -> None:
        self.min_response_length = min_response_length
        self.max_response_length = max_response_length
        self.check_code_syntax = check_code_syntax
        self.check_numerics = check_numerics

    def check(
        self,
        query: str,
        response: str,
        probs: Optional[Sequence[float]] = None,
    ) -> SelfCheckResult:
        result = SelfCheckResult(passed=True)

        # 1) longitud
        n = len(response or "")
        if n < self.min_response_length:
            result.add_issue("response too short or empty")
        if n > self.max_response_length:
            result.add_issue(f"response too long ({n} > {self.max_response_length})")

        # 2) eco del prompt (la respuesta es idéntica al query)
        if query and response and response.strip().lower() == query.strip().lower():
            result.add_issue("response is exact echo of query")

        # 3) sintaxis de código si parece código
        if self.check_code_syntax and response:
            if _looks_like_python_code(response):
                err = _check_python_syntax(response)
                if err is not None:
                    result.add_issue(err)
            if not _check_brackets_balanced(response):
                result.add_issue("unbalanced brackets")

        # 4) consistencia numérica
        if self.check_numerics and query and response:
            qnums = set(_extract_numbers(query))
            if qnums:
                rnums = set(_extract_numbers(response))
                # Si el query tiene números pero la respuesta tiene OTROS números
                # que NO incluyen ninguno del query, marcamos como sospechoso.
                if rnums and not (qnums & rnums) and len(rnums) >= 2:
                    # Excepción: si la respuesta tiene exactamente un número
                    # (probablemente el resultado), no lo marcamos.
                    result.add_issue("numeric inconsistency: query numbers absent from response")

        # 5) confidence
        if probs is not None:
            score = confidence_from_probs(probs)
            result.confidence = score
            result.confidence_level = classify_confidence(score)
            # confidence muy bajo NO marca passed=False — el llamador decide
            # qué política aplicar (search_then_respond, etc).

        return result


# ─────────────────────────────────────────────────────────────────────────────
# ERROR LOG
# ─────────────────────────────────────────────────────────────────────────────


ERROR_LOG_DOMAIN = "error_log"


@dataclass
class ErrorRecord:
    error:      str
    cause:      str
    prevention: str
    domain:     str
    timestamp:  float = field(default_factory=time.time)

    def to_text(self) -> str:
        return (
            f"error: {self.error} | cause: {self.cause} | "
            f"prevention: {self.prevention}"
        )


class ErrorLog:
    """
    Persiste errores en MEM con domain="error_log" para no repetirlos.

    Uso:
        log = ErrorLog(mem)
        log.record(
            error="generated code with syntax error",
            cause="missing closing parenthesis on line 15",
            prevention="always verify bracket balance",
            domain="forge_c",
        )
        previous = log.recall(domain="forge_c")
    """

    def __init__(self, mem: Any = None) -> None:
        self.mem = mem
        self._local: List[ErrorRecord] = []  # fallback si no hay MEM

    def record(self, error: str, cause: str, prevention: str, domain: str) -> ErrorRecord:
        rec = ErrorRecord(error=error, cause=cause, prevention=prevention, domain=domain)
        self._local.append(rec)
        if self.mem is not None:
            try:
                key = f"err_{int(rec.timestamp * 1000)}_{domain}"
                self.mem.store(key, rec.to_text(), domain=ERROR_LOG_DOMAIN)
            except Exception:
                pass
        return rec

    def recall(self, query: str = "", domain: Optional[str] = None, top_k: int = 5) -> List[ErrorRecord]:
        """
        Devuelve errores previos relevantes. Si MEM está disponible, busca por
        similitud filtrando por domain="error_log". Como fallback usa el
        historial local en memoria.
        """
        if self.mem is not None:
            try:
                results = self.mem.search(query or "error", top_k=top_k, domain=ERROR_LOG_DOMAIN)
            except TypeError:
                results = self.mem.search(query or "error", top_k=top_k)
            except Exception:
                results = []
            recs: List[ErrorRecord] = []
            for item in results or []:
                if isinstance(item, tuple) and len(item) >= 2:
                    text = item[1]
                    parts = {p.split(":", 1)[0].strip(): p.split(":", 1)[1].strip()
                             for p in text.split(" | ") if ":" in p}
                    recs.append(ErrorRecord(
                        error=parts.get("error", ""),
                        cause=parts.get("cause", ""),
                        prevention=parts.get("prevention", ""),
                        domain=domain or "",
                    ))
            return recs
        # fallback local
        if domain:
            return [r for r in self._local if r.domain == domain][:top_k]
        return list(self._local[:top_k])

    def __len__(self) -> int:
        return len(self._local)


__all__ = [
    "HIGH_CONFIDENCE_THRESHOLD",
    "LOW_CONFIDENCE_THRESHOLD",
    "ConfidenceLevel",
    "confidence_from_probs",
    "classify_confidence",
    "policy_for_confidence",
    "SelfCheckResult",
    "SelfChecker",
    "ErrorRecord",
    "ErrorLog",
    "ERROR_LOG_DOMAIN",
]
