"""
agent/tools.py — Herramientas disponibles para el AgentLoop
============================================================

Cada herramienta sigue la interfaz:
    tool.run(args: dict) -> ToolResult

ToolResult contiene stdout, stderr, exit_code y la representación
compacta que el motor ve como respuesta.

Herramientas:
  BashTool    — ejecuta comandos de shell
  GrepTool    — busca patrones en archivos
  FindTool    — encuentra archivos por nombre
  CatTool     — muestra contenido de archivos
  PytestTool  — ejecuta tests pytest

Todas son mockeable pasando `runner` (función externa) en el constructor.
El runner por defecto usa subprocess.run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# RESULT
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    """
    Resultado de ejecutar una herramienta.

    stdout:    Salida estándar (string).
    stderr:    Salida de error (string).
    exit_code: Código de retorno (0 = éxito).
    tool_name: Nombre de la herramienta que lo generó.
    """
    stdout:    str
    stderr:    str
    exit_code: int
    tool_name: str = ""

    @property
    def ok(self) -> bool:
        """True si exit_code == 0."""
        return self.exit_code == 0

    def as_text(self) -> str:
        """Representación compacta para el contexto del motor."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[stderr] {self.stderr}")
        if self.exit_code != 0:
            parts.append(f"[exit_code={self.exit_code}]")
        return "\n".join(parts) if parts else "(empty)"

    def __repr__(self) -> str:
        return (
            f"ToolResult(tool={self.tool_name!r}, ok={self.ok}, "
            f"stdout={self.stdout[:60]!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BASE TOOL
# ─────────────────────────────────────────────────────────────────────────────

class BaseTool:
    """
    Clase base para todas las herramientas.

    Subclases implementan:
        name:          str — identificador único de la herramienta
        description:   str — descripción para el motor
        run(args)      → ToolResult

    El runner opcional permite sustituir subprocess.run por un mock en tests.
    """

    name: str = "base"
    description: str = ""

    def __init__(self, runner: Optional[Callable[..., Any]] = None) -> None:
        self._runner = runner or self._default_runner

    @staticmethod
    def _default_runner(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            **kwargs,
        )

    def run(self, args: Dict[str, Any]) -> ToolResult:
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS CONCRETAS
# ─────────────────────────────────────────────────────────────────────────────

class BashTool(BaseTool):
    """
    Ejecuta un comando bash arbitrario.

    Args:
        command (str): comando a ejecutar en el shell

    Ejemplo:
        tool.run({"command": "ls -la"})
    """

    name        = "bash"
    description = "Ejecuta un comando de shell y retorna su salida."

    def run(self, args: Dict[str, Any]) -> ToolResult:
        command = args.get("command", "")
        try:
            result = self._runner(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return ToolResult(
                stdout    = result.stdout or "",
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("", "Timeout", 1, self.name)
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


class GrepTool(BaseTool):
    """
    Busca un patrón en archivos.

    Args:
        pattern (str): expresión regular a buscar
        path    (str): directorio o archivo donde buscar (default: ".")
        flags   (str): flags adicionales para grep (default: "-r")

    Ejemplo:
        tool.run({"pattern": "def train", "path": "experiments/"})
    """

    name        = "grep"
    description = "Busca un patrón regex en archivos del proyecto."

    def run(self, args: Dict[str, Any]) -> ToolResult:
        pattern = args.get("pattern", "")
        path    = args.get("path", ".")
        flags   = args.get("flags", "-r")
        cmd     = ["grep", flags, "--include=*.py", pattern, path]
        try:
            result = self._runner(cmd)
            return ToolResult(
                stdout    = result.stdout or "",
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


class FindTool(BaseTool):
    """
    Encuentra archivos por nombre o extensión.

    Args:
        name    (str): patrón de nombre (glob) — p.ej. "*.py"
        path    (str): directorio de búsqueda (default: ".")
        maxdepth(int): profundidad máxima (default: sin límite)

    Ejemplo:
        tool.run({"name": "test_*.py", "path": "tests/"})
    """

    name        = "find"
    description = "Encuentra archivos por nombre o extensión."

    def run(self, args: Dict[str, Any]) -> ToolResult:
        pattern  = args.get("name", "*")
        path     = args.get("path", ".")
        maxdepth = args.get("maxdepth", None)
        cmd = ["find", path, "-name", pattern]
        if maxdepth is not None:
            cmd += ["-maxdepth", str(maxdepth)]
        try:
            result = self._runner(cmd)
            return ToolResult(
                stdout    = result.stdout or "",
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


class CatTool(BaseTool):
    """
    Muestra el contenido de un archivo.

    Args:
        path  (str): ruta al archivo
        lines (int): máximo de líneas a mostrar (default: sin límite)

    Ejemplo:
        tool.run({"path": "experiments/training_utils.py", "lines": 30})
    """

    name        = "cat"
    description = "Lee y muestra el contenido de un archivo."

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path  = args.get("path", "")
        lines = args.get("lines", None)

        if not path:
            return ToolResult("", "No path provided", 1, self.name)

        cmd = ["cat", path]
        try:
            result = self._runner(cmd)
            stdout = result.stdout or ""
            if lines is not None and stdout:
                stdout = "\n".join(stdout.splitlines()[:lines])
            return ToolResult(
                stdout    = stdout,
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


class PytestTool(BaseTool):
    """
    Ejecuta tests pytest.

    Args:
        path    (str): archivo o directorio de tests (default: "tests/")
        flags   (str): flags adicionales (default: "-q")
        timeout (int): timeout en segundos (default: 60)

    Ejemplo:
        tool.run({"path": "tests/test_training_utils.py", "flags": "-v"})
    """

    name        = "pytest"
    description = "Ejecuta tests pytest y retorna el resumen."

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path    = args.get("path", "tests/")
        flags   = args.get("flags", "-q")
        timeout = args.get("timeout", 60)

        cmd = ["python", "-m", "pytest", path] + flags.split()
        try:
            result = self._runner(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return ToolResult(
                stdout    = result.stdout or "",
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except subprocess.TimeoutExpired:
            return ToolResult("", f"Timeout after {timeout}s", 1, self.name)
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS ADICIONALES: WEB + FILE
# ─────────────────────────────────────────────────────────────────────────────

class WebSearchTool(BaseTool):
    """
    Busca en la web usando un motor de búsqueda.

    Args:
        query   (str): consulta de búsqueda
        max_results (int): máximo de resultados (default: 5)

    Mockeable: pasa un runner que reciba (query, max_results) y retorne
    un CompletedProcess con stdout = JSON de resultados.
    """

    name        = "web_search"
    description = "Busca información en la web."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        search_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        super().__init__(runner)
        self._search_fn = search_fn  # callable(query, max_results) -> str

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query       = args.get("query", "")
        max_results = args.get("max_results", 5)

        if not query:
            return ToolResult("", "No query provided", 1, self.name)

        # Si hay función de búsqueda inyectada (mock o real), usarla
        if self._search_fn is not None:
            try:
                result_text = self._search_fn(query, max_results)
                return ToolResult(result_text, "", 0, self.name)
            except Exception as e:
                return ToolResult("", str(e), 1, self.name)

        # Default: intenta usar el runner (subprocess)
        try:
            result = self._runner(
                ["curl", "-s", f"https://api.duckduckgo.com/?q={query}&format=json"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            return ToolResult(
                stdout    = result.stdout or "(no results)",
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


class WebFetchTool(BaseTool):
    """
    Descarga el contenido de una URL.

    Args:
        url     (str): URL a descargar
        timeout (int): timeout en segundos (default: 15)

    Mockeable: pasa un fetch_fn que reciba (url) y retorne string.
    """

    name        = "web_fetch"
    description = "Descarga contenido de una URL."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        fetch_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        super().__init__(runner)
        self._fetch_fn = fetch_fn

    def run(self, args: Dict[str, Any]) -> ToolResult:
        url     = args.get("url", "")
        timeout = args.get("timeout", 15)

        if not url:
            return ToolResult("", "No URL provided", 1, self.name)

        if self._fetch_fn is not None:
            try:
                content = self._fetch_fn(url)
                return ToolResult(content, "", 0, self.name)
            except Exception as e:
                return ToolResult("", str(e), 1, self.name)

        try:
            result = self._runner(
                ["curl", "-sL", "--max-time", str(timeout), url],
                capture_output=True,
                text=True,
                timeout=timeout + 5,
            )
            return ToolResult(
                stdout    = result.stdout or "",
                stderr    = result.stderr or "",
                exit_code = result.returncode,
                tool_name = self.name,
            )
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


class FileReadTool(BaseTool):
    """
    Lee un archivo del disco (sin usar subprocess).

    Args:
        path     (str): ruta al archivo
        encoding (str): codificación (default: utf-8)
        max_lines(int): máximo de líneas a leer (default: sin límite)

    Siempre mockeable (no usa subprocess).
    """

    name        = "file_read"
    description = "Lee el contenido de un archivo del disco."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        read_fn: Optional[Callable[..., str]] = None,
    ) -> None:
        super().__init__(runner)
        self._read_fn = read_fn

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path      = args.get("path", "")
        encoding  = args.get("encoding", "utf-8")
        max_lines = args.get("max_lines", None)

        if not path:
            return ToolResult("", "No path provided", 1, self.name)

        if self._read_fn is not None:
            try:
                content = self._read_fn(path)
                return ToolResult(content, "", 0, self.name)
            except Exception as e:
                return ToolResult("", str(e), 1, self.name)

        try:
            with open(path, "r", encoding=encoding) as f:
                if max_lines is not None:
                    lines = []
                    for i, line in enumerate(f):
                        if i >= max_lines:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
            return ToolResult(content, "", 0, self.name)
        except FileNotFoundError:
            return ToolResult("", f"File not found: {path}", 1, self.name)
        except Exception as e:
            return ToolResult("", str(e), 1, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# HERRAMIENTAS DEL TOOL SYSTEM (Parte 4 del MEGA-PROMPT)
# ─────────────────────────────────────────────────────────────────────────────
#
# Estas herramientas implementan la spec de tools del modelo:
#   write_file, edit_file, run_code, call_api, search_mem, store_mem
#
# Junto con las ya existentes (web_search → search_web, file_read → read_file)
# cubren los 8 tools listados en 4.1.
#
# Diseño:
#   - Cada herramienta es mockeable (callable inyectable en el constructor)
#     para que los tests no toquen disco, red ni MEM real.
#   - Sandbox enforcement vive en agent/tool_executor.py — las herramientas
#     llaman a las funciones de validación de path/dominio antes de actuar.
# ─────────────────────────────────────────────────────────────────────────────

import os
import json
import urllib.request
import urllib.error
from pathlib import Path


def _default_output_root() -> Path:
    """Raíz de escritura por defecto: <repo>/output/. Creada bajo demanda."""
    here = Path(__file__).resolve().parent.parent
    return (here / "output").resolve()


def _validate_write_path(path: str, output_root: Path) -> Path:
    """
    Verifica que `path` resuelve dentro de `output_root`.
    Permite paths absolutos siempre que estén dentro del root,
    y paths relativos (se anclan al root).

    Raises:
        PermissionError si el path escapa del sandbox.
    """
    if not path:
        raise ValueError("empty path")
    p = Path(path)
    if not p.is_absolute():
        p = output_root / p
    p = p.resolve()
    root = output_root.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise PermissionError(f"path outside sandbox: {p} (root={root})")
    return p


class WriteFileTool(BaseTool):
    """
    Escribe un archivo dentro del sandbox `output_root` (default: <repo>/output/).

    Args:
        path    (str): ruta destino (relativa al root o absoluta dentro del root)
        content (str): contenido a escribir

    Constructor:
        output_root: Path raíz del sandbox (default: <repo>/output/)
        write_fn:    callable(path, content) → None — para mocks
    """

    name        = "write_file"
    description = "Escribe contenido a un archivo dentro de la carpeta /output/."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        output_root: Optional[Path] = None,
        write_fn: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(runner)
        self._output_root = Path(output_root).resolve() if output_root else _default_output_root()
        self._write_fn = write_fn

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path    = args.get("path", "")
        content = args.get("content", "")
        if not path:
            return ToolResult("", "No path provided", 1, self.name)
        try:
            target = _validate_write_path(path, self._output_root)
        except (PermissionError, ValueError) as exc:
            return ToolResult("", str(exc), 1, self.name)

        if self._write_fn is not None:
            try:
                self._write_fn(str(target), content)
                return ToolResult(f"File written: {target}", "", 0, self.name)
            except Exception as exc:
                return ToolResult("", str(exc), 1, self.name)

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content)
            return ToolResult(f"File written: {target}", "", 0, self.name)
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)


class EditFileTool(BaseTool):
    """
    Reemplaza una porción de un archivo existente dentro del sandbox.

    Args:
        path        (str): ruta al archivo dentro del sandbox
        old         (str): texto a reemplazar (debe ser único en el archivo)
        new         (str): texto nuevo
        replace_all (bool): si True, reemplaza todas las ocurrencias

    Constructor:
        output_root: Path sandbox (default: <repo>/output/)
        read_fn:     callable(path) → str — para mocks
        write_fn:    callable(path, content) → None — para mocks
    """

    name        = "edit_file"
    description = "Reemplaza texto en un archivo del sandbox (diff old→new)."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        output_root: Optional[Path] = None,
        read_fn:  Optional[Callable[[str], str]] = None,
        write_fn: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(runner)
        self._output_root = Path(output_root).resolve() if output_root else _default_output_root()
        self._read_fn = read_fn
        self._write_fn = write_fn

    def run(self, args: Dict[str, Any]) -> ToolResult:
        path        = args.get("path", "")
        old         = args.get("old", "")
        new         = args.get("new", "")
        replace_all = bool(args.get("replace_all", False))
        if not path:
            return ToolResult("", "No path provided", 1, self.name)
        if not old:
            return ToolResult("", "Empty 'old' string", 1, self.name)
        try:
            target = _validate_write_path(path, self._output_root)
        except (PermissionError, ValueError) as exc:
            return ToolResult("", str(exc), 1, self.name)

        try:
            if self._read_fn is not None:
                content = self._read_fn(str(target))
            else:
                if not target.exists():
                    return ToolResult("", f"File not found: {target}", 1, self.name)
                with open(target, "r", encoding="utf-8") as f:
                    content = f.read()
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)

        count = content.count(old)
        if count == 0:
            return ToolResult("", f"'old' string not found in {target}", 1, self.name)
        if count > 1 and not replace_all:
            return ToolResult(
                "",
                f"'old' string is not unique in {target} ({count} occurrences); set replace_all=true",
                1,
                self.name,
            )
        new_content = content.replace(old, new) if replace_all else content.replace(old, new, 1)

        try:
            if self._write_fn is not None:
                self._write_fn(str(target), new_content)
            else:
                with open(target, "w", encoding="utf-8") as f:
                    f.write(new_content)
            return ToolResult(f"Edited: {target} ({count} replacement{'s' if count != 1 else ''})", "", 0, self.name)
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)


class RunCodeTool(BaseTool):
    """
    Ejecuta código en un subprocess sandboxeado con timeout.

    Args:
        language (str): "python" | "bash"  (default: "python")
        code     (str): código a ejecutar
        timeout  (int): timeout en segundos (default: 30, hard cap: 60)

    El runner subprocess es mockeable. Por defecto usa subprocess.run con
    capture_output, text=True y timeout. Sin acceso a red garantizado a nivel
    proceso (limitación documentada en 4.3).
    """

    name        = "run_code"
    description = "Ejecuta código (python|bash) en subprocess con timeout."

    HARD_TIMEOUT_CAP = 60
    ALLOWED_LANGUAGES = ("python", "bash")

    def run(self, args: Dict[str, Any]) -> ToolResult:
        language = args.get("language", "python")
        code     = args.get("code", "")
        timeout  = min(int(args.get("timeout", 30)), self.HARD_TIMEOUT_CAP)
        if language not in self.ALLOWED_LANGUAGES:
            return ToolResult("", f"language not allowed: {language}", 1, self.name)
        if not code:
            return ToolResult("", "No code provided", 1, self.name)

        if language == "python":
            import sys as _sys
            cmd = [_sys.executable, "-c", code]
        else:  # bash
            cmd = ["bash", "-c", code]

        try:
            result = self._runner(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            stdout = result.stdout or ""
            stderr = result.stderr or ""
            # cap output to avoid context explosion
            if len(stdout) > 8000: stdout = stdout[:8000] + "\n...[truncated]"
            if len(stderr) > 4000: stderr = stderr[:4000] + "\n...[truncated]"
            return ToolResult(stdout, stderr, result.returncode, self.name)
        except subprocess.TimeoutExpired:
            return ToolResult("", f"Timeout after {timeout}s", 1, self.name)
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)


class CallApiTool(BaseTool):
    """
    Hace una petición HTTP a una URL en la lista blanca de dominios.

    Args:
        url     (str)
        method  (str): GET | POST | PUT | DELETE  (default: GET)
        headers (dict): cabeceras opcionales
        body    (str|dict): cuerpo opcional (dict serializa a JSON)
        timeout (int): segundos (default: 15)

    Constructor:
        allowed_domains (set[str]): dominios permitidos (default: vacío = deny all)
        fetch_fn:        callable(url, method, headers, body, timeout) → (status, text)
                         — para mocks
    """

    name        = "call_api"
    description = "Hace HTTP request a un dominio whitelisted."

    ALLOWED_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH")

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        allowed_domains: Optional[set] = None,
        fetch_fn: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(runner)
        self._allowed_domains = set(allowed_domains) if allowed_domains else set()
        self._fetch_fn = fetch_fn

    def _domain_allowed(self, url: str) -> bool:
        from urllib.parse import urlparse
        try:
            host = urlparse(url).hostname or ""
        except Exception:
            return False
        host = host.lower()
        for d in self._allowed_domains:
            d = d.lower()
            if host == d or host.endswith("." + d):
                return True
        return False

    def run(self, args: Dict[str, Any]) -> ToolResult:
        url     = args.get("url", "")
        method  = args.get("method", "GET").upper()
        headers = args.get("headers", {}) or {}
        body    = args.get("body", None)
        timeout = int(args.get("timeout", 15))
        if not url:
            return ToolResult("", "No URL provided", 1, self.name)
        if method not in self.ALLOWED_METHODS:
            return ToolResult("", f"method not allowed: {method}", 1, self.name)
        if not self._domain_allowed(url):
            return ToolResult("", f"domain not in whitelist: {url}", 1, self.name)

        if self._fetch_fn is not None:
            try:
                status, text = self._fetch_fn(url, method, headers, body, timeout)
                return ToolResult(text, "", 0 if 200 <= int(status) < 400 else 1, self.name)
            except Exception as exc:
                return ToolResult("", str(exc), 1, self.name)

        try:
            data = None
            if body is not None:
                if isinstance(body, (dict, list)):
                    data = json.dumps(body).encode("utf-8")
                    headers.setdefault("Content-Type", "application/json")
                elif isinstance(body, str):
                    data = body.encode("utf-8")
                else:
                    data = str(body).encode("utf-8")
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                return ToolResult(text[:16000], "", 0, self.name)
        except urllib.error.HTTPError as exc:
            return ToolResult("", f"HTTP {exc.code}: {exc.reason}", 1, self.name)
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)


class SearchMemTool(BaseTool):
    """
    Busca entradas relevantes en MEM (SemanticStore).

    Args:
        query  (str)
        top_k  (int): default 5
        domain (str|None): filtra por dominio

    Constructor:
        mem: SemanticStore (o cualquier objeto con .search(query, top_k, domain))
    """

    name        = "search_mem"
    description = "Busca entradas en MEM por similitud semántica."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        mem: Any = None,
    ) -> None:
        super().__init__(runner)
        self._mem = mem

    def run(self, args: Dict[str, Any]) -> ToolResult:
        query  = args.get("query", "")
        top_k  = int(args.get("top_k", 5))
        domain = args.get("domain", None)
        if not query:
            return ToolResult("", "No query provided", 1, self.name)
        if self._mem is None:
            return ToolResult("", "MEM not configured", 1, self.name)
        try:
            results = self._mem.search(query, top_k=top_k, domain=domain)
        except TypeError:
            results = self._mem.search(query, top_k=top_k)
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)

        lines = []
        for item in results or []:
            # SemanticStore returns (key, value, score) tuples
            if isinstance(item, tuple) and len(item) >= 3:
                k, v, s = item[0], item[1], item[2]
                lines.append(f"[{s:.2f}] {k}: {v}")
            else:
                lines.append(str(item))
        return ToolResult("\n".join(lines) if lines else "(no matches)", "", 0, self.name)


class StoreMemTool(BaseTool):
    """
    Guarda una entrada en MEM.

    Args:
        key    (str)
        value  (str)
        domain (str): default "general"

    Constructor:
        mem: SemanticStore
    """

    name        = "store_mem"
    description = "Guarda una entrada en MEM."

    def __init__(
        self,
        runner: Optional[Callable[..., Any]] = None,
        mem: Any = None,
    ) -> None:
        super().__init__(runner)
        self._mem = mem

    def run(self, args: Dict[str, Any]) -> ToolResult:
        key    = args.get("key", "")
        value  = args.get("value", "")
        domain = args.get("domain", "general")
        if not key or not value:
            return ToolResult("", "key and value are required", 1, self.name)
        if self._mem is None:
            return ToolResult("", "MEM not configured", 1, self.name)
        try:
            self._mem.store(key, value, domain=domain)
            return ToolResult(f"Stored: {key} (domain={domain})", "", 0, self.name)
        except Exception as exc:
            return ToolResult("", str(exc), 1, self.name)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

# Herramientas por defecto disponibles en el agente
DEFAULT_TOOLS: Dict[str, BaseTool] = {
    # Existentes (developer-facing)
    "bash":       BashTool(),
    "grep":       GrepTool(),
    "find":       FindTool(),
    "cat":        CatTool(),
    "pytest":     PytestTool(),
    "web_search": WebSearchTool(),
    "web_fetch":  WebFetchTool(),
    "file_read":  FileReadTool(),
    # Tool system del modelo (Parte 4.1) — los 6 nuevos
    "write_file": WriteFileTool(),
    "edit_file":  EditFileTool(),
    "run_code":   RunCodeTool(),
    "call_api":   CallApiTool(),
    # search_mem y store_mem requieren un MEM inyectado, se omiten del default
    # Aliases con los nombres canónicos del MEGA-PROMPT (Parte 4.1)
    "search_web": WebSearchTool(),
    "read_file":  FileReadTool(),
}


def build_tool_registry(
    runner: Optional[Callable[..., Any]] = None,
    search_fn: Optional[Callable[..., str]] = None,
    fetch_fn:  Optional[Callable[..., str]] = None,
    read_fn:   Optional[Callable[..., str]] = None,
    output_root: Optional[Path] = None,
    allowed_domains: Optional[set] = None,
    api_fetch_fn: Optional[Callable[..., Any]] = None,
    mem: Any = None,
) -> Dict[str, BaseTool]:
    """
    Construye el registro completo de herramientas.

    Args:
        runner:          función runner para herramientas subprocess (mocks).
        search_fn:       búsqueda web custom (mocks).
        fetch_fn:        fetch web custom (mocks).
        read_fn:         lectura de archivos custom (mocks).
        output_root:     raíz del sandbox para write_file/edit_file.
        allowed_domains: whitelist de dominios para call_api.
        api_fetch_fn:    fetch HTTP custom para call_api (mocks).
        mem:             SemanticStore para search_mem/store_mem.

    Returns:
        Dict[name → BaseTool] con TODAS las herramientas, incluyendo aliases
        canónicos (search_web, read_file).
    """
    write_tool = WriteFileTool(runner, output_root=output_root)
    edit_tool  = EditFileTool(runner, output_root=output_root)
    run_tool   = RunCodeTool(runner)
    api_tool   = CallApiTool(runner, allowed_domains=allowed_domains, fetch_fn=api_fetch_fn)
    search_mem_tool = SearchMemTool(runner, mem=mem)
    store_mem_tool  = StoreMemTool(runner, mem=mem)
    web_search_tool = WebSearchTool(runner, search_fn)
    file_read_tool  = FileReadTool(runner, read_fn)
    return {
        "bash":       BashTool(runner),
        "grep":       GrepTool(runner),
        "find":       FindTool(runner),
        "cat":        CatTool(runner),
        "pytest":     PytestTool(runner),
        "web_search": web_search_tool,
        "web_fetch":  WebFetchTool(runner, fetch_fn),
        "file_read":  file_read_tool,
        # Tool system del modelo
        "write_file": write_tool,
        "edit_file":  edit_tool,
        "run_code":   run_tool,
        "call_api":   api_tool,
        "search_mem": search_mem_tool,
        "store_mem":  store_mem_tool,
        # Aliases canónicos del MEGA-PROMPT
        "search_web": web_search_tool,
        "read_file":  file_read_tool,
    }
