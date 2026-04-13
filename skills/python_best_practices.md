---
name: python_best_practices
domain: forge_c
tags: [python, code]
---
Python best practices.

- Use type hints on function signatures: def f(x: int) -> str.
- Prefer f-strings: f"value={x}" over "value=" + str(x).
- Use context managers: with open(p) as f.
- Catch specific exceptions, never bare except.
- Use dataclasses for plain data, not dicts.
- Validate inputs at function boundaries, not internally.
- Prefer pure functions; mutate explicitly.
- Use pathlib.Path over os.path string concatenation.
- Use list/dict comprehensions for simple transforms; loops for complex logic.
- Run mypy and ruff before committing.
