---
name: code_debugging
domain: forge_c
tags: [debugging, code]
---
Code debugging approach.

- Read the actual error message. Read it again. Don't guess.
- Identify the file and line number. Open it. See the surrounding code.
- Reproduce the bug deterministically before trying to fix it.
- Form a hypothesis about the cause. State it explicitly.
- Test the hypothesis with the smallest possible change.
- One change at a time. Verify after each. Roll back if it doesn't help.
- Check assumptions you've been treating as facts: variable types, return shapes, env vars.
- When stuck, simplify: remove code until the bug disappears, then add back.
- After fixing, write a test that would have caught it. So it can't return.
- Record the root cause in MEM, not just the fix.
