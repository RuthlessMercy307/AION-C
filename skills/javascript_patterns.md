---
name: javascript_patterns
domain: forge_c
tags: [javascript, web]
---
JavaScript patterns.

- Prefer const, then let; never var.
- Use async/await over .then() chains.
- Use === and !==, never == or !=.
- Use template literals: `${x}` over string concatenation.
- Destructure objects and arrays at the call site.
- Use modules (import/export), not globals.
- Prefer Array methods (map/filter/reduce) over for loops for transforms.
- Handle Promise rejections explicitly with try/catch around await.
- Validate API responses before using them; never trust shape.
- Use the DOM cautiously: cache queries, batch updates.
