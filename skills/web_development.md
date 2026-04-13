---
name: web_development
domain: forge_c
tags: [web, html, css, http, rest]
---
Web development patterns.

- HTML is structure, CSS is presentation, JS is behavior. Don't mix.
- Semantic tags first: header/main/article/nav/footer over div soup.
- Mobile-first CSS: design for narrow, then add breakpoints up.
- REST APIs: GET reads, POST creates, PUT replaces, PATCH updates, DELETE removes.
- HTTP status codes matter: 200 OK, 201 Created, 400 bad input, 401 no auth,
  403 forbidden, 404 missing, 500 server error.
- Auth: never roll your own. Use JWT or session cookies with HttpOnly+Secure.
- Always sanitize user input on the server, not just the client.
- CORS: only allow the origins you actually need. Wildcard is for public APIs.
- Cache static assets aggressively, API responses cautiously.
- Use HTTPS everywhere. Plain HTTP is for local dev only.
