---
name: sqlite_patterns
domain: forge_c
tags: [sqlite, db, sql]
---
SQLite patterns.

- Always use parameterized queries. Never f-string SQL: vulnerable to injection.
  Bad:  cur.execute(f"SELECT * FROM users WHERE id={uid}")
  Good: cur.execute("SELECT * FROM users WHERE id=?", (uid,))
- Use context managers for connection: with sqlite3.connect(path) as conn.
- Enable foreign keys explicitly: conn.execute("PRAGMA foreign_keys=ON").
- Use WAL mode for better concurrency: PRAGMA journal_mode=WAL.
- Index columns used in WHERE/JOIN, but not too many — writes get slower.
- Migrations: store schema_version in a table. Apply diffs idempotently.
- For bulk inserts, wrap in a transaction (10-100x faster).
- Connection per thread; don't share connection objects.
- Always commit() or use a context manager that does it for you.
