## `linting.md`

### Scope

This document defines how linting and formatting are applied in this repository.

Linting is used to ensure **consistency, readability, and correctness**, not to rewrite logic.

---

### When linting is done

Linting is the **final step** in the file-level workflow.

It is applied **after**:

1. Core logic is implemented
2. Type hints are finalized
3. Error handling and logging are added
4. Docstrings are complete

Linting should never drive design decisions.

---

### What linting is responsible for

Linting is used to enforce:

* Code style consistency
* Formatting standards
* Obvious correctness issues
* Unused imports and variables
* Simple structural issues

---

### What linting is not responsible for

Linting is **not** used to:

* Change control flow
* Restructure logic
* Decide error-handling strategy
* Replace code review or reasoning

---

### How linting issues should be handled

* Fix warnings and errors where reasonable
* Prefer clarity over strictness when conflicts arise
* Suppress rules only when justified and documented

---

### Guiding principle

> **Linting enforces consistency after correctness is achieved.**
