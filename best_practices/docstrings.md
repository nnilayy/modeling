## `docstrings.md`

### Scope

This document defines how docstrings are written in this repository.

Docstrings are used to explain **intent and contracts**, not implementation details or types.

---

### Where docstrings are required

Docstrings are written at **three levels only**:

1. **Module-level**
2. **Class-level**
3. **Function-level**

No other docstrings are required.

---

### Module-level docstring

**Purpose**

* Explain what the file does
* Describe what it provides at a high level

**Rules**

* Placed at the top of the file
* Short paragraph (a few lines)
* No implementation details

---

### Class-level docstring

**Purpose**

* Explain the responsibility of the class

**Rules**

* One-line definition
* Optional 1–2 lines of clarification
* Do not list attributes or methods
* Do not describe internal behavior

---

### Function-level docstring

**Purpose**

* Explain what the function does
* Define the contract

**Structure**

1. One-line summary (required)
2. `Args:` section (only if parameters exist)
3. `Returns:` section (only if a value is returned)

**Rules**

* Do not repeat type information (types come from type hints)
* Keep explanations concise
* Describe behavior, not step-by-step logic

---

### What to avoid

* Restating code or type hints
* Overly long explanations
* Documenting trivial or obvious logic
* Adding docstrings outside the three defined levels

---

### Guiding principle

> **Docstrings explain intent and contracts; type hints explain structure.**
