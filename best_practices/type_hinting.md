## `type_hinting.md`

### Scope

This document defines how type hinting is applied in this repository.

Type hinting is limited to **function signatures only**:

* Standalone functions
* Class methods

Only **function parameters and return types** are type hinted.

---

### When type hinting is done

Type hinting is applied **after the function logic is implemented**.

Before adding or fixing type hints:

* Read the **entire file from top to bottom**
* Understand how the file works as a whole
* Identify what each function returns across all code paths

---

### How type hinting is applied

* Work **one function at a time**
* Accurately type hint:

  * All parameters
  * The return value
* Type hints must reflect **actual behavior**, not assumptions

---

### What is intentionally not type hinted

* Local variables
* Class attributes
* Module-level variables
* Internal implementation details

---

### Guiding principle

> **Type hint function boundaries accurately, based on full file context.**
