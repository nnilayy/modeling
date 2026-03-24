## `README.md`

### Purpose

This folder documents the **engineering standards and workflow** followed in this repository.

The goal is to:

* Keep code consistent
* Reduce rework and drift
* Make files easy to read, review, and maintain

These practices apply **file by file** and should be followed for all new code and major refactors.

---

### Development Workflow (File-by-File)

When working on any file, follow this order:

1. **Raw implementation**

   * Write the core logic first
   * Define classes and functions
   * Focus on correctness and structure

2. **Type hinting**

   * Add full type hints to public functions and methods
   * Fix class attribute types and return types
   * Ensure async functions have correct return annotations

3. **Error handling**

   * Add appropriate try/except blocks
   * Decide what errors are handled locally vs propagated
   * Avoid swallowing exceptions

4. **Logging**

   * Add logging at meaningful boundaries
   * Use appropriate log levels
   * Do not log sensitive data
   * Avoid excessive or noisy logs

5. **Docstrings**

   * Add module-level docstrings
   * Add concise class-level docstrings (1–3 lines)
   * Add function-level docstrings (one-line summary with Args and Returns where applicable)
   * Ensure docstrings reflect final behavior and types

6. **Linting and formatting**

   * Run linters and formatters
   * Fix warnings and errors
   * Ensure the file meets repository linting standards

This order is intentional and helps prevent duplicated effort and inconsistent documentation.

---

### What lives in this folder

Each file in this directory focuses on a single concern:

* `docstrings.md` – How and where to write docstrings
* `type_hinting.md` – Type hinting rules and conventions
* `logging.md` – Logging standards and guidelines
* `error_handling.md` – Exception handling best practices
* `linting.md` – Linting and formatting rules

Refer to the relevant document when working on that aspect of the code.

---

### Guiding principle

> **Correctness first, clarity second, polish last.**
