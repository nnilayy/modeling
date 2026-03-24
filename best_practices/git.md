---

## Cleaned-up, correct checklist

### One-time

```bash
git clone <repo-url>
cd <repo-name>
git branch
```

Identify the default branch (`dev` or `main`).

```bash
git checkout -b feat/my-feature
```

---

### Repeat (daily loop)

#### 0. Check for updates (non-invasive)

```bash
git fetch origin
git status
```

This lets you see if the default branch is behind its remote without changing anything.

---

#### 1. Sync default branch (only if updates exist)

```bash
git checkout dev
git pull --ff-only origin dev
```

---

#### 2. Rebase feature branch on latest default

```bash
git checkout feat/my-feature
git rebase dev
```

---

#### 3. Continue development

```bash
# edit files
git add .
git commit -m "Implement X"
```

Repeat commits as needed.

---

#### 4. Push feature branch (when needed)

```bash
git push --force-with-lease
```

---

This version now explicitly includes **fetch + status** for safely checking updates before syncing.
