# Antigravity Workspace Plugins

This directory contains 4 specialized workspace plugins configured for Antigravity:

| Plugin | Type | Primary Use Case | Trigger Prompt Examples |
| :--- | :--- | :--- | :--- |
| **Get Shit Done (GSD)** | Execution | High-velocity, rapid-prototyping, zero-fluff delivery with smoke tests | `"GSD: build this feature"`, `"Ship it fast"`, `"Get shit done"` |
| **Roo Code** | Modes | Multi-mode development personas (Architect, Code, Ask, Debug, Test) | `"Switch to Roo Architect mode"`, `"Debug mode for this issue"` |
| **Ralph Loop** | Self-Healing | Autonomous test-driven feedback and error repair loop | `"Run Ralph Loop on pytest"`, `"Self-heal until tests pass"` |
| **CodeRabbit** | Code Review | Multi-dimensional static analysis, PR review, security/perf audits | `"Run CodeRabbit review"`, `"Audit changes for security"` |

---

## Directory Structure

```text
.agents/
├── plugins.json
├── plugins/
│   ├── get-shit-done/
│   │   ├── plugin.json
│   │   ├── rules/
│   │   │   └── gsd-rules.md
│   │   └── skills/
│   │       └── get-shit-done/
│   │           └── SKILL.md
│   ├── roo-code/
│   │   ├── plugin.json
│   │   ├── rules/
│   │   │   └── roo-modes.md
│   │   └── skills/
│   │       └── roo-code/
│   │           └── SKILL.md
│   ├── ralph-loop/
│   │   ├── plugin.json
│   │   ├── rules/
│   │   │   └── ralph-loop-rules.md
│   │   └── skills/
│   │       └── ralph-loop/
│   │           └── SKILL.md
│   └── coderabbit/
│       ├── plugin.json
│       ├── rules/
│       │   └── coderabbit-rules.md
│       └── skills/
│           └── coderabbit-review/
│               └── SKILL.md
```
