# ML Pipeline Auditors

**17 specialized AI agents for comprehensive ML pipeline bug detection.**

Works with **GitHub Copilot** (VS Code) and **Claude Code** — any environment that supports `.agent.md` files.

## What This Is

A collection of 16 focused auditor agents + 1 orchestrator that systematically find silent bugs in ML training pipelines. Each agent specializes in one class of errors and encodes expert knowledge about detection patterns, false positive avoidance, and severity classification.

These are not generic linters — they understand ML-specific semantics like autocast promotion rules, manifold geometry, gradient flow through custom autograd functions, and distributed training synchronization patterns.

## Auditors

| Agent | Finds | Categories |
|-------|-------|------------|
| **Orchestrator** | Coordinates all auditors | — |
| Numerical Stability | bf16 overflow, NaN, autocast boundaries, precision loss | 30+ |
| Gradient Flow | detach bugs, dead neurons, vanishing/exploding gradients | 22 |
| Silent Shape Bugs | broadcasting errors, reshape bugs, einsum mismatches | 18 |
| Loss / Metric Mismatch | wrong reduction, double softmax, loss/metric misalignment | 17 |
| Evaluation Bugs | missing eval(), EMA not swapped, train augmentation in val | 19 |
| Data Leakage | train-test contamination, normalization before split | 15 |
| Data Pipeline | augmentation order, preprocessing mismatch, collation bugs | 20 |
| Distributed Training | DDP sync, SyncBN, DistributedSampler, rank-dependent bugs | 19 |
| Checkpoint / Reproducibility | incomplete state_dict, resume bugs, dtype mismatch | 18 |
| Memory / Compute Waste | memory leaks, OOM, torch.compile graph breaks | 20 |
| Hyperparameter / Config | LR schedule mismatch, warmup bugs, config inconsistency | 19 |
| Tokenizer / Vocab | vocab size mismatch, special tokens, ignore_index | 14 |
| Stochastic Nondeterminism | seed management, cuDNN benchmark, worker seeding | 19 |
| Regularization Conflicts | over-regularization, dropout stacking, WD on bias | 15 |
| Dead Code / Unreachable Paths | unused functions, config-gated dead code, orphan files | 20 |
| Geometric Mismatch | manifold/loss incompatibility, simplex, Riemannian geometry | 32 |

**Total: ~300 bug categories across 17 agents.**

## Installation

### Quick Install (all auditors)

```bash
git clone https://github.com/aogavrilov/ml-pipeline-auto-auditor.git
cd ml-pipeline-auditors
./install.sh /path/to/your-ml-project
```

### Selective Install

```bash
# Only the top 5 most impactful auditors
./install.sh -s numerical-stability,gradient-flow,silent-shape-bugs,loss-metric,evaluation-bugs .

# Specific auditors for your use case
./install.sh -s numerical-stability,geometric-mismatch,loss-metric /path/to/project
```

### Manual Install

Copy the files you want from `agents/` into your project's `.github/agents/`:

```bash
mkdir -p /path/to/project/.github/agents
cp agents/*.agent.md /path/to/project/.github/agents/
```

### Uninstall

```bash
./install.sh --uninstall /path/to/your-ml-project
```

## Usage

After installation, agents appear in VS Code Copilot Chat:

### Full Audit (recommended)

```
@ml-pipeline-audit-orchestrator run full audit of this codebase
```

The orchestrator will:
1. **Pre-flight** — identify your framework, data domain, and scope
2. **Triage** — skip irrelevant auditors (e.g., skip Distributed if single GPU)
3. **Run auditors** in dependency-aware order across 5 phases
4. **Cross-reference** findings between auditors (dead code downgrades, config↔loss checks)
5. **Produce unified report** with CRITICAL/WARNING/INFO severity

### Quick Audit

```
@ml-pipeline-audit-orchestrator quick audit
```

Runs only the top 5 auditors for a fast check.

### Individual Auditors

```
@numerical-stability-auditor audit src/ for dtype safety issues
@gradient-flow-auditor check for detached tensors in the training loop
@geometric-mismatch-auditor is my loss function compatible with simplex data?
@silent-shape-bugs-auditor check attention mask shapes
```

## How It Works

Each agent follows the same pattern:

1. **Principles** — Core rules that prevent false positives (e.g., "trace full dtype chains before classifying severity")
2. **Tiered Categories** — Bug types organized by severity/likelihood
3. **grep-based Methodology** — Systematic search patterns for each category
4. **Severity Classification** — CRITICAL / WARNING / INFO with clear criteria
5. **Constraints** — Explicit rules about what NOT to flag

### Example: Numerical Stability Auditor

The agent knows that `F.cross_entropy` is in PyTorch's autocast fp32 promotion list, so it won't flag `model(x)` → `F.cross_entropy(logits, target)` as CRITICAL even if logits are bf16 — it traces the full dtype chain first. But it WILL flag manual loss implementations that bypass autocast.

### Example: Geometric Mismatch Auditor

The agent knows that data on a simplex (probability distributions) requires different loss functions (KL, not MSE), different noise processes (Dirichlet, not Gaussian), and different interpolation (geodesic, not linear). It checks whether your code matches the geometry of your data.

## Supported Frameworks

- PyTorch (raw)
- PyTorch Lightning
- HuggingFace Transformers/Trainer
- Hydra/OmegaConf configs
- Any Python ML codebase

## Requirements

- VS Code with GitHub Copilot (Chat) — or any tool that reads `.agent.md` files
- No dependencies, no runtime, no API keys — agents are plain Markdown

## Customization

Each `.agent.md` file is self-contained Markdown. You can:

- **Edit categories** to match your domain (add vision-specific checks, remove NLP checks)
- **Adjust severity** criteria for your team's standards
- **Add grep patterns** specific to your codebase (custom loss functions, etc.)
- **Remove agents** you don't need (uninstall selectively with `-s`)

## Contributing

PRs welcome. To add a new auditor:

1. Create `agents/your-auditor-name.agent.md` following the existing pattern
2. Add it to the orchestrator's agent table and execution phases
3. Update this README

Each auditor should have: YAML frontmatter (description, name, tools), principles, tiered categories, methodology with grep commands, severity classification, and constraints.

## License

MIT
