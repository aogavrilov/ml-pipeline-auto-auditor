# Promotion Guide for ml-pipeline-auto-auditor

## Where to Post

### Tier 1 — Highest Impact

#### Hacker News — Show HN
URL: https://news.ycombinator.com/submit
```
Title: Show HN: 17 AI agents that audit your ML training pipeline for silent bugs
URL: https://github.com/aogavrilov/ml-pipeline-auto-auditor
```
Comment to post immediately after:
```
I built a set of 17 specialized Copilot agents that systematically find
silent bugs in ML training pipelines — the kind that don't crash but
silently hurt model quality.

Each agent specializes in one class of errors:
- Numerical stability (bf16 overflow, autocast boundaries)
- Gradient flow (detach bugs, dead neurons)
- Silent shape bugs (broadcasting errors that don't throw)
- Loss/metric mismatch (wrong reduction, double softmax)
- Data leakage (normalization before train/test split)
- Geometric mismatch (Euclidean loss on simplex data)
- ... and 11 more categories

~300 bug categories total. Works with GitHub Copilot Chat in VS Code.

Install:
  npx ml-pipeline-auto-auditor install .

Then in Copilot Chat:
  @ml-pipeline-audit-orchestrator run full audit

They're not generic linters — they understand ML semantics like autocast
promotion rules, manifold geometry, and distributed training sync patterns.

Built these while debugging my own diffusion model training pipeline and
realized the methodology could be encoded as reusable agents.
```

#### Reddit — r/MachineLearning (2.9M members)
URL: https://www.reddit.com/r/MachineLearning/submit
```
Title: [P] ML Pipeline Auto-Auditor: 17 AI agents that find silent bugs in your training pipeline
```
Body:
```
I open-sourced a collection of 17 specialized AI agents for GitHub Copilot
that systematically audit ML training pipelines for silent bugs — the kind
that don't crash but quietly hurt your model.

**What it finds (~300 categories):**
- ⚡ Numerical stability: bf16 overflow, NaN, autocast boundary violations
- 🔀 Gradient flow: detached tensors, dead neurons, vanishing gradients
- 📐 Silent shape bugs: broadcasting that doesn't throw but gives wrong results
- 📉 Loss/metric mismatch: wrong reduction, double softmax, label smoothing bugs
- 🔒 Data leakage: normalization before split, duplicate samples across sets
- 🌀 Geometric mismatch: Euclidean loss on simplex data, wrong noise process
- ... and 11 more specialized auditors

**How it works:**
Each agent encodes expert knowledge about detection patterns, false positive
avoidance, and severity classification. They use grep-based methodology
so they actually trace code paths rather than guessing.

The orchestrator coordinates all 16 auditors in dependency-aware order:
pre-flight triage → phased execution → cross-referencing → unified report.

**Install:**
```
npx ml-pipeline-auto-auditor install .
```

**Use in VS Code Copilot Chat:**
```
@ml-pipeline-audit-orchestrator run full audit
```

GitHub: https://github.com/aogavrilov/ml-pipeline-auto-auditor

Built this while debugging a simplex diffusion model and realized the
systematic methodology could help others. MIT licensed, PRs welcome.
```

#### Reddit — r/LocalLLaMA (800K+ members)
URL: https://www.reddit.com/r/LocalLLaMA/submit
Same body as above, but add a note about Claude Code compatibility.


### Tier 2 — Good Reach

#### Twitter/X
```
I open-sourced 17 AI agents that audit ML training pipelines for silent bugs.

~300 bug categories:
• Numerical stability (bf16/NaN)
• Gradient flow (detach bugs)
• Silent shape broadcasting
• Loss geometry mismatch
• Data leakage
• 12 more categories

Works with GitHub Copilot in VS Code:
npx ml-pipeline-auto-auditor install .
Then: @ml-pipeline-audit-orchestrator run full audit

They encode expert debugging knowledge — not generic linting.

https://github.com/aogavrilov/ml-pipeline-auto-auditor
```

#### Reddit — r/github, r/vscode, r/learnmachinelearning
Same post adapted for each audience.

#### Dev.to Article
URL: https://dev.to/new
Title: "I Built 17 AI Agents That Find Silent Bugs in ML Training Pipelines"
(Use README content as the article body, add personal story about building it)


### Tier 3 — Niche but Targeted

#### GitHub Discussions — github/copilot-docs or similar
Post in community discussions about useful Copilot agents.

#### awesome-copilot list
URL: https://github.com/component-driven/awesome-copilot
Submit a PR adding the repo to the "Agents" section.

#### awesome-mlops
URL: https://github.com/visenger/awesome-mlops
Submit a PR to the "Testing" or "Quality" section.

#### PyTorch Forums
URL: https://discuss.pytorch.org
"Tools" or "Show & Tell" category.


## npm Publish (pending auth)

To make `npx` work publicly:
```bash
npm login          # enter username, password, email, OTP
npm publish        # publishes to npmjs.com
```

After publish, anyone can run:
```bash
npx ml-pipeline-auto-auditor install .
```


## Timeline Recommendation

Day 1: Reddit r/MachineLearning + Twitter/X
Day 2: Hacker News (Show HN) — post around 8-10am US Eastern
Day 3: Dev.to article
Day 4: PRs to awesome-lists
Ongoing: Respond to comments, fix issues, iterate
