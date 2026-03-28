#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const AGENTS_DIR = path.join(__dirname, 'agents');
const VERSION = '1.0.0';

const AGENTS = fs.readdirSync(AGENTS_DIR)
  .filter(f => f.endsWith('.agent.md'))
  .map(f => f.replace('.agent.md', ''));

function usage() {
  console.log(`ML Pipeline Auto-Auditor v${VERSION}

Usage:
  npx ml-pipeline-auto-auditor install [TARGET]     Install all agents
  npx ml-pipeline-auto-auditor install -s A,B [T]   Install selected agents
  npx ml-pipeline-auto-auditor uninstall [TARGET]    Remove installed agents
  npx ml-pipeline-auto-auditor list                  List available agents

Options:
  -s, --select NAMES   Comma-separated agent names (short names OK)
  -f, --force          Overwrite existing files
  --dry-run            Show what would be done
  -h, --help           Show help

Examples:
  npx ml-pipeline-auto-auditor install .
  npx ml-pipeline-auto-auditor install -s numerical-stability,gradient-flow .
  npx ml-pipeline-auto-auditor uninstall .
`);
}

function listAgents() {
  console.log('Available auditors:\n');
  const meta = {
    'ml-pipeline-audit-orchestrator': 'Orchestrates all auditors',
    'numerical-stability-auditor': 'dtype, bf16, NaN, autocast, precision',
    'gradient-flow-auditor': 'detach, vanishing/exploding gradients',
    'silent-shape-bugs-auditor': 'broadcasting, reshape, einsum',
    'loss-metric-auditor': 'wrong reduction, double softmax',
    'evaluation-bugs-auditor': 'missing eval(), EMA, preprocessing',
    'data-leakage-auditor': 'train-test contamination',
    'data-pipeline-auditor': 'augmentation order, collation',
    'distributed-training-auditor': 'DDP sync, SyncBN, rank bugs',
    'checkpoint-reproducibility-auditor': 'state_dict, resume, dtype',
    'memory-compute-auditor': 'OOM, leaks, compile graph breaks',
    'hyperparameter-config-auditor': 'LR schedule, warmup, config',
    'tokenizer-vocab-auditor': 'vocab size, special tokens',
    'stochastic-nondeterminism-auditor': 'seeds, cuDNN, reproducibility',
    'regularization-conflicts-auditor': 'over-regularization, dropout',
    'dead-code-paths-auditor': 'unused functions, orphan files',
    'geometric-mismatch-auditor': 'manifold/loss, simplex, Riemannian',
  };
  for (const name of AGENTS) {
    const desc = meta[name] || '';
    console.log(`  ${name.padEnd(45)} ${desc}`);
  }
  console.log(`\nTotal: ${AGENTS.length} agents`);
}

function matchAgent(name, selections) {
  for (const sel of selections) {
    if (name === sel || name === `${sel}-auditor` || name === `${sel}-orchestrator`) {
      return true;
    }
  }
  return false;
}

function install(target, opts) {
  const dest = path.join(path.resolve(target), '.github', 'agents');
  console.log(`ML Pipeline Auto-Auditor v${VERSION}`);
  console.log(`Installing to: ${dest}\n`);

  if (!opts.dryRun) {
    fs.mkdirSync(dest, { recursive: true });
  }

  let installed = 0, skipped = 0;
  for (const name of AGENTS) {
    if (opts.select && !matchAgent(name, opts.select)) continue;

    const srcFile = path.join(AGENTS_DIR, `${name}.agent.md`);
    const destFile = path.join(dest, `${name}.agent.md`);

    if (fs.existsSync(destFile) && !opts.force) {
      console.log(`  Skip (exists): ${name}.agent.md — use -f to overwrite`);
      skipped++;
      continue;
    }

    if (opts.dryRun) {
      console.log(`  [dry-run] Would install: ${name}.agent.md`);
    } else {
      fs.copyFileSync(srcFile, destFile);
      console.log(`  Installed: ${name}.agent.md`);
    }
    installed++;
  }

  console.log(`\nDone: ${installed} installed, ${skipped} skipped.\n`);
  console.log('Usage in VS Code Copilot Chat:');
  console.log('  @ml-pipeline-audit-orchestrator run full audit');
}

function uninstall(target, opts) {
  const dest = path.join(path.resolve(target), '.github', 'agents');
  console.log(`Uninstalling from: ${dest}\n`);

  let removed = 0;
  for (const name of AGENTS) {
    const destFile = path.join(dest, `${name}.agent.md`);
    if (fs.existsSync(destFile)) {
      if (opts.dryRun) {
        console.log(`  [dry-run] Would remove: ${name}.agent.md`);
      } else {
        fs.unlinkSync(destFile);
        console.log(`  Removed: ${name}.agent.md`);
      }
      removed++;
    }
  }
  console.log(`\nDone: ${removed} removed.`);
}

// Parse args
const args = process.argv.slice(2);
if (args.length === 0 || args.includes('-h') || args.includes('--help')) {
  usage();
  process.exit(0);
}

const command = args[0];
if (command === 'list') { listAgents(); process.exit(0); }

const opts = { force: false, dryRun: false, select: null };
let target = null;

for (let i = 1; i < args.length; i++) {
  switch (args[i]) {
    case '-f': case '--force': opts.force = true; break;
    case '--dry-run': opts.dryRun = true; break;
    case '-s': case '--select':
      opts.select = args[++i].split(',');
      break;
    default:
      target = args[i];
  }
}

if (!target) target = '.';
if (!fs.existsSync(target)) {
  console.error(`Error: Directory '${target}' does not exist.`);
  process.exit(1);
}

switch (command) {
  case 'install': install(target, opts); break;
  case 'uninstall': uninstall(target, opts); break;
  default: console.error(`Unknown command: ${command}`); usage(); process.exit(1);
}
