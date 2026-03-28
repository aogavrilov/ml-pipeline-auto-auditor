#!/usr/bin/env bash
set -euo pipefail

# ML Pipeline Auditors — Installer
# Copies agent files into your project's .github/agents/ directory.

VERSION="1.0.0"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENTS_SRC="$SCRIPT_DIR/agents"

usage() {
    cat <<EOF
ML Pipeline Auditors v$VERSION

Usage:
    $(basename "$0") [OPTIONS] [TARGET_PROJECT_DIR]

Options:
    -h, --help          Show this help
    -l, --list          List available auditors
    -s, --select NAMES  Install only selected auditors (comma-separated)
    -f, --force         Overwrite existing agent files
    -u, --uninstall     Remove installed auditors from target
    --dry-run           Show what would be done without doing it

Examples:
    # Install all auditors into current project
    $(basename "$0") .

    # Install into specific project
    $(basename "$0") /path/to/my-ml-project

    # Install only specific auditors
    $(basename "$0") -s numerical-stability,gradient-flow,loss-metric .

    # Quick audit set (top 5 most impactful)
    $(basename "$0") -s numerical-stability,gradient-flow,silent-shape-bugs,loss-metric,evaluation-bugs .

    # Uninstall all auditors
    $(basename "$0") --uninstall /path/to/my-ml-project
EOF
}

list_auditors() {
    echo "Available auditors:"
    echo ""
    printf "  %-45s %s\n" "AGENT" "FOCUS"
    printf "  %-45s %s\n" "-----" "-----"
    printf "  %-45s %s\n" "ml-pipeline-audit-orchestrator" "Orchestrates all auditors (install this!)"
    printf "  %-45s %s\n" "numerical-stability-auditor" "dtype, bf16, NaN, autocast, precision"
    printf "  %-45s %s\n" "gradient-flow-auditor" "detach, vanishing/exploding gradients"
    printf "  %-45s %s\n" "silent-shape-bugs-auditor" "broadcasting, reshape, einsum"
    printf "  %-45s %s\n" "loss-metric-auditor" "wrong reduction, double softmax"
    printf "  %-45s %s\n" "evaluation-bugs-auditor" "missing eval(), EMA, preprocessing"
    printf "  %-45s %s\n" "data-leakage-auditor" "train-test contamination"
    printf "  %-45s %s\n" "data-pipeline-auditor" "augmentation order, collation"
    printf "  %-45s %s\n" "distributed-training-auditor" "DDP sync, SyncBN, rank bugs"
    printf "  %-45s %s\n" "checkpoint-reproducibility-auditor" "state_dict, resume, dtype"
    printf "  %-45s %s\n" "memory-compute-auditor" "OOM, leaks, compile graph breaks"
    printf "  %-45s %s\n" "hyperparameter-config-auditor" "LR schedule, warmup, config"
    printf "  %-45s %s\n" "tokenizer-vocab-auditor" "vocab size, special tokens"
    printf "  %-45s %s\n" "stochastic-nondeterminism-auditor" "seeds, cuDNN, reproducibility"
    printf "  %-45s %s\n" "regularization-conflicts-auditor" "over-regularization, dropout"
    printf "  %-45s %s\n" "dead-code-paths-auditor" "unused functions, orphan files"
    printf "  %-45s %s\n" "geometric-mismatch-auditor" "manifold/loss, simplex, Riemannian"
    echo ""
    echo "Total: 17 agents (16 auditors + 1 orchestrator)"
}

# Parse arguments
FORCE=false
DRY_RUN=false
UNINSTALL=false
SELECT=""
TARGET=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -l|--list) list_auditors; exit 0 ;;
        -s|--select) SELECT="$2"; shift 2 ;;
        -f|--force) FORCE=true; shift ;;
        -u|--uninstall) UNINSTALL=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -*) echo "Unknown option: $1"; usage; exit 1 ;;
        *) TARGET="$1"; shift ;;
    esac
done

# Validate target
if [[ -z "$TARGET" ]]; then
    echo "Error: No target directory specified."
    echo "Usage: $(basename "$0") [OPTIONS] TARGET_DIR"
    exit 1
fi

if [[ ! -d "$TARGET" ]]; then
    echo "Error: Directory '$TARGET' does not exist."
    exit 1
fi
TARGET="$(cd "$TARGET" && pwd)"
DEST="$TARGET/.github/agents"

# Check source exists
if [[ ! -d "$AGENTS_SRC" ]]; then
    echo "Error: Agents source directory not found at $AGENTS_SRC"
    exit 1
fi

# Uninstall mode
if [[ "$UNINSTALL" == true ]]; then
    echo "Uninstalling ML Pipeline Auditors from $TARGET..."
    count=0
    for src_file in "$AGENTS_SRC"/*.agent.md; do
        name="$(basename "$src_file")"
        dest_file="$DEST/$name"
        if [[ -f "$dest_file" ]]; then
            if [[ "$DRY_RUN" == true ]]; then
                echo "  [dry-run] Would remove: $dest_file"
            else
                rm "$dest_file"
                echo "  Removed: $name"
            fi
            count=$((count + 1))
        fi
    done
    echo "Done. Removed $count agent(s)."
    exit 0
fi

# Install mode
echo "ML Pipeline Auditors v$VERSION"
echo "Installing to: $DEST"
echo ""

# Create destination
if [[ "$DRY_RUN" != true ]]; then
    mkdir -p "$DEST"
fi

# Determine which agents to install
installed=0
skipped=0

for src_file in "$AGENTS_SRC"/*.agent.md; do
    name="$(basename "$src_file" .agent.md)"
    
    # Filter by selection if specified
    if [[ -n "$SELECT" ]]; then
        match=false
        IFS=',' read -ra SELECTIONS <<< "$SELECT"
        for sel in "${SELECTIONS[@]}"; do
            if [[ "$name" == "$sel" ]] || [[ "$name" == "${sel}-auditor" ]] || [[ "$name" == "${sel}-orchestrator" ]]; then
                match=true
                break
            fi
        done
        if [[ "$match" != true ]]; then
            continue
        fi
    fi
    
    dest_file="$DEST/$(basename "$src_file")"
    
    # Check if exists
    if [[ -f "$dest_file" ]] && [[ "$FORCE" != true ]]; then
        echo "  Skip (exists): $(basename "$src_file")  — use -f to overwrite"
        skipped=$((skipped + 1))
        continue
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "  [dry-run] Would install: $(basename "$src_file")"
    else
        cp "$src_file" "$dest_file"
        echo "  Installed: $(basename "$src_file")"
    fi
    installed=$((installed + 1))
done

echo ""
echo "Done: $installed installed, $skipped skipped."
echo ""
echo "Usage in VS Code Copilot Chat:"
echo '  @ml-pipeline-audit-orchestrator run full audit'
echo '  @numerical-stability-auditor check this codebase'
echo '  @gradient-flow-auditor audit src/'
