"""
run_pipeline.py -- End-to-end pipeline orchestrator for Cardiotox-Fusion.

This script runs the full pipeline in the correct order, with proper
dependency checking at each step. Steps that require internet access
or large downloads are clearly flagged.

Pipeline steps:
  Step 1: Fetch labels         (local file read -- instant)
  Step 2: Resolve SMILES       (ChEMBL API -- ~10-15 min, internet required)
  Step 3: Match LINCS          (local metadata -- fast; GCTX download is optional/slow)
  Step 4: Build graphs         (local RDKit -- fast)
  Step 5: [Already done]       05_trace_ten_compounds.py (trace-through)
  Step 6: Train GNN            (GPU recommended, ~30-60 min)
  Step 7: Train Transformer    (GPU recommended, ~30-60 min)
  Step 8: Train Fusion         (GPU recommended, ~45-90 min)
  Step 9: Evaluate all models  (fast)
  Step 10: Interpretability    (moderate)

Usage:
  # Full pipeline (all steps):
  python run_pipeline.py

  # Data prep only (steps 1-4):
  python run_pipeline.py --data-only

  # Training only (steps 6-8, requires data prep done):
  python run_pipeline.py --train-only

  # Evaluation only (step 9-10, requires training done):
  python run_pipeline.py --eval-only

  # Specific steps only:
  python run_pipeline.py --steps 1 2 3

  # Skip steps that need internet (offline mode):
  python run_pipeline.py --offline
"""

import os
import sys
import argparse
import subprocess
import time
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import CFG
from scripts.utils import get_logger, ensure_dirs, file_exists_nonempty

LOG_FILE = os.path.join(CFG.RESULTS_DIR, "logs", "run_pipeline.log")
logger = get_logger("run_pipeline", LOG_FILE)


# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS = {
    1: {
        "name": "Fetch Labels (DICTrank)",
        "script": "scripts/01_fetch_labels.py",
        "output": CFG.LABELS_CSV,
        "requires_internet": False,
        "requires_gpu": False,
        "notes": "Reads local dictrank_dataset_508.xlsx -- instant",
    },
    2: {
        "name": "Resolve SMILES (ChEMBL API)",
        "script": "scripts/02_resolve_smiles.py",
        "output": CFG.SMILES_CSV,
        "requires_internet": True,
        "requires_gpu": False,
        "notes": "~10-15 min. Use --resume to restart from checkpoint.",
        "extra_args": [],
    },
    3: {
        "name": "Match LINCS L1000",
        "script": "scripts/03_match_lincs.py",
        "output": CFG.LINCS_MATCHED_CSV,
        "requires_internet": False,   # metadata local; GCTX is optional download
        "requires_gpu": False,
        "notes": "Name + InChIKey matching against local metadata. "
                 "Use --download to fetch 5 GB GCTX. Use --extract to get expression matrix.",
        "extra_args": [],
    },
    4: {
        "name": "Build Molecular Graphs",
        "script": "scripts/04_build_graphs.py",
        "output": os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv"),
        "requires_internet": False,
        "requires_gpu": False,
        "notes": "SMILES -> PyTorch Geometric graph objects",
    },
    5: {
        "name": "Trace Ten Compounds (pre-pipeline validation)",
        "script": "scripts/05_trace_ten_compounds.py",
        "output": CFG.TRACE_LOG,
        "requires_internet": True,
        "requires_gpu": False,
        "notes": "MANDATORY trace-through -- already completed in initial notebook session",
    },
    6: {
        "name": "Train GNN (structure-only baseline)",
        "script": "scripts/06_train_gnn.py",
        "output": CFG.GNN_CHECKPOINT,
        "requires_internet": False,
        "requires_gpu": True,
        "notes": "~30-60 min with GPU. Trains on full 1,211 compounds.",
    },
    7: {
        "name": "Train Transformer (biology-only baseline)",
        "script": "scripts/07_train_transformer.py",
        "output": CFG.TRANSFORMER_CKPT,
        "requires_internet": False,
        "requires_gpu": True,
        "notes": "~30-60 min with GPU. Trains on ~423 LINCS-covered compounds.",
    },
    8: {
        "name": "Train Fusion Model (GNN + Transformer + CrossAttention)",
        "script": "scripts/08_train_fusion.py",
        "output": CFG.FUSION_CHECKPOINT,
        "requires_internet": False,
        "requires_gpu": True,
        "notes": "~45-90 min with GPU. Main model of the project.",
    },
    9: {
        "name": "Evaluate All Models (fair comparison)",
        "script": "scripts/09_evaluate.py",
        "output": CFG.BASELINE_REPORT,
        "requires_internet": False,
        "requires_gpu": False,
        "notes": "All models on same LINCS-covered test set. Generates baseline_comparison.md",
    },
    10: {
        "name": "Interpretability Analysis",
        "script": "scripts/10_interpretability.py",
        "output": CFG.INTERP_REPORT,
        "requires_internet": False,
        "requires_gpu": False,
        "notes": "Attention -> toxicophore validation. Generates interpretability_validation.md",
    },
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def check_step_dependencies(step_num: int) -> list:
    """Check if all required outputs from prior steps exist."""
    missing = []
    dependencies = {
        2: [CFG.LABELS_CSV],
        3: [CFG.SMILES_CSV],
        4: [CFG.LINCS_MATCHED_CSV],
        6: [os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")],
        7: [CFG.EXPRESSION_CSV, os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")],
        8: [CFG.EXPRESSION_CSV, os.path.join(CFG.PROCESSED_DIR, "graph_manifest.csv")],
        9: [CFG.GNN_CHECKPOINT, CFG.TRANSFORMER_CKPT, CFG.FUSION_CHECKPOINT],
        10: [CFG.FUSION_CHECKPOINT, CFG.EXPRESSION_CSV],
    }
    for required_file in dependencies.get(step_num, []):
        if not file_exists_nonempty(required_file):
            missing.append(required_file)
    return missing


def run_step(
    step_num: int,
    extra_args: list = None,
    skip_if_done: bool = True,
) -> bool:
    """
    Run a single pipeline step.

    Args:
        step_num:      Step number (1-10)
        extra_args:    Additional CLI arguments to pass to the script
        skip_if_done:  If True, skip step if output already exists

    Returns:
        True if step succeeded (or was skipped), False if it failed.
    """
    step = STEPS.get(step_num)
    if step is None:
        logger.error(f"Unknown step: {step_num}")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP {step_num}: {step['name']}")
    logger.info(f"Script : {step['script']}")
    logger.info(f"Notes  : {step['notes']}")
    if step["requires_internet"]:
        logger.info("[!]  REQUIRES INTERNET ACCESS")
    if step["requires_gpu"]:
        logger.info("⚡ BENEFITS FROM GPU (CPU will work but slower)")

    # Check if already done
    if skip_if_done and file_exists_nonempty(step["output"]):
        logger.info(f"OK Output already exists: {step['output']}")
        logger.info(f"  Skipping step {step_num}. Use --force to re-run.")
        return True

    # Check dependencies
    missing = check_step_dependencies(step_num)
    if missing:
        logger.error(f"Missing required inputs for step {step_num}:")
        for f in missing:
            logger.error(f"  - {f}")
        logger.error("Run the preceding steps first.")
        return False

    # Run the script
    script_path = os.path.join(PROJECT_ROOT, step["script"])
    cmd = [sys.executable, script_path] + (extra_args or step.get("extra_args", []))

    logger.info(f"Running: {' '.join(cmd)}")
    start = time.time()

    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, check=True,
            capture_output=False,  # Let stdout/stderr flow through
        )
        elapsed = time.time() - start
        logger.info(f"OK Step {step_num} completed in {elapsed:.0f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        logger.error(f"X Step {step_num} FAILED after {elapsed:.0f}s (exit code: {e.returncode})")
        return False

    except Exception as e:
        logger.error(f"X Step {step_num} FAILED with unexpected error: {e}")
        logger.error(traceback.format_exc())
        return False


def print_pipeline_status():
    """Print the status of all pipeline outputs."""
    logger.info("\n" + "="*60)
    logger.info("PIPELINE STATUS OVERVIEW")
    logger.info("="*60)
    for step_num, step in STEPS.items():
        output_exists = file_exists_nonempty(step["output"])
        status = "OK DONE" if output_exists else "PENDING"
        internet = " [internet]" if step["requires_internet"] else ""
        gpu = " [GPU]" if step["requires_gpu"] else ""
        logger.info(f"  Step {step_num}: {status}{internet}{gpu} -- {step['name']}")
    logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Cardiotox-Fusion end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data-only", action="store_true",
                        help="Run only data prep steps (1-4)")
    parser.add_argument("--train-only", action="store_true",
                        help="Run only training steps (6-8)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Run only evaluation steps (9-10)")
    parser.add_argument("--steps", nargs="+", type=int,
                        help="Run specific step numbers only, e.g. --steps 1 2 3")
    parser.add_argument("--offline", action="store_true",
                        help="Skip steps that require internet access")
    parser.add_argument("--force", action="store_true",
                        help="Re-run steps even if output already exists")
    parser.add_argument("--status", action="store_true",
                        help="Show pipeline status and exit")
    parser.add_argument("--download-gctx", action="store_true",
                        help="During step 3, also download the 5 GB GCTX expression matrix")
    parser.add_argument("--extract-expression", action="store_true",
                        help="During step 3, also extract the expression matrix from GCTX")
    args = parser.parse_args()

    ensure_dirs(CFG.PROCESSED_DIR, CFG.GRAPHS_DIR, CFG.CHECKPOINTS_DIR,
                CFG.RESULTS_DIR, os.path.join(CFG.RESULTS_DIR, "logs"))

    if args.status:
        print_pipeline_status()
        return

    # Determine which steps to run
    if args.steps:
        steps_to_run = sorted(args.steps)
    elif args.data_only:
        steps_to_run = [1, 2, 3, 4]
    elif args.train_only:
        steps_to_run = [6, 7, 8]
    elif args.eval_only:
        steps_to_run = [9, 10]
    else:
        steps_to_run = list(STEPS.keys())

    # Build extra args for step 3 if needed
    step3_args = []
    if args.download_gctx:
        step3_args.append("--download")
    if args.extract_expression:
        step3_args.append("--extract")

    logger.info("="*60)
    logger.info("CARDIOTOX-FUSION PIPELINE STARTING")
    logger.info(f"Steps to run: {steps_to_run}")
    logger.info("="*60)
    print_pipeline_status()

    results = {}
    pipeline_start = time.time()

    for step_num in steps_to_run:
        step = STEPS.get(step_num)
        if step is None:
            continue

        # Skip internet-required steps in offline mode
        if args.offline and step["requires_internet"]:
            logger.info(f"Step {step_num} requires internet -- skipping (--offline mode)")
            results[step_num] = "skipped"
            continue

        extra_args = step3_args if step_num == 3 else step.get("extra_args", [])
        success = run_step(step_num, extra_args=extra_args, skip_if_done=not args.force)
        results[step_num] = "success" if success else "failed"

        if not success:
            logger.error(f"Pipeline halted at step {step_num}.")
            logger.error("Fix the error above and resume with --steps {step_num} ... "
                         "or --force to re-run from scratch.")
            break

    # Final summary
    total_time = time.time() - pipeline_start
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    for step_num, status in results.items():
        icon = "OK" if status == "success" else ("-" if status == "skipped" else "X")
        logger.info(f"  {icon} Step {step_num}: {status}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
