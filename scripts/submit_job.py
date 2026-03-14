"""SLURM job submission helper.

Wraps common submission patterns with pre-flight validation, dependency
chaining, and job monitoring. Prevents wasted GPU hours by checking
prerequisites before submission.

Usage:
    # Submit Phase A training
    python scripts/submit_job.py phaseA

    # Submit Phase B (auto-chains after Phase A)
    python scripts/submit_job.py phaseB

    # Submit batch inference
    python scripts/submit_job.py inference --checkpoint checkpoints_phaseA/final

    # Submit evaluation
    python scripts/submit_job.py evaluate --checkpoint checkpoints_phaseA/final

    # Dry run (show SLURM script without submitting)
    python scripts/submit_job.py phaseA --dry-run

    # Check all prerequisites
    python scripts/submit_job.py check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORK_DIR = Path(__file__).resolve().parent.parent


def check_data_exists(data_dir: str) -> bool:
    """Check that training data directory exists and has files."""
    d = WORK_DIR / data_dir
    if not d.exists():
        return False
    inputs = list(d.glob("*_input.png"))
    return len(inputs) > 0


def check_checkpoint_exists(ckpt_dir: str) -> bool:
    """Check that a checkpoint directory exists."""
    d = WORK_DIR / ckpt_dir
    if not d.exists():
        d = Path(ckpt_dir)  # absolute path
    return d.exists()


def check_conda_env() -> bool:
    """Check that the landmarkdiff conda environment exists."""
    try:
        result = subprocess.run(
            ["conda", "info", "--envs"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return "landmarkdiff" in result.stdout
    except Exception:
        return False


def get_active_jobs() -> list[dict]:
    """Get active SLURM jobs for current user."""
    try:
        result = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "--format=%i %j %t %M", "--noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        jobs = []
        for line in result.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 4:
                jobs.append(
                    {
                        "id": parts[0],
                        "name": parts[1],
                        "state": parts[2],
                        "time": parts[3],
                    }
                )
        return jobs
    except Exception:
        return []


def submit_slurm_script(script_path: str, dry_run: bool = False) -> str | None:
    """Submit a SLURM script and return the job ID."""
    path = WORK_DIR / script_path
    if not path.exists():
        print(f"ERROR: SLURM script not found: {path}")
        return None

    if dry_run:
        print(f"\n--- DRY RUN: {path} ---")
        with open(path) as f:
            print(f.read())
        return "DRY_RUN"

    result = subprocess.run(
        ["sbatch", str(path)],
        capture_output=True,
        text=True,
        cwd=str(WORK_DIR),
    )

    if result.returncode != 0:
        print(f"ERROR: sbatch failed: {result.stderr}")
        return None

    # Parse job ID from "Submitted batch job 12345"
    output = result.stdout.strip()
    job_id = output.split()[-1] if output else None
    return job_id


def submit_inline_job(
    job_name: str,
    command: str,
    partition: str = "batch_gpu",
    account: str = os.environ.get("SLURM_ACCOUNT", "default_gpu"),
    gpu: str = "nvidia_rtx_a6000:1",
    mem: str = "48G",
    cpus: int = 8,
    time_limit: str = "24:00:00",
    dry_run: bool = False,
) -> str | None:
    """Submit an inline SLURM job."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --account={account}
#SBATCH --gres=gpu:{gpu}
#SBATCH --mem={mem}
#SBATCH --cpus-per-task={cpus}
#SBATCH --time={time_limit}
#SBATCH --output=slurm-{job_name}-%j.out

set -euo pipefail
cd "{WORK_DIR}"
source $CONDA_PREFIX/etc/profile.d/conda.sh
conda activate landmarkdiff

echo "Job: {job_name}"
echo "Start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

{command}

echo ""
echo "Done: $(date)"
"""

    if dry_run:
        print(f"\n--- DRY RUN: {job_name} ---")
        print(script)
        return "DRY_RUN"

    # Write temp script
    tmp_script = WORK_DIR / f".tmp_{job_name}.sh"
    with open(tmp_script, "w") as f:
        f.write(script)
    tmp_script.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(tmp_script)],
        capture_output=True,
        text=True,
        cwd=str(WORK_DIR),
    )

    tmp_script.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"ERROR: sbatch failed: {result.stderr}")
        return None

    output = result.stdout.strip()
    job_id = output.split()[-1] if output else None
    return job_id


def cmd_phaseA(args) -> None:
    """Submit Phase A training job."""
    print("Phase A: ControlNet pre-training (diffusion loss only)")

    # Pre-flight checks
    checks = {
        "Training data": check_data_exists("data/training_combined"),
        "Conda env": check_conda_env(),
        "SLURM script": (WORK_DIR / "scripts/train_phaseA_slurm.sh").exists(),
    }

    all_ok = True
    for name, ok in checks.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}")
        if not ok:
            all_ok = False

    if not all_ok:
        print("\nPre-flight checks failed. Fix issues and retry.")
        if not checks["Training data"]:
            print("  Hint: Run build_training_dataset.py first")
        sys.exit(1)

    # Check for active training jobs
    active = get_active_jobs()
    training_jobs = [j for j in active if "phaseA" in j["name"]]
    if training_jobs:
        print(f"\nWARNING: Active Phase A job found: {training_jobs[0]['id']}")
        print("Cancel it first or use --force")
        if not args.force:
            sys.exit(1)

    # Data stats
    data_dir = WORK_DIR / "data" / "training_combined"
    n_pairs = len(list(data_dir.glob("*_input.png")))
    print(f"\n  Training pairs: {n_pairs:,}")

    job_id = submit_slurm_script("scripts/train_phaseA_slurm.sh", args.dry_run)
    if job_id:
        print(f"\n  Submitted Phase A job: {job_id}")
        print(f"  Monitor: tail -f slurm-surgery_phaseA-{job_id}.out")


def cmd_phaseB(args) -> None:
    """Submit Phase B training job."""
    print("Phase B: 4-term loss fine-tuning")

    # Find Phase A checkpoint
    ckpt_dir = WORK_DIR / "checkpoints_phaseA"
    if not ckpt_dir.exists():
        print("  [FAIL] Phase A checkpoints not found")
        print("  Run Phase A training first")
        sys.exit(1)

    final = ckpt_dir / "final"
    if final.exists():
        ckpt = final
    else:
        ckpts = sorted(
            ckpt_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        )
        ckpts = [c for c in ckpts if (c / "controlnet_ema").exists()]
        if not ckpts:
            print("  [FAIL] No valid Phase A checkpoints found")
            sys.exit(1)
        ckpt = ckpts[-1]

    print(f"  [PASS] Phase A checkpoint: {ckpt.name}")

    job_id = submit_slurm_script("scripts/train_phaseB_slurm.sh", args.dry_run)
    if job_id:
        print(f"\n  Submitted Phase B job: {job_id}")


def cmd_inference(args) -> None:
    """Submit batch inference job."""
    if not args.checkpoint:
        print("ERROR: --checkpoint required for inference")
        sys.exit(1)

    if not check_checkpoint_exists(args.checkpoint):
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    test_dir = args.test_dir or "data/test_pairs"
    if not check_data_exists(test_dir):
        print(f"ERROR: Test data not found: {test_dir}")
        sys.exit(1)

    cmd = (
        f"python scripts/batch_inference.py "
        f"--input {test_dir} "
        f"--output results/inference_$(date +%Y%m%d_%H%M) "
        f"--all-procedures --save-intermediates "
        f"--displacement-model data/displacement_model.npz"
    )

    job_id = submit_inline_job(
        "surgery_inference",
        cmd,
        time_limit="4:00:00",
        mem="32G",
        dry_run=args.dry_run,
    )
    if job_id:
        print(f"\n  Submitted inference job: {job_id}")


def cmd_evaluate(args) -> None:
    """Submit evaluation job."""
    if not args.checkpoint:
        print("ERROR: --checkpoint required for evaluation")
        sys.exit(1)

    cmd = (
        f"python scripts/evaluate_checkpoint.py "
        f"--checkpoint {args.checkpoint} "
        f"--data_dir data/training_combined "
        f"--output_dir results/eval_$(date +%Y%m%d_%H%M) "
        f"--save_images --max_samples 200"
    )

    job_id = submit_inline_job(
        "surgery_eval",
        cmd,
        time_limit="2:00:00",
        mem="32G",
        dry_run=args.dry_run,
    )
    if job_id:
        print(f"\n  Submitted evaluation job: {job_id}")


def cmd_check(args) -> None:
    """Run all pre-flight checks without submitting."""
    print("Pre-flight System Check")
    print("=" * 50)

    checks = [
        ("Conda environment", check_conda_env()),
        ("Training data (combined)", check_data_exists("data/training_combined")),
        ("Test data", check_data_exists("data/test_pairs")),
        ("Wave 1 (synthetic)", check_data_exists("data/synthetic_surgery_pairs")),
        ("Wave 2 (scaled)", check_data_exists("data/synthetic_surgery_pairs_v2")),
        ("Wave 3 (realistic)", check_data_exists("data/synthetic_surgery_pairs_v3")),
        ("Real surgery pairs", check_data_exists("data/real_surgery_pairs/pairs")),
        ("Displacement model", (WORK_DIR / "data" / "displacement_model.npz").exists()),
        ("Phase A SLURM script", (WORK_DIR / "scripts" / "train_phaseA_slurm.sh").exists()),
        ("Phase B SLURM script", (WORK_DIR / "scripts" / "train_phaseB_slurm.sh").exists()),
        ("Phase A checkpoints", (WORK_DIR / "checkpoints_phaseA").exists()),
        ("Phase B checkpoints", (WORK_DIR / "checkpoints_phaseB").exists()),
    ]

    for name, ok in checks:
        status = "PASS" if ok else "----"
        print(f"  [{status}] {name}")

    # Data counts
    print("\nData counts:")
    for label, pattern in [
        ("Combined", "data/training_combined/*_input.png"),
        ("Test", "data/test_pairs/*_input.png"),
        ("Wave 1", "data/synthetic_surgery_pairs/*_input.png"),
        ("Wave 2", "data/synthetic_surgery_pairs_v2/*_input.png"),
        ("Wave 3", "data/synthetic_surgery_pairs_v3/*_input.png"),
    ]:
        count = len(list(WORK_DIR.glob(pattern)))
        print(f"  {label}: {count:,} pairs")

    # Active SLURM jobs
    jobs = get_active_jobs()
    surgery_jobs = [j for j in jobs if j["name"].startswith("surgery_")]
    print(f"\nActive jobs: {len(surgery_jobs)}")
    for j in surgery_jobs:
        print(f"  {j['id']} | {j['name']} | {j['state']} | {j['time']}")


def main():
    parser = argparse.ArgumentParser(description="SLURM job submission helper")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show SLURM script without submitting"
    )
    parser.add_argument("--force", action="store_true", help="Force submission even with warnings")
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint path (for inference/evaluate)"
    )
    parser.add_argument("--test_dir", default=None, help="Test data directory")

    subparsers = parser.add_subparsers(dest="command", help="Job type")
    subparsers.add_parser("phaseA", help="Submit Phase A training")
    subparsers.add_parser("phaseB", help="Submit Phase B training")
    subparsers.add_parser("inference", help="Submit batch inference")
    subparsers.add_parser("evaluate", help="Submit model evaluation")
    subparsers.add_parser("check", help="Run pre-flight checks only")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "phaseA": cmd_phaseA,
        "phaseB": cmd_phaseB,
        "inference": cmd_inference,
        "evaluate": cmd_evaluate,
        "check": cmd_check,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
