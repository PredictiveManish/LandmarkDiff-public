#!/usr/bin/env python3
"""Automated post-training pipeline.

Runs after training completes to:
1. Analyze training run (convergence, loss curves)
2. Score/rank checkpoints
3. Run evaluation on test split
4. Generate LaTeX tables for paper
5. Export best model

Usage:
    # Full pipeline after Phase A
    python scripts/post_training_pipeline.py --checkpoint_dir checkpoints_phaseA

    # Quick mode (fewer eval samples)
    python scripts/post_training_pipeline.py --checkpoint_dir checkpoints_phaseA --quick

    # Skip evaluation (analysis + export only)
    python scripts/post_training_pipeline.py --checkpoint_dir checkpoints_phaseA --skip-eval

    # Custom output directory
    python scripts/post_training_pipeline.py --checkpoint_dir checkpoints_phaseA --output results/phaseA
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    from scripts.experiment_lineage import LineageDB

    HAS_LINEAGE = True
except ImportError:
    HAS_LINEAGE = False


class PipelineStep:
    """Track a pipeline step's status."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.status = "pending"
        self.result: dict = {}
        self.elapsed: float = 0.0
        self.error: str | None = None

    def run(self, func, *args, **kwargs) -> dict:
        """Execute a pipeline step with timing and error handling."""
        print(f"\n{'─' * 60}")
        print(f"  [{self.name}] {self.description}")
        print(f"{'─' * 60}")

        self.status = "running"
        t0 = time.time()

        try:
            self.result = func(*args, **kwargs) or {}
            self.status = "completed"
        except Exception as e:
            self.error = str(e)
            self.status = "failed"
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

        self.elapsed = time.time() - t0
        status_icon = "OK" if self.status == "completed" else "FAIL"
        print(f"  [{status_icon}] {self.name} ({self.elapsed:.1f}s)")

        return self.result

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "elapsed_s": round(self.elapsed, 1),
            "error": self.error,
            "result": self.result,
        }


def step_analyze_training(checkpoint_dir: str, output_dir: Path) -> dict:
    """Step 1: Analyze the training run."""
    from scripts.analyze_training_run import (
        TrainingMetrics,
        check_phase_transition,
        detect_convergence_issues,
        find_checkpoints,
        generate_report,
        parse_slurm_log,
    )

    ckpt_path = Path(checkpoint_dir)

    # Find and parse training logs
    metrics = TrainingMetrics()
    log_candidates = sorted(
        PROJECT_ROOT.glob("slurm-phase*-*.out"), key=lambda p: p.stat().st_mtime
    )
    if log_candidates:
        log_path = str(log_candidates[-1])
        print(f"  Parsing log: {Path(log_path).name}")
        metrics = parse_slurm_log(log_path)

    # Find checkpoints
    checkpoints = find_checkpoints(str(ckpt_path))
    print(f"  Found {len(checkpoints)} checkpoints")

    # Convergence analysis
    issues = detect_convergence_issues(metrics)
    issue_types = [i["type"] for i in issues]
    print(f"  Convergence: {', '.join(issue_types)}")

    # Phase transition readiness
    transition = check_phase_transition(metrics)
    print(f"  Phase transition ready: {transition['ready']}")

    # Generate report
    report = generate_report(str(ckpt_path), metrics, checkpoints, issues)
    report_path = output_dir / "training_analysis.md"
    report_path.write_text(report)
    print(f"  Report: {report_path}")

    return {
        "n_checkpoints": len(checkpoints),
        "convergence": issue_types,
        "phase_ready": transition["ready"],
        "steps": metrics.steps[-1] if metrics.steps else 0,
        "final_loss": metrics.losses[-1] if metrics.losses else None,
        "report_path": str(report_path),
    }


def step_score_checkpoints(checkpoint_dir: str, output_dir: Path, n_val: int = 20) -> dict:
    """Step 2: Score and rank checkpoints by validation metrics."""
    from scripts.score_checkpoints import (
        compute_tps_score,
        load_val_samples,
        rank_checkpoints,
        score_checkpoint_tps,
    )

    # Load validation samples
    val_dir = PROJECT_ROOT / "data" / "splits" / "val"
    if not val_dir.exists():
        val_dir = PROJECT_ROOT / "data" / "training_combined"

    samples = load_val_samples(str(val_dir), n_val)
    if not samples:
        print("  No validation samples available")
        return {"error": "no_val_samples"}

    print(f"  Loaded {len(samples)} validation samples")

    # Baseline score
    baseline = compute_tps_score(samples)
    print(f"  Baseline SSIM: {baseline['ssim_mean']:.4f}")

    # Score checkpoints
    from scripts.analyze_training_run import find_checkpoints

    checkpoints = find_checkpoints(checkpoint_dir)

    scores = {"baseline": baseline}
    if checkpoints:
        for ckpt in checkpoints:
            name = Path(ckpt["path"]).name
            print(f"  Scoring {name}...")
            score = score_checkpoint_tps(ckpt["path"], samples)
            scores[name] = score

    # Rank
    ranked = rank_checkpoints(scores)
    best = ranked[0] if ranked else ("baseline", 0.0)
    print(f"  Best: {best[0]} (SSIM={best[1]:.4f})")

    # Save scores
    scores_path = output_dir / "checkpoint_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2, default=str)

    return {
        "best_checkpoint": best[0],
        "best_ssim": best[1],
        "baseline_ssim": baseline["ssim_mean"],
        "n_scored": len(scores) - 1,
        "scores_path": str(scores_path),
    }


def step_evaluate_test(checkpoint_dir: str, output_dir: Path, max_samples: int = 0) -> dict:
    """Step 3: Run evaluation on test split."""
    from scripts.run_evaluation import (
        aggregate_metrics,
        evaluate_tps_baseline,
        generate_fitzpatrick_table,
        generate_latex_table,
        generate_per_procedure_table,
        load_test_set,
    )

    # Use test split
    test_dir = PROJECT_ROOT / "data" / "splits" / "test"
    if not test_dir.exists():
        print(f"  Test split not found at {test_dir}")
        return {"error": "no_test_split"}

    samples = load_test_set(test_dir, max_samples)
    if not samples:
        print("  No test samples loaded")
        return {"error": "no_test_samples"}

    print(f"  Loaded {len(samples)} test samples")

    # Run TPS baseline evaluation
    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    print("  Evaluating TPS baseline...")
    tps_results = evaluate_tps_baseline(samples, eval_dir)
    tps_agg = aggregate_metrics(tps_results)

    print(
        f"  TPS Baseline: SSIM={tps_agg.get('ssim_mean', 0):.4f}, LPIPS={tps_agg.get('lpips_mean', 0):.4f}"
    )

    # Generate LaTeX tables
    latex_dir = eval_dir / "latex"
    latex_dir.mkdir(exist_ok=True)

    method_agg = {"TPS_baseline": tps_agg}

    main_table = generate_latex_table(method_agg)
    (latex_dir / "main_results.tex").write_text(main_table)

    proc_table = generate_per_procedure_table(method_agg)
    (latex_dir / "per_procedure.tex").write_text(proc_table)

    fitz_table = generate_fitzpatrick_table(method_agg)
    (latex_dir / "fitzpatrick_equity.tex").write_text(fitz_table)

    print(f"  LaTeX tables: {latex_dir}")

    # Save full results
    results_path = eval_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "test_dir": str(test_dir),
                "n_samples": len(samples),
                "methods": method_agg,
            },
            f,
            indent=2,
            default=str,
        )

    return {
        "n_samples": len(samples),
        "tps_ssim": tps_agg.get("ssim_mean"),
        "tps_lpips": tps_agg.get("lpips_mean"),
        "latex_dir": str(latex_dir),
        "results_path": str(results_path),
    }


def step_export_model(
    checkpoint_dir: str, output_dir: Path, best_checkpoint: str | None = None
) -> dict:
    """Step 4: Export the best model checkpoint."""
    ckpt_path = Path(checkpoint_dir)
    export_dir = output_dir / "export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Find the best checkpoint to export
    source = None
    if best_checkpoint and best_checkpoint != "baseline":
        candidate = ckpt_path / best_checkpoint / "controlnet_ema"
        if candidate.exists():
            source = candidate

    if source is None:
        # Try final checkpoint
        final = ckpt_path / "final" / "controlnet_ema"
        if final.exists():
            source = final
        else:
            # Try highest-numbered checkpoint
            ckpts = sorted(ckpt_path.glob("checkpoint-*/controlnet_ema"))
            if ckpts:
                source = ckpts[-1]

    if source is None:
        print("  No exportable checkpoint found")
        return {"error": "no_checkpoint"}

    # Copy model files
    dest = export_dir / "controlnet_ema"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(source, dest)

    # Create model card
    model_card = f"""# LandmarkDiff ControlNet

## Model Details
- Source: {source}
- Type: ControlNet (SD 1.5 compatible)
- Conditioning: Facial landmark wireframe (MediaPipe 478-point mesh)
- Phase: {"B (4-term loss)" if "phaseB" in str(ckpt_path) else "A (diffusion loss)"}

## Usage
```python
from diffusers import ControlNetModel
controlnet = ControlNetModel.from_pretrained("{dest}")
```
"""
    (export_dir / "README.md").write_text(model_card)

    # Count files
    n_files = len(list(dest.rglob("*")))
    total_size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())

    print(f"  Exported: {dest}")
    print(f"  Files: {n_files}, Size: {total_size / 1e9:.2f}GB")

    return {
        "source": str(source),
        "export_dir": str(dest),
        "n_files": n_files,
        "size_gb": round(total_size / 1e9, 2),
    }


def step_generate_summary(output_dir: Path, steps: list[PipelineStep]) -> dict:
    """Step 5: Generate pipeline summary report."""
    report_lines = [
        "# Post-Training Pipeline Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Pipeline Steps",
        "",
    ]

    for step in steps:
        icon = (
            "PASS" if step.status == "completed" else "FAIL" if step.status == "failed" else "SKIP"
        )
        report_lines.append(f"### [{icon}] {step.name}: {step.description}")
        report_lines.append(f"- Time: {step.elapsed:.1f}s")
        if step.error:
            report_lines.append(f"- Error: {step.error}")
        if step.result:
            for k, v in step.result.items():
                if not k.endswith("_path"):
                    report_lines.append(f"- {k}: {v}")
        report_lines.append("")

    # Key metrics summary
    report_lines.extend(["## Key Metrics", ""])
    for step in steps:
        if step.name == "Evaluate":
            ssim = step.result.get("tps_ssim")
            lpips = step.result.get("tps_lpips")
            if ssim:
                report_lines.append(f"- TPS Baseline SSIM: {ssim:.4f}")
            if lpips:
                report_lines.append(f"- TPS Baseline LPIPS: {lpips:.4f}")
        if step.name == "Score":
            best = step.result.get("best_checkpoint")
            best_ssim = step.result.get("best_ssim")
            if best:
                report_lines.append(f"- Best Checkpoint: {best} (SSIM={best_ssim:.4f})")

    report = "\n".join(report_lines)
    report_path = output_dir / "pipeline_report.md"
    report_path.write_text(report)
    print(f"\n  Pipeline report: {report_path}")

    # Also save as JSON
    pipeline_json = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": [s.to_dict() for s in steps],
        "total_elapsed_s": round(sum(s.elapsed for s in steps), 1),
    }
    json_path = output_dir / "pipeline_results.json"
    with open(json_path, "w") as f:
        json.dump(pipeline_json, f, indent=2, default=str)

    return {"report_path": str(report_path)}


def run_pipeline(
    checkpoint_dir: str,
    output_dir: str = "results/post_training",
    quick: bool = False,
    skip_eval: bool = False,
    skip_export: bool = False,
) -> dict:
    """Run the full post-training pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("  Post-Training Pipeline")
    print(f"{'=' * 60}")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {'quick' if quick else 'full'}")

    steps: list[PipelineStep] = []
    t0 = time.time()

    # Step 1: Analyze training
    analyze_step = PipelineStep("Analyze", "Training run analysis")
    analyze_step.run(step_analyze_training, checkpoint_dir, out)
    steps.append(analyze_step)

    # Step 2: Score checkpoints
    n_val = 10 if quick else 20
    score_step = PipelineStep("Score", "Checkpoint scoring")
    score_result = score_step.run(step_score_checkpoints, checkpoint_dir, out, n_val)
    steps.append(score_step)

    # Step 3: Evaluate on test split
    if not skip_eval:
        max_samples = 30 if quick else 0
        eval_step = PipelineStep("Evaluate", "Test set evaluation")
        eval_result = eval_step.run(step_evaluate_test, checkpoint_dir, out, max_samples)
        steps.append(eval_step)

        # Record evaluation in experiment lineage
        if HAS_LINEAGE and eval_step.status == "completed":
            try:
                db = LineageDB.load()
                results_path = eval_result.get("results_path", "")
                metrics = {}
                if eval_result.get("tps_ssim") is not None:
                    metrics["ssim_mean"] = eval_result["tps_ssim"]
                if eval_result.get("tps_lpips") is not None:
                    metrics["lpips_mean"] = eval_result["tps_lpips"]
                db.record_evaluation(
                    checkpoint_path=checkpoint_dir,
                    results_path=results_path,
                    metrics=metrics,
                    n_samples=eval_result.get("n_samples", 0),
                )
                db.save()
                print("  Lineage: evaluation recorded")
            except Exception as e:
                print(f"  Lineage recording failed: {e}")

    # Step 4: Export best model
    if not skip_export:
        best_ckpt = score_result.get("best_checkpoint") if score_result else None
        export_step = PipelineStep("Export", "Model export")
        export_step.run(step_export_model, checkpoint_dir, out, best_ckpt)
        steps.append(export_step)

    # Step 5: Summary
    summary_step = PipelineStep("Summary", "Pipeline report generation")
    summary_step.run(step_generate_summary, out, steps)
    steps.append(summary_step)

    # Final status
    total = time.time() - t0
    n_passed = sum(1 for s in steps if s.status == "completed")
    n_failed = sum(1 for s in steps if s.status == "failed")

    print(f"\n{'=' * 60}")
    print(f"  Pipeline Complete: {n_passed}/{len(steps)} steps passed ({total:.1f}s)")
    if n_failed:
        print(f"  {n_failed} step(s) FAILED")
    print(f"  Results: {output_dir}")
    print(f"{'=' * 60}")

    return {
        "passed": n_passed,
        "failed": n_failed,
        "total_elapsed_s": round(total, 1),
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-training pipeline")
    parser.add_argument("--checkpoint_dir", required=True, help="Training checkpoint directory")
    parser.add_argument("--output", default="results/post_training", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Quick mode (fewer eval samples)")
    parser.add_argument("--skip-eval", action="store_true", help="Skip test evaluation")
    parser.add_argument("--skip-export", action="store_true", help="Skip model export")
    args = parser.parse_args()

    run_pipeline(
        args.checkpoint_dir,
        args.output,
        args.quick,
        args.skip_eval,
        args.skip_export,
    )
