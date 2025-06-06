# nbpb/cli.py
import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import logging
from .config.config_loader import load_config
from .nucleus import NBPBNucleus
from .hooks.pytorch_hook import PyTorchHook
from .utils.logging import get_nbpb_logger

cli_logger = get_nbpb_logger("CLI")

def main():
    """CLI entry point per NBPB"""
    parser = argparse.ArgumentParser(description="NBPB - NeuroBulla Pipeline Bodies")
    parser.add_argument("--config", "-c", type=str, default="nbpb_config.yaml",
                       help="Path to NBPB configuration file")
    parser.add_argument("--demo", action="store_true", 
                       help="Run demo with synthetic data")
    parser.add_argument("--validate-config", action="store_true",
                       help="Validate configuration file")
    parser.add_argument("--health-check", action="store_true",
                       help="Run health check on NBPB system")
    args = parser.parse_args()
    
    try:
        if not Path(args.config).exists():
            cli_logger.error(f"Config file {args.config} not found")
            cli_logger.info("Use --demo to run with default configuration")
            return 1 
        config = load_config(args.config)
        cli_logger.info(f"Loaded NBPB config: {config.project_name} v{config.version}")
        if args.validate_config:
            cli_logger.info("Configuration is valid!")
            return 0
        nucleus = NBPBNucleus(config)
        hook = PyTorchHook(nucleus)
        if args.health_check:
            nucleus.activate()
            health = nucleus.get_health_report()
            cli_logger.info(f"NBPB Health Status: {health}")
            nucleus.deactivate()
            return 0  
        if args.demo:
            run_demo(nucleus, hook)
            return 0
        cli_logger.info("NBPB system ready. Integrate with your PyTorch training loop:")
        cli_logger.info("```python")
        cli_logger.info("from nbpb import NBPBNucleus, PyTorchHook")
        cli_logger.info("from nbpb.config import load_config")
        cli_logger.info("")
        cli_logger.info("config = load_config('nbpb_config.yaml')")
        cli_logger.info("nucleus = NBPBNucleus(config)")
        cli_logger.info("hook = PyTorchHook(nucleus)")
        cli_logger.info("hook.start_training()")
        cli_logger.info("# ... your training loop ...")
        cli_logger.info("hook.stop_training()")
        cli_logger.info("```")
        
    except FileNotFoundError as e:
        cli_logger.error(f"Configuration Error: {e}", exc_info=True)
        return 1
    except ValueError as e:
        cli_logger.error(f"Configuration Error: {e}", exc_info=True)
        return 1
    except Exception as e:
        cli_logger.critical(f"An unexpected error occurred in CLI: {e}", exc_info=True)
        return 1
    return 0


def run_demo(nucleus: NBPBNucleus, hook: PyTorchHook):
    """Eseguo demo con dati sintetici"""
    cli_logger.info("Running NBPB Demo with synthetic medical data...")
    hook.start_training()
    cli_logger.info("\n=== Simulating Clean Data ===")
    clean_features = torch.randn(1000, 20)
    clean_target = torch.randint(0, 2, (1000,)).float()

    feature_names_demo = [f"feat_{i}" for i in range(clean_features.shape[1])]
    result = hook.monitor_data(clean_features, clean_target, feature_names=feature_names_demo, stage="clean_data_demo")
    cli_logger.info(f"Clean data check: {'PASSED' if result else 'FAILED'}")

    cli_logger.info("\n=== Simulating Data Leakage ===")
    leaky_features = torch.randn(1000, 20)
    leaky_target = torch.randint(0, 2, (1000,)).float()
    leaky_features[:, 0] = leaky_target + torch.randn(1000) * 0.01

    result = hook.monitor_data(leaky_features, leaky_target, feature_names=feature_names_demo, stage="leaky_data_demo")
    cli_logger.info(f"Leaky data check: {'PASSED' if result else 'BLOCKED (Expected)'}")

    cli_logger.info("\n=== Simulating Distribution Shift ===")
    shifted_features = torch.randn(1000, 20) * 10 + 50
    shifted_target = torch.randint(0, 2, (1000,)).float()

    result = hook.monitor_data(shifted_features, shifted_target, feature_names=feature_names_demo, stage="shifted_data_demo") 
    cli_logger.info(f"Shifted data check: {'PASSED' if result else 'BLOCKED/WARNED (Expected)'}")

    cli_logger.info("\n=== Simulating Training Step with Loss ===")
    def dummy_train_step(batch_features, batch_target):
        if torch.rand(1).item() < 0.1:
            raise ValueError("A simulated error occurred in training step!")
        loss = torch.rand(1).item() * 5
        class Result: pass
        res = Result()
        res.loss = torch.tensor(loss, dtype=torch.float32)
        return res

    wrapped_step = hook.training_step_wrapper(dummy_train_step)
    try:
        for i in range(3):
            cli_logger.info(f"Demo training step {i+1}")
            step_result = wrapped_step(clean_features[:100], clean_target[:100])
            if hasattr(step_result, 'loss'):
                cli_logger.info(f"  Step {i+1} loss: {step_result.loss.item():.4f}")
    except Exception as e:
        cli_logger.error(f"  Demo training step failed: {e}")

    cli_logger.info("\n=== Final NBPB Health Report ===")
    health_report = nucleus.get_health_report()

    report_str = f"""
Pipeline Health Score: {health_report.get('overall_health_score', 'N/A'):.3f}
Status: {health_report.get('status', 'N/A')}
Uptime: {health_report.get('uptime_human', 'N/A')}
Total Events Processed: {health_report.get('total_events_processed', 0)}
Critical Events: {health_report.get('critical_events_count', 0)}
Warning Events: {health_report.get('warning_events_count', 0)}
Active Organs: {health_report.get('active_organs', [])}
"""
    cli_logger.info(report_str)

    if health_report.get('critical_events_count', 0) > 0 and health_report.get('recent_critical_events'):
        cli_logger.warning("\nRecent Critical Events:")
        for event in health_report['recent_critical_events']:
            cli_logger.warning(f"  - Organ: {event.get('organ_name', 'N/A')}, Type: {event.get('event_type', 'N/A')}, Msg: {event.get('message','N/A')}")

    if health_report.get('system_notes'):
        cli_logger.info("\nSystem Notes:")
        for note in health_report['system_notes']:
            cli_logger.info(f"  - {note}")

    hook.stop_training()
    cli_logger.info("\nNBPB Demo completed! Check log file for detailed logs.")
if __name__ == "__main__":
    sys.exit(main())
