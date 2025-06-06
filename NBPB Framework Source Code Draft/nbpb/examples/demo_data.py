# examples/demo_data.py
"""
NBPB Demo Script - Data Pipeline Simulation
Demonstrates data loss and distribution shift detection
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Add NBPB to the path
sys.path.append(str(Path(__file__).parent.parent))
from nbpb import NBPBNucleus, PyTorchHook
from nbpb.config import load_config
from nbpb.utils.logging import setup_nbpb_logging

class MedicalClassifier(nn.Module):
    """Simple data classifier"""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def create_clean_medical_data(n_samples=1000, n_features=20):
    """Creates clean dataset for training"""
    print("Generating clean medical dataset...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        class_sep=0.8,
        random_state=42
    )

    # Simulated medical feature names
    feature_names = [
        'age', 'bmi', 'blood_pressure_sys', 'blood_pressure_dia', 'heart_rate',
        'cholesterol_total', 'cholesterol_hdl', 'cholesterol_ldl', 'glucose',
        'hemoglobin', 'white_blood_cells', 'platelets', 'creatinine', 'urea',
        'sodium', 'potassium', 'weight', 'height', 'smoking_years', 'exercise_hours'
    ]

    return torch.FloatTensor(X), torch.FloatTensor(y), feature_names

def inject_data_leakage(X, y, leakage_strength=0.9):
    """Injects data leakage into dataset"""
    print(f"Injecting data leakage (strength: {leakage_strength})...")

    X_leaky = X.clone()
    X_leaky[:, 0] = y * leakage_strength + torch.randn_like(y) * 0.1

    return X_leaky

def create_shifted_data(X, shift_factor=3.0):
    """Creates data with distribution shift"""
    print(f"Creating distribution shifted data (factor: {shift_factor})...")

    X_shifted = X.clone()
    X_shifted[:, :5] = X_shifted[:, :5] * shift_factor + 2.0
    return X_shifted


def training_step(model, X_batch, y_batch, optimizer, criterion):
    """Single training step"""
    model.train()
    optimizer.zero_grad()
    outputs = model(X_batch).squeeze()
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def run_nbpb_demo():
    """Runs complete NBPB demo"""
    print("=" * 60)
    print("NBPB DEMO - Medical Data Pipeline with Leakage Detection")
    print("=" * 60)

    # Load configuration
    config_path = Path(__file__).parent / "nbpb_config.yaml"
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        print("Make sure nbpb_config.yaml is in the examples/ directory")
        return

    config = load_config(str(config_path))
    print(f"Loaded config: {config.project_name} v{config.version}")

    # Configure NBPB logging
    log_file_setting = getattr(config.logging, 'file', 'nbpb_demo.log')
    setup_nbpb_logging(config=config.logging.__dict__, default_log_file_path=log_file_setting)

    # Create hook (internal parameters can be passed here)
    hook = PyTorchHook()

    # Initialize Nucleus passing callbacks from hook
    nucleus = NBPBNucleus(
        config=config,
        param_adj_callback=getattr(hook, '_adjust_pytorch_parameters', None),
        data_mod_callback=getattr(hook, '_modify_pytorch_data', None)
    )

    # Connect hook to nucleus and pass other resources if needed
    if hasattr(hook, 'set_nucleus'):
        hook.set_nucleus(nucleus)

    # Create medical data
    X_clean, y_clean, feature_names = create_clean_medical_data()

    print(f"\nDataset: {X_clean.shape[0]} samples, {X_clean.shape[1]} features")
    print(f"Features: {feature_names[:5]}...")

    # Initialize model, criterion and optimizer
    model = MedicalClassifier(input_dim=X_clean.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Pass model and optimizer to hook if expected
    if hasattr(hook, 'set_model'):
        hook.set_model(model)
    if hasattr(hook, 'set_optimizer'):
        hook.set_optimizer(optimizer)

    # Attach hook to model and signal training start
    hook.attach_to_model(model)
    hook.start_training()

    print("\n" + "=" * 50)
    print("SCENARIO 1: Training with clean data")
    print("=" * 50)

    # Test with clean data
    clean_ok = hook.monitor_data(X_clean, y_clean, feature_names, "clean_training_data")
    print(f"⋆༺𓆩☠︎︎𓆪༻⋆ Clean data validation: {'PASSED' if clean_ok else 'FAILED'}")

    if clean_ok:
        # Simulate some training steps
        for epoch in range(3):
            indices = torch.randperm(X_clean.shape[0])[:100]  # Random mini-batch
            X_batch, y_batch = X_clean[indices], y_clean[indices]

            loss = training_step(model, X_batch, y_batch, optimizer, criterion)
            print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

    print("\n" + "=" * 50)
    print("SCENARIO 2: Data Leakage Detection")
    print("=" * 50)

    # Test with data leakage
    X_leaky = inject_data_leakage(X_clean, y_clean, leakage_strength=0.95)
    leaky_ok = hook.monitor_data(X_leaky, y_clean, feature_names, "leaky_training_data")

    if leaky_ok:
        print("➃WARNING: Leakage not detected - check ImmunoGuard configuration")
    else:
        print("🌈⃤ BLOCKED: Data leakage detected by ImmunoGuard!")
        print("⛐Training stopped to prevent overfitting")
    print("\n" + "=" * 50)
    print("SCENARIO 3: Distribution Shift Detection")
    print("=" * 50)

    # Test with distribution shift
    X_shifted = create_shifted_data(X_clean, shift_factor=4.0)
    shifted_ok = hook.monitor_data(X_shifted, y_clean, feature_names, "shifted_data")

    # shifted_ok == is True you have  → shift within tolerable limits
    # shifted_ok == is False you have → shift considered severe
    if shifted_ok:
        print("✟ Data shift detected but within acceptable limits")
    else:
        print("🦄CRITICAL: Severe data shift detected by Watcher!")

    print("\n" + "=" * 50)
    print("SCENARIO 4: Health Report & Recommendations")
    print("=" * 50)

    # Generate final report
    training_report = hook.get_training_report()
    health = training_report.get('nbpb_health_report', {})

    print(f"Pipeline Health Score: {health.get('health_score', 0):.3f}")
    print(f"Total Events Logged: {health.get('total_events', 0)}")
    print(f"Critical Events: {health.get('critical_events', 0)}")
    print(f"Warning Events: {health.get('warning_events', 0)}")

    if health.get('critical_events', 0) > 0:
        print("\n🔍 Critical Issues Found:")
        for event in health.get('recent_critical', []):
            print(f"  • {event['organ']}: {event['event_type']}")
            if 'data' in event and 'description' in event['data']:
                print(f"    Description: {event['data']['description']}")

    # Correlation report from ImmunoGuard (if present)
    if 'immuno_correlation_report' in training_report:
        corr_report = training_report['immuno_correlation_report']
        if corr_report.get('high_risk_features'):
            print(f"\n🪅High-risk features (potential leakage):")
            for feat in corr_report['high_risk_features']:
                corr_score = corr_report['target_correlations'].get(feat, 0)
                print(f"•{feat}: correlation = {corr_score:.3f}")

    hook.stop_training()
    hook.cleanup()

    print("\n" + "=" * 60)
    print("DEMO COMPLETED ⋆♱✮♱⋆")
    print("=" * 60)
    print("⩇⩇:⩇⩇ Check 'nbpb.log' for detailed structured logs")
    print("♛ Modify 'nbpb_config.yaml' to adjust detection thresholds")
    print("̤̮Integration guide: See PyTorchHook documentation")

    # Integration example output
    print("\n ༗ Integration Example:")
    print("""
# Your existing PyTorch training loop:
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        # Add NBPB monitoring before processing
        if not hook.monitor_data(data, target, stage=f"epoch_{epoch}_batch_{batch_idx}"):
            print("𖥠 𖥠 Training stopped - data quality issues detected 𖥠𖥠 ")
            break
        # Your normal training code
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    """)
if __name__ == "__main__":
    try:
        run_nbpb_demo()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
