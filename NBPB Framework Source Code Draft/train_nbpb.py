#!/usr/bin/env python3
"""
NBPB Framework Integration Test Script
This script demonstrates how to integrate NBPB into a real training pipeline
and tests all organ functionalities as outlined in the verification plan.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from nbpb import NBPBNucleus, PyTorchHook
from nbpb.config.config_loader import load_config
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
from datetime import datetime

class SimpleModel(nn.Module):
    """Simple neural network for testing"""
    def __init__(self, input_size=10, hidden_size=20, output_size=1):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def create_test_data(n_samples=1000, n_features=10, imbalanced=False, inject_leakage=False, distribution_shift=False):
    """Create test datasets for different scenarios"""
    np.random.seed(42)
    
    # Base data generation
    X = np.random.randn(n_samples, n_features)
    if imbalanced:
        # Create imbalanced dataset (90/10 split)
        y = np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1])
    else:
        y = np.random.choice([0, 1], size=n_samples, p=[0.5, 0.5])
    if inject_leakage:
        # Inject data leakage by making a feature highly correlated with target
        leakage_strength = 0.95
        X[:, 0] = y + np.random.normal(0, 0.1, n_samples) * (1 - leakage_strength)
        print(f"✪ Injected data leakage with strength {leakage_strength}")
    if distribution_shift:
        # Create distribution shift by scaling features
        shift_factor = 6.0  # This should trigger the watcher
        X = X * shift_factor
        print(f"🪽 Applied distribution shift with factor {shift_factor}")
    
    return X.astype(np.float32), y.astype(np.float32)

def test_scenario(scenario_name, nucleus, hook, model, criterion, optimizer, X, y, epochs=3):
    """Test a specific scenario"""
    print(f"\n{'='*60}")
    print(f"⋆༺𓆩☠︎︎𓆪༻⋆Testing Scenario: {scenario_name}")
    print(f"{'='*60}")
    
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    training_stopped = False
    
    for epoch in range(epochs):
        print(f"\n🦄Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        batch_count = 0
        
        for batch_idx, (data, target) in enumerate(loader):
            # Monitor data with NBPB before forward pass
            stage_name = f"epoch_{epoch}_batch_{batch_idx}"
            
            if not hook.monitor_data(data.numpy(), target.numpy(), stage=stage_name):
                print(f"🃏Training stopped by NBPB at {stage_name}: data quality issues detected")
                training_stopped = True
                break
            # Standard training step
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.squeeze(), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
        
        if training_stopped:
            break
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        print(f"  Average Loss: {avg_loss:.4f}")
        
        # End of epoch hook for hormone control
        hook.end_of_epoch(epoch)
    return not training_stopped

def main():
    """Main testing function"""
    print("𒉭Starting NBPB Framework Comprehensive Testing")
    print(f"Timestamp: {datetime.now()}")
    
    # Initialize NBPB
    config_path = "nbpb/examples/nbpb_config.yaml"
    print(f"\n💀Loading NBPB configuration from: {config_path}")
    
    try:
        config = load_config(config_path)
        nucleus = NBPBNucleus(config=config)
        hook = PyTorchHook(nucleus)
        print("👒🏴‍☠️☠🍖NBPB Nucleus initialized successfully")
        print(f"   Active organs: {len(nucleus.organs)}")
    except Exception as e:
        print(f"╰┈➤ Failed to initialize NBPB: {e}")
        return False
    # Initialize model
    model = SimpleModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # Test scenarios as outlined in the verification plan
    test_results = {}
    
    # 1. Test ImmunoGuard - Data Leakage Detection
    print("\n Monkey•D•Luffy Test 1: ImmunoGuard - Data Leakage Detection")
    X_leakage, y_leakage = create_test_data(n_samples=500, inject_leakage=True)
    test_results['leakage_detection'] = test_scenario(
        "Data Leakage Detection", nucleus, hook, model, criterion, optimizer, 
        X_leakage, y_leakage, epochs=2
    )
    
    # 2. Test Watcher - Distribution Shift Detection
    print("\nᎪS̷ϾᎬ Test 2: Watcher - Distribution Shift Detection")
    X_shift, y_shift = create_test_data(n_samples=500, distribution_shift=True)
    test_results['distribution_shift'] = test_scenario(
        "Distribution Shift Detection", nucleus, hook, model, criterion, optimizer,
        X_shift, y_shift, epochs=2
    )
    
    # 3. Test Loss-Smith - Imbalanced Dataset
    print("\n°• Luffy🏴‍☠️ •°Test 3: Loss-Smith - Imbalanced Dataset Handling")
    X_imbalanced, y_imbalanced = create_test_data(n_samples=500, imbalanced=True)
    test_results['imbalanced_data'] = test_scenario(
        "Imbalanced Dataset Handling", nucleus, hook, model, criterion, optimizer,
        X_imbalanced, y_imbalanced, epochs=3
    )
    
    # 4. Test Normal Training (all organs active)
    print("\n🍥🍜🦊Naruto Test 4: Normal Training - All Organs Active")
    X_normal, y_normal = create_test_data(n_samples=500)
    test_results['normal_training'] = test_scenario(
        "Normal Training", nucleus, hook, model, criterion, optimizer,
        X_normal, y_normal, epochs=3
    )
    
    # Generate final health report with timeout
    print("\n📊 Generating final health report...")
    try:
        import threading
        import time
        report_path = "health_final_report.json"
        result = [None]
        exception = [None]
        def generate_report():
            try:
                result[0] = nucleus.generate_health_report(report_path)
            except Exception as e:
                exception[0] = e
        
        # Start report generation in a separate thread
        thread = threading.Thread(target=generate_report)
        thread.daemon = True
        thread.start()
        thread.join(timeout=15)  # 15 second timeout 
        if thread.is_alive():
            print(f"𐕣 𖤐 𐕣 Health report generation timed out after 15 seconds")
            # Create a minimal report manually
            minimal_report = {
                "timestamp": time.time(),
                "status": "timeout_during_generation",
                "nucleus_active": nucleus.is_active,
                "organs_count": len(nucleus.organs)
            }
            import json
            with open(report_path, 'w') as f:
                json.dump(minimal_report, f, indent=2)
            print(f"ᯓ  ✈︎   ▌ ▌Minimal health report saved to: {report_path}")
        elif exception[0]:
            raise exception[0]
        else:
            print(f"✇ Health report saved to: {report_path}")
            
    except Exception as e:
        print(f"Could not generate health report: {e}")
        # Try to create a basic error report
        try:
            error_report = {
                "timestamp": time.time(),
                "error": str(e),
                "status": "generation_failed"
            }
            import json
            with open("health_error_report.json", 'w') as f:
                json.dump(error_report, f, indent=2)
            print("Nika𓆩☠︎︎𓆪 : Error report saved to: health_error_report.json")
        except:
            pass
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, success in test_results.items():
        status = "⁀➴PASSED" if success else "❌ FAILED (Expected for some tests)"
        print(f"  {test_name}: {status}")
    
    print(f"\n🔥 Testing completed at {datetime.now()}")
    print("\n 🦈๋࣭⭑Check the following files for detailed logs:")
    print("- nbpb_demo.log (NBPB organ logs)")
    print("- health_final_report.json (Final health report)")
    
    return True
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)