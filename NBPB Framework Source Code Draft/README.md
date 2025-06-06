
# Structure 

```
nbpb-project/                 
├── nbpb/                    
│   ├── __init__.py
│   ├── nucleus.py
│   ├── organs/
│   │   ├── __init__.py
│   │   ├── base_organ.py
│   │   └── ... (watcher.py, etc.)
│   ├── hooks/
│   │   ├── __init__.py
│   │   └── pytorch_hook.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── config_loader.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   └── metrics.py
│   └── cli.py
├── examples/             
│   ├── demo_data.py
│   └── nbpb_config.yaml
├── tests/                    
│   └── ...
├── .gitignore
├── LICENSE                  
├── README.md                 
├── pyproject.toml           
└── setup.py                 
``` 
# NBPB - NeuroBulla Pipeline Bodies 🧠✨

> A Biologically-Inspired Supervisory Framework for Machine-Learning Pipelines

NBPB introduces a new level of "homeostasis" in Machine Learning systems: a set of independent micro-agents that live inside your ML pipeline, monitor the entire data lifecycle, and intervene to maintain consistency, ethics, and performance. Think of it as a "digital immune system" for your AI.

## Core Concept: The Digital Body for AI

NBPB equips your AI with a "Digital Body" composed of five specialized organs, each ensuring a different aspect of pipeline health:

1.  ⚓ **Watcher**: Sensors that meticulously track every data transformation and statistical property.
2.  ✌︎︎ **Immuno-Guard**: Detects and blocks data leakage, spurious correlations, and other integrity threats.
3. 🪞🪷🌕✨ **Loss-Smith**: Synthesizes custom, adaptive loss functions via `.loss.json` files. *(Coming in v0.2)*
4. 🕴🏻 **Reverse-Engine**: Deduces ideal dataset characteristics from target outputs to guide data augmentation and collection. *(Coming in v0.2)*
5. 𒅒𒈔𒅒𒇫𒄆 **Hormone-Ctrl**: Dynamically regulates learning rates, schedulers, and layer freezing for optimal training homeostasis. *(Coming in v0.2)*

##  Quick Start

### 1. Installation

```bash
pip install nbpb

(Note: NBPB is currently in Alpha. For the latest development version, see the Development section.)

2. Basic Usage (PyTorch Example)
from nbpb import NBPBNucleus, PyTorchHook
from nbpb.config import load_config, NBPBConfig # Assuming NBPBConfig for type hints or defaults

# 1. Load NBPB configuration (or create one)
# config = load_config('path/to/your/nbpb_config.yaml')
# For a quick start, you can use default settings:
config = NBPBConfig() # Uses built-in defaults
# Customize if needed: config.organs.watcher.enabled = True

# 2. Initialize PyTorchHook (it might take its own config in the future)
pt_hook = PyTorchHook()

# 3. Initialize NBPBNucleus, providing callbacks from the hook
# These callbacks allow organs like HormoneCtrl to adjust PyTorch parameters
nucleus = NBPBNucleus(
    config=config,
    param_adj_callback=pt_hook._adjust_pytorch_parameters, # Implement in PyTorchHook
    data_mod_callback=pt_hook._modify_pytorch_data      # Implement in PyTorchHook
)
# 4. Set the nucleus instance for the hook (and other PyTorch components if needed)
pt_hook.set_nucleus(nucleus)
# pt_hook.set_model(your_pytorch_model)
# pt_hook.set_optimizer(your_optimizer)

# --- Integrate with your training loop ---
pt_hook.start_training() # Activates NBPB monitoring

for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        
        # NBPB data monitoring before processing
        # You can also pass feature_names if available: feature_names=your_feature_names
        if not pt_hook.monitor_data(data, target, stage=f"epoch_{epoch}_batch_{batch_idx}"):
            print("NBPB: Training stopped due to critical data quality issues.")
            break
            
        # Your normal training code
        # optimizer.zero_grad()
        # output = model(data)
        # loss = criterion(output, target)
        # loss.backward()
        # optimizer.step()
    if not pt_hook.training_active: # Check if NBPB stopped training
        break

pt_hook.stop_training() # Deactivates NBPB
final_report = pt_hook.get_training_report()
print(final_report)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

📊 Demo & Examples

See NBPB in action! Run the built-in demo script which simulates a medical data pipeline:

# Ensure you have cloned the repo and are in the project root
# Option 1: Using the nbpb CLI (after 'pip install -e .')
nbpb --demo --config examples/nbpb_config.yaml

# Option 2: Running the Python script directly
python examples/demo_medical_data.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

The demo showcases:

ִ ࣪𖤐 Clean data validation

🛡️ Data leakage detection and blocking by Immuno-Guard

📈 Distribution shift monitoring by Watcher

📋 Comprehensive health reporting with actionable insights

Explore the examples/ directory for more usage patterns.

⚙️ Configuration

NBPB is configured via a YAML file (e.g., nbpb_config.yaml). Create one in your project:

project_name: "My_ML_Project_Alpha"
version: "0.1.0" # Your project version, NBPB version is separate

organs:
  watcher:
    enabled: true
    drift_threshold: 0.05 # Lower for more sensitivity
    schema_validation: true
    memory_threshold_mb: 2048
    
  immuno_guard:
    enabled: true
    target_correlation_critical_threshold: 0.8 # Block if target correlation > 80%
    target_correlation_warning_threshold: 0.6  # Warn if target correlation > 60%
    multicollinearity_threshold: 0.95          # Warn for highly correlated features
    temporal_check: false # Enable if you have time-series data

  # Loss-Smith, Reverse-Engine, Hormone-Ctrl are disabled by default in v0.1
  loss_smith:
    enabled: false
  reverse_engine:
    enabled: false
  hormone_ctrl:
    enabled: false

logging:
  level: "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json" # or "text"
  file: "logs/nbpb_project_alpha.log" # Path to log file
  console_output: true # Output logs to console
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Yaml
IGNORE_WHEN_COPYING_END

Refer to the Configuration Reference for all available options.

 𐐘💥╾━╤デ╦︻ඞා Architecture Overview

NBPB operates with a central Nucleus orchestrating several specialized Organs:

┌─────────────────┐  Pipeline Events
│   ML Pipeline   │ ─────────────► ┌────────────────┐
│  (PyTorch/HF)   │                │  NBPB Nucleus  │ Core Orchestrator
└───────┬─────────┘                └────────┬───────┘
        │                                   │ Organ Coordination
        │ Interventions (e.g., stop,        │
        │ adjust params)                    ▼
┌───────┴───────┬───────┴───────┬───────┴───────┬───────┴───────┬─────────┴───┐
│   🌈⃤ Watcher   │  ☕︎ Immuno   │  𖤓 Loss     │  𓆩♱𓆪 Reverse  │  ☠ Hormone  │
│ Data Monitor │  Guard        │ Smith       │ Engine      │   Ctrl      │
└───────────────┴───────────────┴─────────────┴─────────────┴─────────────┘
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
Hooks (like PyTorchHook) integrate NBPB прозрачно into your existing ML framework.

 Key Use Cases

Preventing Data Leakage in Medical AI: Watcher detects distribution shifts from data imputation errors, while Immuno-Guard flags features highly correlated with patient outcomes (e.g., a feature derived from the target), preventing models from learning spurious signals.

Ensuring Robustness in Financial Models: Immuno-Guard's (future) temporal checks can identify features that inadvertently use future information, ensuring models generalize to out-of-sample, out-of-time data.

Maintaining Data Consistency in NLP Pipelines: Watcher monitors token distributions, vocabulary shifts, and text length anomalies across different stages of text pre-processing.

🫡 Current Status (v0.1.0 - Alpha)

Implemented & Ready for Use:

ִ ࣪𖤐 Core Nucleus orchestration system

ִ ࣪𖤐 Watcher Organ: Data drift, schema validation, statistical property monitoring, memory usage.

ִ ࣪𖤐 Immuno-Guard Organ: Target leakage detection, multicollinearity analysis, constant feature identification.

ִ ࣪𖤐 PyTorch Integration Hook: Basic data monitoring, model layer hooks (via attach_to_model), training step wrapping.

ִ ࣪𖤐 YAML-based Configuration System (nbpb_config.yaml loaded by NBPBConfig).

ִ ࣪𖤐 Structured JSON Logging.

ִ ࣪𖤐 Command-Line Interface (nbpb) with demo runner and config validation.

Planned for v0.2.0 (Beta):

𖨆 Loss-Smith Organ: Initial version for custom loss synthesis.

𖨆 Reverse-Engine Organ: Basic capabilities for ideal dataset characteristic deduction.

𖨆 Hormone-Ctrl Organ: First iteration of adaptive hyperparameter tuning (e.g., learning rate).

꒰ঌ ໒꒱ Enhanced HuggingFace Transformers integration.

🐕 Basic web dashboard for real-time monitoring visualization.

🤝 Framework Integration

NBPB aims to be framework-agnostic via its hook system.

PyTorch (Supported in v0.1.0)
# (See Quick Start for more detailed PyTorch integration)
# hook = PyTorchHook(...) 
# hook.attach_to_model(model) # For detailed layer-wise monitoring
# hook.monitor_data(features, target) # For explicit data checks
# wrapped_step = hook.training_step_wrapper(your_training_step_function)
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
HuggingFace Transformers (Planned for v0.2.0)
# from nbpb.hooks import HuggingFaceTrainerHook # Example name
# hf_hook = HuggingFaceTrainerHook(nucleus)
# trainer = Trainer(..., callbacks=[hf_hook.get_hf_callback()])
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
scikit-learn (Planned Post-v0.2.0)
# from nbpb.hooks import SklearnPipelineHook # Example name
# sklearn_hook = SklearnPipelineHook(nucleus)
# pipeline = Pipeline([
#     ('scaler', StandardScaler()),
#     ('nbpb_monitor', sklearn_hook.as_transformer()), # Exposes NBPB as a transformer
#     ('classifier', RandomForestClassifier())
# ])
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
🔍 Monitoring & Alert Examples

NBPB provides real-time alerts and logs for:

Data Quality Violations: Schema changes, null value explosions, memory spikes.

Statistical Drifts: Significant shifts in mean, variance, or overall distribution.

Integrity Threats: Target leakage, high inter-feature correlations, (future) temporal violations.

(Future) Performance Anomalies: Stagnant training, gradient issues, convergence problems.

Example JSON log for a critical alert:

{
  "timestamp": 1719936000.12345,
  "name": "NBPB.ImmunoGuard", 
  "level": "CRITICAL",
  "message": "NBPB Event from ImmunoGuard: Critical leakage detected.",
  "extra_data": {
    "component": "ImmunoGuard",
    "event_type": "critical_leakage_detected",
    "severity": "CRITICAL", 
    "organ_name": "ImmunoGuard",
    "event_data": {
      "stage": "leaky_training_data",
      "critical_alerts": [{
          "leakage_type": "perfect_correlation", 
          "severity": "critical", 
          "correlation_score": 0.9876, 
          "features_involved": ["age"], 
          "description": "Perfect correlation with target: 0.9876", 
          "recommendation": "Remove feature age - likely data leakage"
      }],
      "blocked_features": ["age"],
      "recommendation": "Stop training immediately - data leakage detected"
    },
    "intervention_suggestion": "BLOCK_TRAINING"
  }
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Json
IGNORE_WHEN_COPYING_END
📚 Documentation (Future Structure)

While full documentation is under development, a planned structure includes:

docs/getting_started.md: Installation and basic setup.

docs/configuration.md: Detailed YAML configuration options.

docs/organs/: In-depth explanation of each organ.

docs/hooks/: Guides for PyTorch, HuggingFace, etc.

docs/api/: Auto-generated API reference.

docs/examples/: More complex usage scenarios.

For now, please refer to this README, the example scripts, and the source code comments.

🔧 Development Setup

Ensure you have Python >= 3.8.

# 1. Clone the repository
git clone https://github.com/fra150/NeuroBulla-Pipeline-Bodies-NBPB-.git.git # Replace with your actual repo URL
cd nbpb

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install NBPB in editable mode with development dependencies
pip install -e ".[dev]"

# --- Running Checks & Tools ---
# Run tests (ensure you have test files in tests/)
pytest tests/

# Run the main demo script
python examples/demo_medical_data.py

# Run the CLI demo
nbpb --demo --config examples/nbpb_config.yaml

# Format code with Black
black .

# Lint with Ruff (or Flake8)
ruff check . # or flake8 .

# Type check with MyPy
mypy nbpb/
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

A Makefile is also provided for common development tasks (see Makefile content in your prompt).

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

🎤 How to Cite NBPB

If you use NBPB in your research or projects, please cite it:

@software{bulla2025nbpb,
  title={{NBPB}: {NeuroBulla Pipeline Bodies - A Biologically-Inspired Supervisory Framework for Machine-Learning Pipelines}},
  author={Bulla, Francesco},
  year={2025},
  url={https://github.com/fra150/NeuroBulla-Pipeline-Bodies-NBPB-.git},
  version={0.1.0}
}
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bibtex
IGNORE_WHEN_COPYING_END

"Every AI deserves a body that protects and evolves it." — Francesco Bulla

Ecco la sezione riscritta in **inglese fluente in prima persona**, coerente con il tuo stile e con la tua **NBPB Open License (NBPB-OL V\_1.0)**:

---

## 🧠 NBPB Integration: My Personal Vision

As the creator of **NBPB – NeuroBulla Pipeline Bodies**, my goal has always been to make AI safer, smarter, and biologically inspired in its defense mechanisms. To do that, I’ve designed NBPB as a **framework-agnostic system** using flexible **hook-based integration**.

---

### ✅ PyTorch Support (Available in v0.1.0)

```python
from nbpb.hooks import PyTorchHook

hook = PyTorchHook(...)
hook.attach_to_model(model)  # Layer-level monitoring
hook.monitor_data(features, target)  # For input/output checks
wrapped_step = hook.training_step_wrapper(your_training_step_function)
```

---

### ⊹ ࣪ ﹏𓊝﹏𓂁﹏⊹ ࣪ ˖ HuggingFace Transformers (Coming in v0.2.0)
```python
from nbpb.hooks import HuggingFaceTrainerHook
hf_hook = HuggingFaceTrainerHook(nucleus)
trainer = Trainer(..., callbacks=[hf_hook.get_hf_callback()])
```

---

### ⎚-⎚ scikit-learn (Planned Post-v0.2.0)

```python
from nbpb.hooks import SklearnPipelineHook
sklearn_hook = SklearnPipelineHook(nucleus)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nbpb_monitor', sklearn_hook.as_transformer()),
    ('classifier', RandomForestClassifier())
])
```
---
### 🔍 Real-Time Monitoring & Alerts

NBPB actively detects and reacts to:

* **Data Quality Issues**: schema changes, NaN spikes, memory overflows
* **Statistical Drifts**: shifts in mean/variance/distribution
* **Integrity Threats**: target leakage, inter-feature correlation
* **(Coming soon)**: training stagnation, gradient vanishing, overfitting patterns

Here’s a **sample critical alert JSON** from NBPB:

```json
{
  "timestamp": 1719936000.12345,
  "name": "NBPB.ImmunoGuard",
  "level": "CRITICAL",
  "message": "Critical leakage detected.",
  "extra_data": {
    "organ_name": "ImmunoGuard",
    "event_data": {
      "stage": "leaky_training_data",
      "critical_alerts": [{
        "leakage_type": "perfect_correlation",
        "features_involved": ["age"],
        "description": "Perfect correlation with target: 0.9876"
      }],
      "recommendation": "Stop training immediately"
    }
  }
}
```
---

## 📚 Documentation Roadmap (Under Development)

I’m actively working on full documentation. Here's the planned structure:

* `docs/getting_started.md`: install & initialize
* `docs/configuration.md`: full YAML config options
* `docs/organs/`: in-depth explanation of each organ
* `docs/hooks/`: integration tutorials (PyTorch, HF, etc.)
* `docs/api/`: auto-generated API references
* `docs/examples/`: advanced case studies and testbeds

---

## 🔧 Developer Setup

You can try NBPB today:

```bash
# Clone the repo
git clone https://github.com/fra150/NeuroBulla-Pipeline-Bodies-NBPB-.git
cd nbpb

# Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run demo
python examples/demo_medical_data.py

# CLI run
nbpb --demo --config examples/nbpb_config.yaml

# Code formatting
black .
ruff check .
mypy nbpb/
```

A `Makefile` is also available for common tasks.

---

## 📜 License: NBPB Open License (NBPB-OL v1.0)

NBPB is released under a custom open license that **allows free use for research, personal, and non-commercial purposes**.

➡️ **Commercial usage is prohibited without written permission.**
➡️ **Attribution to me (Francesco Bulla) is always required.**
Read the full license in [`LICENSE`](https://github.com/fra150/NeuroBulla-Pipeline-Bodies-NBPB-.git/blob/main/LICENSE)

---

## 🔬 Citation

If you use NBPB in your work:

```bibtex
@software{bulla2025nbpb,
  title = {{NBPB}: {NeuroBulla Pipeline Bodies – A Biologically-Inspired Supervisory Framework for Machine-Learning Pipelines}},
  author = {Bulla, Francesco},
  year = {2025},
  version = {0.1.0},
  url = {https://github.com/fra150/NeuroBulla-Pipeline-Bodies-NBPB-.git}
}
```
> “Every AI deserves a body that protects and evolves it.”
> — *Francesco Bulla, 2025*
> Collaborators. Stefanie Ewelu 


