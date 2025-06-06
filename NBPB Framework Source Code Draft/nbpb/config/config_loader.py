import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import os

@dataclass
class OrganConfig:
    """Base configuration for an organ"""
    enabled: bool = False

@dataclass
class WatcherConfig(OrganConfig):
    """Configuration for Watcher organ"""
    drift_threshold: float = 0.1
    schema_validation: bool = True
    memory_threshold_mb: int = 1000

@dataclass
class ImmunoGuardConfig(OrganConfig):
    """Configuration for ImmunoGuard organ"""
    target_correlation_critical_threshold: float = 0.8
    target_correlation_warning_threshold: float = 0.6
    multicollinearity_threshold: float = 0.95
    temporal_check: bool = True

@dataclass
class LossSmithConfig(OrganConfig):
    """Configuration for LossSmith organ (scaffold)"""
    custom_loss_definitions_path: Optional[str] = None

@dataclass
class ReverseEngineConfig(OrganConfig):
    """Configuration for ReverseEngine organ (scaffold)"""
    synthesis_target_metric: Optional[str] = None

@dataclass
class HormoneCtrlConfig(OrganConfig):
    """Configuration for HormoneCtrl organ (scaffold)"""
    enable_dynamic_lr: bool = False
    min_lr: float = 1e-6

@dataclass
class LoggingConfig:
    """Configuration for logging"""
    level: str = "INFO"
    format: str = "json"
    
@dataclass
class OrgansConfig:
    """Configuration for all organs"""
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    immuno_guard: ImmunoGuardConfig = field(default_factory=ImmunoGuardConfig)
    loss_smith: LossSmithConfig = field(default_factory=LossSmithConfig)
    reverse_engine: ReverseEngineConfig = field(default_factory=ReverseEngineConfig)
    hormone_ctrl: HormoneCtrlConfig = field(default_factory=HormoneCtrlConfig)

@dataclass
class NBPBConfig:
    """Main NBPB Configuration"""
    project_name: str = "NBPB_Project"
    version: str = "0.1.0"
    organs: OrgansConfig = field(default_factory=OrgansConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def _create_organ_config(cls, OrganClass: type, config_data: Optional[Dict[str, Any]]) -> Any:
        """Helper to create organ configuration instances with error handling"""
        if config_data is None:
            config_data = {}
        try:
            valid_keys = {f.name for f in OrganClass.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in config_data.items() if k in valid_keys}
            return OrganClass(**filtered_data)
        except TypeError as e:
            raise ValueError(
                f"Configuration type error for {OrganClass.__name__} with data {config_data}: {e}"
            ) from e

    @staticmethod
    def from_dict(cfg: dict) -> "NBPBConfig":
        # Example of loading individual sub-configs from YAML dict
        return NBPBConfig(
            project_name=cfg["project_name"],
            version=cfg["version"],
            watcher=WatcherConfig(**cfg["organs"]["watcher"]),
            immuno_guard=ImmunoGuardConfig(**cfg["organs"]["immuno_guard"]),
            loss_smith=LossSmithConfig(**cfg["organs"]["loss_smith"]),
            reverse_engine=ReverseEngineConfig(**cfg["organs"]["reverse_engine"]),
            hormone_ctrl=HormoneCtrlConfig(**cfg["organs"]["hormone_ctrl"]),
            logging=LoggingConfig(**cfg["logging"]),
        )

def load_config(config_path: str) -> NBPBConfig:
    """
    Load NBPB configuration from YAML file
    Args:
        config_path: Path to YAML configuration file
    Returns:
        NBPBConfig instance
    Raises:
        FileNotFoundError: If configuration file not found
        ValueError: If parsing or type errors in YAML file
        yaml.YAMLError: If YAML file is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"NBPB configuration file not found: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
            if config_dict is None:
                config_dict = {}
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration file {config_path}: {e}") from e
    except Exception as e:
        raise IOError(f"Could not read configuration file {config_path}: {e}") from e

    organs_data = config_dict.get('organs', {})
    if not isinstance(organs_data, dict):
        raise ValueError(f"Invalid 'organs' section in config: expected a dictionary, got {type(organs_data)}")
    try:
        organs_config = OrgansConfig(
            watcher=NBPBConfig._create_organ_config(WatcherConfig, organs_data.get('watcher')),
            immuno_guard=NBPBConfig._create_organ_config(ImmunoGuardConfig, organs_data.get('immuno_guard')),
            loss_smith=NBPBConfig._create_organ_config(LossSmithConfig, organs_data.get('loss_smith')),
            reverse_engine=NBPBConfig._create_organ_config(ReverseEngineConfig, organs_data.get('reverse_engine')),
            hormone_ctrl=NBPBConfig._create_organ_config(HormoneCtrlConfig, organs_data.get('hormone_ctrl'))
        )
    except ValueError as e:
        raise ValueError(f"Error processing 'organs' configuration in {config_path}: {e}") from e
    logging_data = config_dict.get('logging', {})
    if not isinstance(logging_data, dict):
        raise ValueError(f"Invalid 'logging' section in config: expected a dictionary, got {type(logging_data)}")
    logging_config = NBPBConfig._create_organ_config(LoggingConfig, logging_data)
    main_config_data = {k: v for k, v in config_dict.items() if k not in ['organs', 'logging']}
    try:
        return NBPBConfig(
            organs=organs_config,
            logging=logging_config,
            **main_config_data
        )
    except TypeError as e:
        raise ValueError(
            f"Configuration type error for main NBPBConfig with data {main_config_data}: {e}"
        ) from e

def get_default_config() -> NBPBConfig:
    """Returns default NBPB configuration"""
    return NBPBConfig()
