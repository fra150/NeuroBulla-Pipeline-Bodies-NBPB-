# nbpb/types.py
"""
Shared types and enums for NBPB framework.
This module contains common types used across the framework to avoid circular imports.
"""
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

class InterventionType(Enum):
    """Tipi di intervento che il sistema può suggerire."""
    BLOCK_TRAINING = "block_training"
    ADJUST_PARAMS = "adjust_params"
    LOG_WARNING = "log_warning"
    MODIFY_DATA = "modify_data"
    NO_INTERVENTION = "no_intervention"


class EventSeverity(Enum):
    """Livelli di severità degli eventi."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class NBPBEvent:
    """Rappresenta un evento nel sistema NBPB."""
    organ_name: str
    event_type: str
    severity: EventSeverity
    message: str
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    intervention_suggestion: Optional[InterventionType] = None