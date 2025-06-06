from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable
import time
import logging
from ..types import NBPBEvent, InterventionType, EventSeverity

class BaseOrgan(ABC):
    """Base class for all NBPB organs"""
    def __init__(self, 
                 config: Any,
                 name: str,
                 nucleus_callback: Callable[[Dict[str, Any]], bool],
                 logger_parent_name: Optional[str] = None):
        self.config = config
        self.name = name
        self._active = False
        self._activation_time: Optional[float] = None
        self.nucleus_callback = nucleus_callback
        logger_name = f"{logger_parent_name}.{self.name}" if logger_parent_name else f"nbpb.organ.{self.name}"
        self.logger = logging.getLogger(logger_name)
        
        self.logger.info(f"Organ '{self.name}' initialized.")

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Activates the organ"""
        if self._active:
            self.logger.info(f"Organ '{self.name}' is already active.")
            return
        self._active = True
        self._activation_time = time.time()
        self.logger.info(f"Organ '{self.name}' activated.")

    def deactivate(self) -> None:
        """Deactivates the organ"""
        if not self._active:
            self.logger.info(f"Organ '{self.name}' is already inactive.")
            return
        self._active = False
        self.logger.info(f"Organ '{self.name}' deactivated.")
        self._activation_time = None

    def _send_event_to_nucleus(self, 
                             event_type: str, 
                             severity: EventSeverity,
                             message: str,
                             data: Optional[Dict[str, Any]] = None, 
                             intervention_suggestion: Optional[InterventionType] = None) -> bool:
        """Sends event to the Nucleus"""
        if not self.nucleus_callback:
            self.logger.error("Nucleus callback not set. Cannot send event.")
            return True

        if not self._active:
            self.logger.debug(f"Organ '{self.name}' is inactive. Event '{event_type}' not sent.")
            return True

        event_payload = {
            "organ_name": self.name,
            "event_type": event_type,
            "severity": severity.value if isinstance(severity, EventSeverity) else str(severity),
            "message": message,
            "data": data if data is not None else {},
            "intervention_suggestion": intervention_suggestion.value if isinstance(intervention_suggestion, InterventionType) else intervention_suggestion
        }
        
        self.logger.debug(f"Organ '{self.name}' sending event: {event_payload}")
        try:
            return self.nucleus_callback(event_payload)
        except Exception as e:
            self.logger.error(f"Error when sending event to nucleus: {e}", exc_info=True)
            return True

    @abstractmethod
    def process(self, pipeline_data: Any, **kwargs) -> Any:
        """Main processing method of the organ"""
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', active={self.is_active})>"
