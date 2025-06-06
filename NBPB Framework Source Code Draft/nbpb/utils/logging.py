import json
import logging
import logging.config  # Not 
import time  
from typing import Dict, Any, Optional
from pathlib import Path
from ..types import NBPBEvent, EventSeverity, InterventionType

class JsonFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON output.
    """
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": record.created,  
            "asctime": self.formatTime(record, self.datefmt),  # Formatted timestamp string
            "name": record.name,  # Logger name e.g. NBPB.WatcherOrgan
            "level": record.levelname, 
            "message": record.getMessage(),  # Formatted log message
        }
        standard_attrs = (
            'args', 'asctime', 'created', 'exc_info', 'exc_text', 'filename',
            'funcName', 'levelname', 'levelno', 'lineno', 'module',
            'msecs', 'message', 'msg', 'name', 'pathname', 'process',
            'processName', 'relativeCreated', 'stack_info', 'thread', 'threadName'
        )
        
        if hasattr(record, 'extra_data') and isinstance(record.extra_data, dict):
            log_entry.update(record.extra_data)
        
        # Add other non-standard attributes from record
        for key, value in record.__dict__.items():
            if key not in standard_attrs and key not in log_entry and not key.startswith('_'):
                log_entry[key] = value
        if record.exc_info:
            log_entry['exc_info'] = self.formatException(record.exc_info)
        return json.dumps(log_entry, default=str)  # default=str for non-serializable types (e.g. Enum)


_NBPB_ROOT_LOGGER_CONFIGURED = False
def setup_nbpb_logging(
    config: Optional[Dict[str, Any]] = None, 
    default_log_file_path: Optional[str] = "nbpb.log"
):
    """
    Configure logging for the entire NBPB system.
    Should be called only once.
    Args:
        config: A logging configuration dictionary.
               Expected to contain keys like 'level', 'format',
               'file' (for log file path), and 'console_output' (boolean).
               This dictionary typically comes from NBPBConfig.logging.__dict__.
        default_log_file_path: Log file path to use if not specified
                              in configuration.
    """
    global _NBPB_ROOT_LOGGER_CONFIGURED
    nbpb_logger = logging.getLogger("NBPB")
    if _NBPB_ROOT_LOGGER_CONFIGURED and not nbpb_logger.handlers:
        _NBPB_ROOT_LOGGER_CONFIGURED = False
    if _NBPB_ROOT_LOGGER_CONFIGURED:
        nbpb_logger.debug("NBPB logging is already configured. Skipping reconfiguration.")
        return
    if config is None:
        config = {"level": "INFO", "format": "json", "file": default_log_file_path, "console_output": True}
    log_level_str = config.get("level", "INFO").upper()
    numeric_log_level = getattr(logging, log_level_str, logging.INFO)
    nbpb_logger.setLevel(numeric_log_level)
    main_format_type = config.get("format", "json").lower()
    log_file_path_from_config = config.get("file", default_log_file_path)
    if log_file_path_from_config:
        try:
            log_path = Path(log_file_path_from_config)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            file_handler.setLevel(numeric_log_level)
            if main_format_type == "json":
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(
                    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                )
            nbpb_logger.addHandler(file_handler)
        except Exception as e:
            print(f"CRITICAL: Failed to configure file logger for NBPB at {log_file_path_from_config}: {e}")
    if config.get("console_output", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_log_level)
        if main_format_type == "json":
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(name)s)')
            )
        nbpb_logger.addHandler(console_handler)
    if nbpb_logger.hasHandlers():
        nbpb_logger.info(
            f"NBPB Logging configured. Root Level: {log_level_str}, Main Format: {main_format_type}, "
            f"File Output: {log_file_path_from_config if log_file_path_from_config else 'Disabled'}, "
            f"Console Output: {config.get('console_output', True)}"
        )
    else:
        nbpb_logger.addHandler(logging.NullHandler())
        print("WARNING: NBPB logging has no handlers configured (file and console output are disabled).")
    _NBPB_ROOT_LOGGER_CONFIGURED = True


class NBPBLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter to automatically add component name to logs.
    """
    def process(self, msg, kwargs):
        if 'extra' not in kwargs or kwargs['extra'] is None:
            kwargs['extra'] = {}
        if 'extra_data' not in kwargs['extra']:
            kwargs['extra']['extra_data'] = {}
        kwargs['extra']['extra_data']['component'] = self.extra['component_name']
        
        return msg, kwargs


def get_nbpb_logger(component_name: str) -> NBPBLoggerAdapter:
    """
    Get a logger for a NBPB component.
    """
    if not _NBPB_ROOT_LOGGER_CONFIGURED:
        setup_nbpb_logging()
    logger = logging.getLogger(f"NBPB.{component_name}")
    adapter = NBPBLoggerAdapter(logger, {'component_name': component_name})
    return adapter


class NBPBLogger:
    """Logger specialized for NBPB with structured JSON output"""
    def __init__(self, component_name: str):
        self.adapter = get_nbpb_logger(component_name)
    def info(self, message: str, **kwargs):
        self.adapter.info(message, extra={'extra_data': kwargs} if kwargs else None)
    def warning(self, message: str, **kwargs):
        self.adapter.warning(message, extra={'extra_data': kwargs} if kwargs else None)
    def error(self, message: str, exc_info=False, **kwargs):
        self.adapter.error(message, exc_info=exc_info, extra={'extra_data': kwargs} if kwargs else None)
    def critical(self, message: str, exc_info=False, **kwargs):
        self.adapter.critical(message, exc_info=exc_info, extra={'extra_data': kwargs} if kwargs else None)
    def log_event(self, event: NBPBEvent):
        """Specific logging for NBPB events"""
        event_payload = {
            "event_type": event.event_type,
            "severity": event.severity.value if isinstance(event.severity, EventSeverity) else str(event.severity),
            "organ_name": event.organ_name,
            "event_data": event.data,
            "intervention_suggestion": event.intervention_suggestion.value if event.intervention_suggestion else None
        }
        
        message = f"NBPB Event from {event.organ_name}: {event.message}"
        if event.severity == EventSeverity.CRITICAL:
            self.adapter.critical(message, extra={'extra_data': event_payload})
        elif event.severity == EventSeverity.WARNING:
            self.adapter.warning(message, extra={'extra_data': event_payload})
        else:
            self.adapter.info(message, extra={'extra_data': event_payload})
