# nbpb/nucleus.py
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable, Iterator, Tuple
from dataclasses import asdict
from threading import Lock
import weakref
from .types import NBPBEvent, InterventionType, EventSeverity
from .organs.base_organ import BaseOrgan
from .organs import (
    WatcherOrgan, 
    ImmunoGuardOrgan, 
    LossSmithOrgan, 
    ReverseEngineOrgan, 
    HormoneCtrlOrgan
)
from .utils.logging import NBPBLogger
from .config.config_loader import NBPBConfig

# Type aliases per le callback
PipelineParameterAdjustmentCallback = Callable[[Dict[str, Any]], bool]
PipelineDataModificationCallback = Callable[[Any], Any]


class NBPBNucleus:
    """
    Questo è  il re Il cervello centrale che orchestra tutti gli organi NBPB sia presenti che futuri.
    Gestisce eventi, decisioni di intervento e coordinamento.
    """

    def __init__(self, 
                 config: NBPBConfig, 
                 param_adj_callback: Optional[PipelineParameterAdjustmentCallback] = None,
                 data_mod_callback: Optional[PipelineDataModificationCallback] = None):
        self.config = config
        self.logger = NBPBLogger("NBPB_Nucleus")
        
        # Thread safety per accesso concorrente
        self._lock = Lock()
        self.organs: Dict[str, BaseOrgan] = {}
        self.events_history: List[NBPBEvent] = []
        self.is_active: bool = False
        self.pipeline_health_score: float = 1.0
        self.initial_timestamp: Optional[float] = None
        
        # Statistiche degli eventi
        self._event_stats: Dict[str, int] = {
            "total": 0,
            "critical": 0,
            "warning": 0,
            "info": 0
        }

        # Callback per interagire con la pipeline ML
        self.param_adj_callback = param_adj_callback
        self.data_mod_callback = data_mod_callback

        # Configurazione health score
        self._severity_impact = {
            EventSeverity.CRITICAL: 0.3,
            EventSeverity.WARNING: 0.1,
            EventSeverity.INFO: 0.01
        }
        self._initialize_organs()

    def _initialize_organs(self) -> None:
        """Qui devo Inizializzare solo gli organi abilitati nella configurazione."""
        organ_configs = self._get_organ_configs()

        for name, (OrganClass, organ_config) in organ_configs.items():
            if not organ_config:
                self.logger.debug(f"Organ {name} not configured, skipping.")
                continue
            if not getattr(organ_config, 'enabled', False):
                self.logger.info(f"Organ {name} is disabled in configuration.")
                continue
            try:
                # Usa weak reference per evitare dipendenze circolari
                self.organs[name] = OrganClass(
                    config=organ_config,
                    name=name,
                    nucleus_callback=weakref.WeakMethod(self.receive_event_from_organ),
                    logger_parent_name=self.logger.adapter.logger.name
                )
                self.logger.info(f"Successfully initialized organ: {name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize organ {name}: {e}", exc_info=True)
        self.logger.info(f"Initialized {len(self.organs)} organs: {list(self.organs.keys())}")

    def _get_organ_configs(self) -> Dict[str, Tuple[type, Any]]:
        """Restituisce la configurazione degli organi."""
        return {
            "watcher": (WatcherOrgan, getattr(self.config.organs, 'watcher', None)),
            "immuno_guard": (ImmunoGuardOrgan, getattr(self.config.organs, 'immuno_guard', None)),
            "loss_smith": (LossSmithOrgan, getattr(self.config.organs, 'loss_smith', None)),
            "reverse_engine": (ReverseEngineOrgan, getattr(self.config.organs, 'reverse_engine', None)),
            "hormone_ctrl": (HormoneCtrlOrgan, getattr(self.config.organs, 'hormone_ctrl', None)),
        }

    def activate(self) -> bool:
        """
        Qui si attiva il sistema NBPB e tutti i  suoi organi.
        Returns:
            bool: True se l'attivazione è riuscita, False altrimenti.
        """
        with self._lock:
            if self.is_active:
                self.logger.info("NBPB Nucleus is already active.")
                return True
            self.is_active = True
            self.initial_timestamp = time.time()
            self.pipeline_health_score = 1.0
            self.events_history.clear()
            self._reset_event_stats()
            self.logger.info("NBPB Nucleus activating - monitoring pipeline health.")

            # Attivo sotto  tutti gli organi
            successful_activations = 0
            for organ_name, organ in self.organs.items():
                try:
                    if hasattr(organ, 'activate'):
                        organ.activate()
                        successful_activations += 1
                        self.logger.info(f"Organ {organ_name} activated successfully.")
                    else:
                        self.logger.warning(f"Organ {organ_name} doesn't have activate method.")
                except Exception as e:
                    self.logger.error(f"Error activating organ {organ_name}: {e}", exc_info=True)
            activation_success = successful_activations == len(self.organs)
            if activation_success:
                self.logger.info(f"NBPB Nucleus fully activated with {successful_activations} organs.")
            else:
                self.logger.warning(f"NBPB Nucleus partially activated: {successful_activations}/{len(self.organs)} organs.")
            return activation_success

    def deactivate(self) -> bool:
        """
        Disattiva il sistema NBPB e i suoi organi.
        Returns:
            bool: True se la disattivazione è riuscita, False altrimenti.
        """
        with self._lock:
            if not self.is_active:
                self.logger.info("NBPB Nucleus is already inactive.")
                return True
            self.is_active = False
            self.logger.info("NBPB Nucleus deactivating...")
            successful_deactivations = 0
            for organ_name, organ in self.organs.items():
                try:
                    if hasattr(organ, 'deactivate'):
                        organ.deactivate()
                        successful_deactivations += 1
                        self.logger.info(f"Organ {organ_name} deactivated successfully.")
                    else:
                        self.logger.warning(f"Organ {organ_name} doesn't have deactivate method.")
                except Exception as e:
                    self.logger.error(f"Error deactivating organ {organ_name}: {e}", exc_info=True)

            deactivation_success = successful_deactivations == len(self.organs)
            if deactivation_success:
                self.logger.info("NBPB Nucleus fully deactivated.")
            else:
                self.logger.warning(f"NBPB Nucleus partially deactivated: {successful_deactivations}/{len(self.organs)} organs.")
            return deactivation_success

    def receive_event_from_organ(self, event_data: Dict[str, Any]) -> bool:
        """
        Callback per gli organi per inviare dati di evento al Nucleus.
        Args:
            event_data: Dizionario contenente i dati dell'evento
        Returns:
            bool: True se la pipeline può continuare, False se deve fermarsi
        """
        with self._lock:
            if not self.is_active:
                self.logger.debug(f"Nucleus inactive. Ignoring event from {event_data.get('organ_name', 'unknown')}")
                return True

            # Valido l'evento 
            if not self._validate_event_data(event_data):
                return True
            try:
                event = self._create_event_from_data(event_data)
                self.events_history.append(event)
                self._update_event_stats(event)
                
                # Log strutturato dell'evento
                self.logger.info(
                    f"Event received from {event.organ_name}: {event.message} (Severity: {event.severity.value})", 
                    extra={"event_details": asdict(event)}
                )
                self._update_health_score(event)
                should_continue = self._process_intervention(event)
                
                # Log del sistema periodico (ogni 10 eventi o eventi critici)
                if len(self.events_history) % 10 == 0 or event.severity == EventSeverity.CRITICAL:
                    self._log_system_status()
                return should_continue
                
            except Exception as e:
                self.logger.error(f"Error processing event: {e}", exc_info=True)
                return True  # Continua in caso di errore interno

    def _validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """Valida i dati dell'evento ricevuto."""
        required_keys = ["organ_name", "event_type", "severity", "message"]
        missing_keys = [key for key in required_keys if key not in event_data]
        if missing_keys:
            self.logger.error(f"Invalid event data received. Missing keys: {missing_keys}. Data: {event_data}")
            return False
        return True

    def _create_event_from_data(self, event_data: Dict[str, Any]) -> NBPBEvent:
        """Crea un oggetto NBPBEvent dai dati ricevuti."""
        return NBPBEvent(
            organ_name=event_data["organ_name"],
            event_type=event_data["event_type"],
            severity=event_data["severity"],
            message=event_data["message"],
            data=event_data.get("data", {}),
            intervention_suggestion=event_data.get("intervention_suggestion")
        )

    def _update_event_stats(self, event: NBPBEvent) -> None:
        """Aggiorna le statistiche degli eventi."""
        self._event_stats["total"] += 1
        self._event_stats[event.severity.value] += 1

    def _reset_event_stats(self) -> None:
        """Resetta le statistiche degli eventi."""
        for key in self._event_stats:
            self._event_stats[key] = 0

    def _update_health_score(self, event: NBPBEvent) -> None:
        """Aggiorna il punteggio di salute della pipeline basato sulla severità dell'evento."""
        impact = self._severity_impact.get(event.severity, 0.0)
        self.pipeline_health_score *= (1 - impact)
        self.pipeline_health_score = max(0.0, min(1.0, self.pipeline_health_score))
        
        self.logger.debug(f"Pipeline health score updated to: {self.pipeline_health_score:.3f} after {event.severity.value} event")

    def _process_intervention(self, event: NBPBEvent) -> bool:
        """
        Processa l'intervento necessario basato sul suggerimento dell'evento.
        Returns:
            bool: True se la pipeline può continuare, False se deve fermarsi
        """
        intervention = event.intervention_suggestion
        if not intervention or intervention == InterventionType.NO_INTERVENTION:
            return True
        intervention_handlers = {
            InterventionType.BLOCK_TRAINING: self._handle_block_training,
            InterventionType.ADJUST_PARAMS: self._handle_adjust_params,
            InterventionType.MODIFY_DATA: self._handle_modify_data,
            InterventionType.LOG_WARNING: self._handle_log_warning
        }

        handler = intervention_handlers.get(intervention)
        if handler:
            return handler(event)
        else:
            self.logger.warning(f"Unknown intervention type: {intervention}")
            return True

    def _handle_block_training(self, event: NBPBEvent) -> bool:
        """Gestisce l'intervento di blocco del training."""
        self.logger.critical(
            f"INTERVENTION: BLOCKING PIPELINE due to critical event from {event.organ_name}. "
            f"Event: {event.message}. Data: {json.dumps(event.data, default=str)}"
        )
        return False

    def _handle_adjust_params(self, event: NBPBEvent) -> bool:
        """Gestisce l'intervento di aggiustamento parametri."""
        self.logger.warning(
            f"INTERVENTION: ADJUSTING PARAMS suggested by {event.organ_name} for event: {event.message}"
        )
        
        if not self.param_adj_callback:
            self.logger.warning("No parameter adjustment callback configured in Nucleus.")
            return True

        params_to_adjust = event.data.get('params_to_adjust')
        if not params_to_adjust:
            self.logger.warning("No parameters specified for adjustment.")
            return True

        try:
            success = self.param_adj_callback(params_to_adjust)
            if success:
                self.logger.info("Pipeline parameters adjusted successfully.")
            else:
                self.logger.warning("Pipeline parameter adjustment returned failure status.")
        except Exception as e:
            self.logger.error(f"Failed to adjust pipeline parameters: {e}", exc_info=True)

        return True

    def _handle_modify_data(self, event: NBPBEvent) -> bool:
        """Gestisce l'intervento di modifica dati."""
        self.logger.warning(
            f"INTERVENTION: MODIFYING DATA suggested by {event.organ_name} for event: {event.message}"
        )
        
        if not self.data_mod_callback:
            self.logger.warning("No data modification callback configured in Nucleus.")
            return True
        self.logger.info("Data modification intervention acknowledged but not fully implemented.")
        return True

    def _handle_log_warning(self, event: NBPBEvent) -> bool:
        """Gestisco l'intervento dei log warning."""
        self.logger.warning(
            f"WARNING logged from {event.organ_name}: {event.message}. "
            f"Data: {json.dumps(event.data, default=str)}"
        )
        return True

    def _log_system_status(self) -> None:
        """Qui si logga lo stato corrente del sistema."""
        if not self.initial_timestamp:
            return

        uptime = time.time() - self.initial_timestamp
        recent_events = [
            e for e in self.events_history 
            if time.time() - e.timestamp < 300  # Ultimi 5 minuti
        ]
        status = {
            "timestamp": time.time(),
            "is_active": self.is_active,
            "pipeline_health_score": round(self.pipeline_health_score, 3),
            "active_organs_count": len([o for o in self.organs.values() if getattr(o, 'is_active', True)]),
            "total_organs": len(self.organs),
            "uptime_seconds": round(uptime, 2),
            "event_stats": self._event_stats.copy(),
            "recent_events_5min": len(recent_events)
        }
        
        self.logger.info("NBPB System Status", extra={"system_status": status})

    def get_health_report(self) -> Dict[str, Any]:
        """
        Genera un report dettagliato della salute e dell'attività del sistema.
        Returns:
            Dict contenente il report completo del sistema
        """
        with self._lock:
            if not self.initial_timestamp:
                return {
                    "status": "Never Activated",
                    "message": "NBPB Nucleus has not been activated in this session.",
                    "timestamp": time.time()
                }

            uptime = time.time() - self.initial_timestamp
            critical_events = [e for e in self.events_history if e.severity == EventSeverity.CRITICAL]
            warning_events = [e for e in self.events_history if e.severity == EventSeverity.WARNING]

            report = {
                "overall_health_score": round(self.pipeline_health_score, 3),
                "status": "Active" if self.is_active else "Inactive",
                "start_time_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(self.initial_timestamp)),
                "current_time_iso": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time())),
                "uptime_seconds": round(uptime, 2),
                "uptime_human": self._format_uptime(uptime),
                "event_statistics": self._event_stats.copy(),
                "critical_events_count": len(critical_events),
                "warning_events_count": len(warning_events),
                "active_organs": list(self.organs.keys()),
                "organs_status": {
                    name: {
                        "active": getattr(organ, 'is_active', True),
                        "class": organ.__class__.__name__
                    }
                    for name, organ in self.organs.items()
                },
                "configuration_summary": self._get_config_summary(),
                "recent_critical_events": [asdict(e) for e in critical_events[-5:]],
                "recent_warning_events": [asdict(e) for e in warning_events[-5:]],
                "system_notes": self._get_system_notes()
            }
            self.logger.info("Health report generated.")
            return report

    def _format_uptime(self, seconds: float) -> str:
        """Formatta l'uptime in formato human-readable."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _get_config_summary(self) -> Dict[str, Any]:
        """Genera un riassunto della configurazione."""
        enabled_organs = []
        for name, organ_config in self._get_organ_configs().items():
            if organ_config[1] and getattr(organ_config[1], 'enabled', False):
                enabled_organs.append(name)
        
        return {
            "organs_enabled": enabled_organs,
            "total_organs_configured": len([c for c in self._get_organ_configs().values() if c[1]]),
            "callbacks_configured": {
                "parameter_adjustment": self.param_adj_callback is not None,
                "data_modification": self.data_mod_callback is not None
            }
        }

    def _get_system_notes(self) -> List[str]:
        """Genera note di sistema per potenziali problemi di configurazione."""
        notes = []
        
        # Controlla callback mancanti
        organs_need_param_adj = any(
            getattr(organ.config, 'suggests_param_adjustments', False) 
            for organ in self.organs.values() 
            if hasattr(organ, 'config')
        )
        if organs_need_param_adj and not self.param_adj_callback:
            notes.append("Parameter adjustment callback not set, but some organs might suggest it.")

        organs_need_data_mod = any(
            getattr(organ.config, 'suggests_data_modification', False) 
            for organ in self.organs.values() 
            if hasattr(organ, 'config')
        )
        if organs_need_data_mod and not self.data_mod_callback:
            notes.append("Data modification callback not set, but some organs might suggest it.")

        # Controlla health score basso
        if self.pipeline_health_score < 0.5:
            notes.append(f"Low pipeline health score detected: {self.pipeline_health_score:.3f}")

        # Controlla eventi critici recenti
        recent_critical = [
            e for e in self.events_history 
            if e.severity == EventSeverity.CRITICAL and time.time() - e.timestamp < 300
        ]
        if recent_critical:
            notes.append(f"{len(recent_critical)} critical events in the last 5 minutes.")

        return notes

    def get_events_log(self, 
                      last_n: Optional[int] = None, 
                      severity_filter: Optional[EventSeverity] = None,
                      organ_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Qui mi Restituisce una lista filtrata degli eventi registrati.
        Args:
            last_n: Numero di eventi più recenti da restituire
            severity_filter: Filtra per severità specifica
            organ_filter: Filtra per organo specifico
        Returns:
            Lista di eventi in formato dizionario
        """
        with self._lock:
            events = self.events_history.copy()
            if severity_filter:# App filter 
                events = [e for e in events if e.severity == severity_filter]
            if organ_filter:
                events = [e for e in events if e.organ_name == organ_filter]
            if last_n is not None and last_n > 0: # Limit event  
                events = events[-last_n:]
            return [asdict(e) for e in events]

    def clear_events_history(self) -> int:
        """
        Pulisce la cronologia degli eventi.
        Returns:
            int: Numero di eventi rimossi
        """
        with self._lock:
            count = len(self.events_history)
            self.events_history.clear()
            self._reset_event_stats()
            self.logger.info(f"Cleared {count} events from history.")
            return count

    def __enter__(self):
        """Context manager entry."""
        self.activate()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.deactivate()

    def _safe_serialize_config(self, config) -> Dict[str, Any]:
        """Safely serialize organ config to dict."""
        try:
            if hasattr(config, '__dict__'):
                return asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
            else:
                return {"config_str": str(config)}
        except Exception as e:
            return {"serialization_error": str(e)}

    def generate_health_report(self, output_path: str) -> Dict[str, Any]:
        """
        Genera un report completo dello stato di salute del sistema NBPB.
        Args:
            output_path: Percorso del file dove salvare il report 
        Returns:
            Dict contenente il report di salute
        """
        try:
            with self._lock:
                # Build report step by step with error handling
                report = {
                    "timestamp": time.time(),
                    "nucleus_status": {
                        "is_active": self.is_active,
                        "health_score": self.pipeline_health_score,
                        "organs_count": len(self.organs),
                        "active_organs": [name for name, organ in self.organs.items() if getattr(organ, 'is_active', True)]
                    },
                    "event_statistics": self._event_stats.copy() if hasattr(self, '_event_stats') else {},
                    "organs_status": {},
                    "recent_events": [],
                    "config_summary": {
                        "project_name": getattr(self.config, 'project_name', 'Unknown'),
                        "version": getattr(self.config, 'version', 'Unknown')
                    }
                }
                
                # Safely add organs status
                try:
                    for name, organ in self.organs.items():
                        report["organs_status"][name] = {
                            "is_active": getattr(organ, 'is_active', True),
                            "config": self._safe_serialize_config(organ.config)
                        }
                except Exception as e:
                    self.logger.warning(f"Error serializing organs status: {e}")
                    report["organs_status"] = {"error": str(e)}
                
                # Safely add recent events
                try:
                    report["recent_events"] = self.get_events_log(last_n=50)
                except Exception as e:
                    self.logger.warning(f"Error getting events log: {e}")
                    report["recent_events"] = [{"error": str(e)}]
                
                # Save report to file
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
                    self.logger.info(f"Health report saved to: {output_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save health report to {output_path}: {e}")
                    # Try to save a minimal report
                    minimal_report = {
                        "timestamp": time.time(),
                        "error": f"Failed to generate full report: {str(e)}",
                        "nucleus_active": self.is_active
                    }
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(minimal_report, f, indent=2, ensure_ascii=False)
                    
                return report
                
        except Exception as e:
            self.logger.error(f"Critical error in generate_health_report: {e}")
            # Return minimal report
            return {
                "timestamp": time.time(),
                "error": f"Critical error: {str(e)}",
                "nucleus_active": getattr(self, 'is_active', False)
            }
    def __repr__(self) -> str:
        """Rappresentazione string del Nucleus."""
        status = "Active" if self.is_active else "Inactive"
        return f"NBPBNucleus(status={status}, organs={len(self.organs)}, health={self.pipeline_health_score:.3f})"