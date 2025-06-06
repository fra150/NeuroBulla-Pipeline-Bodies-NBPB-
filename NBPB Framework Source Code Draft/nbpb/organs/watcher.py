# nbpb/organs/watcher.py
import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import ks_2samp # Aggiunto per il test di Kolmogorov-Smirnov
from .base_organ import BaseOrgan
from ..types import InterventionType, EventSeverity # Aggiunto EventSeverity

@dataclass
class DataSnapshot:
    """Snapshot dei dati in un momento specifico"""
    timestamp: float
    shape: Tuple[int, ...]
    mean: float
    std: float
    min_val: float
    max_val: float
    null_count: int
    dtype: str
    distribution_hash: str

class WatcherOrgan(BaseOrgan):
    """
    Organo Watcher - qui ho i Sensori che tracciano non solo ogni trasformazione dei dati
    Funzionalita principali:
    - Data drift detection
    - Schema validation
    - Statistical shift monitoring
    - Memory usage tracking
    """
    
    def __init__(self, config, name, nucleus_callback, logger_parent_name=None):
        super().__init__(config=config,
                         name=name,
                         nucleus_callback=nucleus_callback,
                         logger_parent_name=logger_parent_name)
        self.data_history: List[DataSnapshot] = []
        self.baseline_stats: Optional[DataSnapshot] = None
        self.baseline_finite_data: Optional[np.ndarray] = None # Per il test KS
        self.drift_threshold = getattr(config, 'drift_threshold', 0.1)
        self.schema_validation = getattr(config, 'schema_validation', True)
        self.memory_threshold_mb = getattr(config, 'memory_threshold_mb', 1000)
        self.logger.info(f"WatcherOrgan configured with drift_threshold: {self.drift_threshold}, schema_validation: {self.schema_validation}, memory_threshold_mb: {self.memory_threshold_mb}")
        
    def process(self, data: Any, stage: str = "unknown", **kwargs) -> bool:
        """
        Processa qui invece i dati e rilevano  anomalie 
        Args:
            data: Input data (torch.Tensor, numpy.array, etc.)
            stage: Nome dello stage della pipeline
        Returns:
            bool: True se i dati sono ok, False se ci sono problemi critici
        """
        if not self.is_active:
            return True 
        try:
            # Converti a numpy per analisi uniforme
            np_data = self._to_numpy(data)
            flat_np_data = np_data.flatten()
            current_finite_data = flat_np_data[np.isfinite(flat_np_data)]
            
            # Crea snapshot corrente
            snapshot = self._create_snapshot(np_data, current_finite_data) # Passa current_finite_data
            self.data_history.append(snapshot)
            
            # Imposta baseline se e il primo snapshot
            if self.baseline_stats is None:
                self.baseline_stats = snapshot
                self.baseline_finite_data = current_finite_data # Salva i dati finiti per la baseline
                return self._send_event_to_nucleus(
                    event_type="baseline_set",
                    severity=EventSeverity.INFO,
                    message=f"Baseline set for stage {stage}. Shape: {snapshot.shape}, Mean: {snapshot.mean:.2f}",
                    data={
                        "stage": stage,
                        "shape": snapshot.shape,
                        "mean": snapshot.mean,
                        "std": snapshot.std
                    }
                )
                
            # Controlli di validazione
            issues = []
            # 1. Schema validation
            if self.schema_validation:
                schema_issue = self._check_schema_drift(snapshot, stage)
                if schema_issue:
                    issues.append(schema_issue)
                    
            # 2. Statistical drift detection
            stat_drift = self._check_statistical_drift(snapshot, stage, current_finite_data)
            if stat_drift:
                issues.append(stat_drift)
                
            # 3. Memory usage check
            memory_issue = self._check_memory_usage(np_data, stage)
            if memory_issue:
                issues.append(memory_issue)
                
            # 4. Null values explosion
            null_issue = self._check_null_explosion(snapshot, stage)
            if null_issue:
                issues.append(null_issue)
                
            # Invia eventi basato sui problemi trovati
            if issues:
                severity_enum = EventSeverity.CRITICAL if any(issue["severity"] == "critical" or (isinstance(issue["severity"], EventSeverity) and issue["severity"] == EventSeverity.CRITICAL) for issue in issues) else EventSeverity.WARNING
                intervention_enum = InterventionType.BLOCK_TRAINING if severity_enum == EventSeverity.CRITICAL else InterventionType.LOG_WARNING
                
                return self._send_event_to_nucleus(
                    event_type="data_anomaly_detected",
                    severity=severity_enum,
                    message=f"{len(issues)} data anomalies detected in stage {stage}",
                    data={
                        "stage": stage,
                        "issues": issues,
                        "snapshot": {
                            "shape": snapshot.shape,
                            "mean": snapshot.mean,
                            "std": snapshot.std,
                            "null_count": snapshot.null_count
                        }
                    },
                    intervention_suggestion=intervention_enum
                )
            # Tutto ok (ok _ok_ok_ok Luffy)
            return self._send_event_to_nucleus(
                event_type="data_processed",
                severity=EventSeverity.INFO,
                message=f"Data processed for stage {stage}. Shape: {snapshot.shape}, Health: passed",
                data={
                    "stage":stage,
                    "shape":snapshot.shape,
                    "health_check": "passed"
                }
            )
            
        except Exception as e:
            return self._send_event_to_nucleus(
                event_type="watcher_error",
                severity=EventSeverity.CRITICAL,
                message=f"Error in WatcherOrgan during stage {stage}: {str(e)}",
                data={
                    "stage": stage,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                intervention_suggestion=InterventionType.BLOCK_TRAINING
            )
    
    def _to_numpy(self, data: Any) -> np.ndarray:
        """Converte vari tipi di dati a numpy array"""
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            return data
        elif hasattr(data, 'values'):  # pandas
            return data.values
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _create_snapshot(self, data: np.ndarray, finite_data: np.ndarray) -> DataSnapshot:
        """Creo uno snapshot statistico dei dati"""
        return DataSnapshot(
            timestamp=time.time(),
            shape=data.shape,
            mean=np.mean(finite_data) if len(finite_data) > 0 else 0.0,
            std=np.std(finite_data) if len(finite_data) > 0 else 0.0,
            min_val=np.min(finite_data) if len(finite_data) > 0 else 0.0,
            max_val=np.max(finite_data) if len(finite_data) > 0 else 0.0,
            null_count=int(np.sum(~np.isfinite(data.flatten()))), # Calcola su data originale per null count
            dtype=str(data.dtype),
            distribution_hash=self._compute_distribution_hash(finite_data)
        )
    
    def _compute_distribution_hash(self, data: np.ndarray) -> str:
        """Computa un hash della distribuzione per rilevare shift"""
        if len(data) == 0:
            return "empty"
            
        # Usa quantili per catturare la forma della distribuzione che dovrebbe avere dai calcoli fatti dalla pipeline DA 0.1 a 0.9 ( SICURO SONO)
        try:
            quantiles = np.quantile(data, [0.1, 0.25, 0.5, 0.75, 0.9])
            hash_input = f"{quantiles[0]:.3f}_{quantiles[1]:.3f}_{quantiles[2]:.3f}_{quantiles[3]:.3f}_{quantiles[4]:.3f}"
            return hash_input
        except Exception as e:
            self.logger.warning(f"Error computing distribution hash: {e}", exc_info=True)
            return "hash_compute_error"
    
    def _check_schema_drift(self, snapshot: DataSnapshot, stage: str) -> Optional[Dict]:
        """Controlla drift nello schema dei dati"""
        if snapshot.shape != self.baseline_stats.shape:
            return {
                "type": "schema_drift",
                "severity": "critical",
                "description":f"""Shape changed from {self.baseline_stats.shape} to {snapshot.shape}""",
                "baseline_shape": self.baseline_stats.shape,
                "current_shape": snapshot.shape
            }
        if snapshot.dtype != self.baseline_stats.dtype:
            return {
                "type": "dtype_drift", 
                "severity": "warning",
                "description": f"Data type changed from {self.baseline_stats.dtype} to {snapshot.dtype}",
                "baseline_dtype": self.baseline_stats.dtype,
                "current_dtype": snapshot.dtype
            }
            
        return None
    
    def _check_statistical_drift(self, snapshot: DataSnapshot, stage: str, current_finite_data: np.ndarray) -> Optional[Dict]:
        """ControllO lO  drift statistico significativo"""
        if self.baseline_stats and self.baseline_stats.std > 0 and snapshot.std > 0: # Assicurati che std non sia zero
            z_score_mean = abs(snapshot.mean - self.baseline_stats.mean) / self.baseline_stats.std
            if z_score_mean > 3.0:  # 3 sigma rule
                return {
                    "type": "statistical_drift_mean",
                    "severity": EventSeverity.CRITICAL, # Usare Enum
                    "description": f"Mean shifted significantly (z-score: {z_score_mean:.2f})",
                    "baseline_mean": self.baseline_stats.mean,
                    "current_mean": snapshot.mean,
                    "z_score": z_score_mean
                }
        
        # Drift nella varianza
        if self.baseline_stats and self.baseline_stats.std > 0 and snapshot.std > 0: # Assicurati che std non sia zero
            variance_ratio = snapshot.std / self.baseline_stats.std
            if variance_ratio > 2.0 or variance_ratio < 0.5:
                return {
                    "type": "variance_drift",
                    "severity": EventSeverity.WARNING, # Usare Enum
                    "description": f"Variance changed significantly (ratio: {variance_ratio:.2f})",
                    "baseline_std": self.baseline_stats.std,
                    "current_std": snapshot.std,
                    "variance_ratio": variance_ratio
                }

        # Test di Kolmogorov-Smirnov per fare il drift della distribuzione
        if self.baseline_finite_data is not None and len(self.baseline_finite_data) > 0 and len(current_finite_data) > 0:
            try:
                ks_statistic, p_value = ks_2samp(self.baseline_finite_data, current_finite_data)
                if p_value < self.drift_threshold:
                    return {
                        "type": "distribution_drift_ks",
                        "severity": EventSeverity.WARNING, # Usare Enum
                        "description": f"Distribution shifted significantly (KS test p-value: {p_value:.3f} < threshold {self.drift_threshold})",
                        "ks_statistic": ks_statistic,
                        "p_value": p_value,
                        "threshold": self.drift_threshold
                    }
            except Exception as e:
                 self.logger.warning(f"Error during KS test: {e}", exc_info=True)
        return None
    
    def _check_memory_usage(self, data: np.ndarray, stage: str) -> Optional[Dict]:
        """Controlla l'uso della memoria"""
        memory_mb = data.nbytes / (1024 * 1024)
        if memory_mb > self.memory_threshold_mb:
            return {
                "type": "memory_usage",
                "severity": EventSeverity.WARNING, # Usare Enum
                "description": f"High memory usage: {memory_mb:.1f}MB > {self.memory_threshold_mb}MB",
                "memory_mb": memory_mb,
                "threshold_mb": self.memory_threshold_mb
            }
        return None
    
    def _check_null_explosion(self, snapshot: DataSnapshot, stage: str) -> Optional[Dict]:
        """Controlla esplosione di valori null"""
        total_elements = np.prod(snapshot.shape)
        null_ratio = snapshot.null_count / total_elements if total_elements > 0 else 0
        baseline_null_ratio = self.baseline_stats.null_count / np.prod(self.baseline_stats.shape) if np.prod(self.baseline_stats.shape) > 0 else 0
        
        if null_ratio > 0.5:  # qua metto Piu del 50% di null
            return {
                "type": "null_explosion",
                "severity": EventSeverity.CRITICAL, # Usare Enum
                "description": f"Too many null values: {null_ratio:.1%}",
                "null_ratio": null_ratio,
                "null_count": snapshot.null_count,
                "total_elements": total_elements
            }
        elif null_ratio > baseline_null_ratio * 2 and null_ratio > 0.1:  # Raddoppio dei null
            return {
                "type": "null_increase",
                "severity": "warning", # Usare Enum
                "description": f"Null values increased: {baseline_null_ratio:.1%} -> {null_ratio:.1%}",
                "baseline_null_ratio": baseline_null_ratio,
                "current_null_ratio": null_ratio
            }
        return None
    
    def get_data_health_summary(self) -> Dict[str, Any]:
        """Ritorna un summary della salute dei dati"""
        if not self.data_history:
            return {"status": "no_data"}
        latest = self.data_history[-1]
        return {
            "total_snapshots": len(self.data_history),
            "latest_snapshot": {
                "timestamp": latest.timestamp,
                "shape": latest.shape,
                "mean": latest.mean,
                "std": latest.std,
                "null_count": latest.null_count
            },
            "baseline_comparison": {
                "shape_stable": latest.shape == self.baseline_stats.shape if self.baseline_stats else False,
                "dtype_stable": latest.dtype == self.baseline_stats.dtype if self.baseline_stats else False
            } if self.baseline_stats else None
        }