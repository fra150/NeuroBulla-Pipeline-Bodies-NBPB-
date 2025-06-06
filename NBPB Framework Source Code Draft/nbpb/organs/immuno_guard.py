import numpy as np
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.metrics import mutual_info_score
from scipy.stats import pearsonr, spearmanr
import pandas as pd
from .base_organ import BaseOrgan
from ..types import InterventionType, EventSeverity

@dataclass
class LeakageAlert:
    """Alert per data leakage rilevato"""
    leakage_type: str
    severity: str
    correlation_score: float
    features_involved: List[str]
    description: str
    recommendation: str

class ImmunoGuardOrgan(BaseOrgan):
    """
    Organo Immuno-Guard - Rileva e blocca leakage o correlazioni spurie    
    Funzionalita principali:
    - Target leakage detection
    - Feature correlation analysis
    - Temporal leakage detection
    - Data contamination prevention
    """
    def __init__(self, config, name, nucleus_callback, logger_parent_name=None):
        super().__init__(config=config,
                        name=name,
                        nucleus_callback=nucleus_callback,
                        logger_parent_name=logger_parent_name)
        self.correlation_threshold = getattr(config, 'correlation_threshold', 0.05)
        self.mutual_info_threshold = getattr(config, 'mutual_info_threshold', 0.3)
        self.temporal_check = getattr(config, 'temporal_check', True)
        self.feature_correlations = {}
        self.target_correlations = {}
        self.blocked_features = set()
        self.logger.info(f"ImmunoGuardOrgan configured with correlation_threshold: {self.correlation_threshold}, temporal_check: {self.temporal_check}")
        
    def process(self, features: Any, target: Any = None, feature_names: List[str] = None, stage: str = "training", **kwargs) -> bool:
        """
        Analizza features e target per rilevare leakage 
        Args:
            features: Feature matrix
            target: Target values (opzionale)
            feature_names: Nomi delle features
            stage: Fase della pipeline
        Returns:
            bool: True se safe, False se leakage critico rilevato
        """
        if not self.is_active:
            return True
        try:
            # Converti a numpy
            X = self._to_numpy(features)
            y = self._to_numpy(target) if target is not None else None
            alerts = []
            
            # 1. Target leakage detection (se abbiamo il target)
            if y is not None:
                target_alerts = self._detect_target_leakage(X, y, feature_names, stage)
                alerts.extend(target_alerts)
                
            # 2. Feature multicollinearity
            collinearity_alerts = self._detect_multicollinearity(X, feature_names, stage)
            alerts.extend(collinearity_alerts)
            
            # 3. Temporal leakage (se richiesto)
            if self.temporal_check:
                temporal_alerts = self._detect_temporal_leakage(X, feature_names, stage)
                alerts.extend(temporal_alerts)
                
            # 4. Constant/quasi-constant features
            constant_alerts = self._detect_constant_features(X, feature_names, stage)
            alerts.extend(constant_alerts)
            
            # Processamento degli alert
            critical_alerts = [a for a in alerts if a.severity == "critical"]
            warning_alerts = [a for a in alerts if a.severity == "warning"]
            
            if critical_alerts:
                # Blocca training per leakage critico
                self._send_event_to_nucleus(
                    event_type="critical_leakage_detected",
                    severity=EventSeverity.CRITICAL,
                    message=f"{len(critical_alerts)} critical leakage alerts detected in stage {stage}",
                    data={
                        "stage": stage,
                        "critical_alerts": [self._alert_to_dict(a) for a in critical_alerts],
                        "blocked_features": list(self.blocked_features),
                        "recommendation": "Stop training immediately - data leakage detected"
                    },
                    intervention_suggestion=InterventionType.BLOCK_TRAINING
                )
                return False
                
            elif warning_alerts:
                # Warning per correlazioni sospette
                self._send_event_to_nucleus(
                    event_type="suspicious_correlations",
                    severity=EventSeverity.WARNING,
                    message=f"{len(warning_alerts)} suspicious correlations detected in stage {stage}",
                    data={
                        "stage": stage,
                        "warning_alerts": [self._alert_to_dict(a) for a in warning_alerts],
                        "recommendation": "Review feature engineering"
                    },
                    intervention_suggestion=InterventionType.LOG_WARNING
                )
                
            # Tutto ok Cazzo Cazzo Cazzo See 
            self._send_event_to_nucleus(
                event_type="leakage_check_passed",
                severity=EventSeverity.INFO,
                message="Leakage check completed successfully",
                data={
                    "stage": stage,
                    "features_checked": X.shape[1] if len(X.shape) > 1 else 1,
                    "status": "clean"
                }
            )
            return True
            
        except Exception as e:
            self._send_event_to_nucleus(
                event_type="immuno_guard_error",
                severity=EventSeverity.CRITICAL,
                message=f"Error during leakage detection: {str(e)}",
                data={
                    "stage": stage,
                    "error": str(e),
                    "error_type": type(e).__name__
                },
                intervention_suggestion=InterventionType.BLOCK_TRAINING
            )
            return False

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Converte dati a numpy array"""
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
    
    def _detect_target_leakage(self, X: np.ndarray, y: np.ndarray, feature_names: List[str], stage: str) -> List[LeakageAlert]:
        """Qui devo Rilevare per forza le correlazioni troppo forti con il target"""
        alerts = []
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        for i in range(n_features):
            feature_col = X[:, i]
            
            # Skip se feature costante
            if np.std(feature_col) == 0:
                continue
                
            try:
                # Correlazione di Pearson
                corr_pearson, p_val = pearsonr(feature_col, y)
                abs_corr = abs(corr_pearson)
                
                # Memorizza correlazione per tracking
                self.target_correlations[feature_names[i]] = abs_corr
                
                # Soglie per leakage
                if abs_corr > 0.95:  # Correlazione quasi perfetta mhhh = leakage certo
                    alerts.append(LeakageAlert(
                        leakage_type="perfect_correlation",
                        severity="critical",
                        correlation_score=abs_corr,
                        features_involved=[feature_names[i]],
                        description=f"Perfect correlation with target: {abs_corr:.4f}",
                        recommendation=f"Remove feature {feature_names[i]} - likely data leakage"
                    ))
                    self.blocked_features.add(feature_names[i])
                    
                elif abs_corr > 0.8:  # Correlazione molto alta = sospetto
                    alerts.append(LeakageAlert(
                        leakage_type="high_correlation", 
                        severity="critical",
                        correlation_score=abs_corr,
                        features_involved=[feature_names[i]],
                        description=f"Suspiciously high correlation with target: {abs_corr:.4f}",
                        recommendation=f"Investigate feature {feature_names[i]} for potential leakage"
                    ))
                    
                elif abs_corr > 0.6:  # Correlazione alta = warning
                    alerts.append(LeakageAlert(
                        leakage_type="moderate_correlation",
                        severity="warning", 
                        correlation_score=abs_corr,
                        features_involved=[feature_names[i]],
                        description=f"High correlation with target: {abs_corr:.4f}",
                        recommendation=f"Review feature {feature_names[i]} - ensure no information leakage"
                    ))
                    
            except Exception as e:
                # Skip features danno sempre problemi nel calcolo correlazione
                continue
                
        return alerts
    
    def _detect_multicollinearity(self, X: np.ndarray, feature_names: List[str], stage: str) -> List[LeakageAlert]:
        """Rileva multicollinearita tra features"""
        alerts = []
        
        if len(X.shape) == 1 or X.shape[1] < 2:
            return alerts
            
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
        # Controlla correlazioni tra coppie di features
        for i in range(n_features):
            for j in range(i + 1, n_features):
                try:
                    feature_i = X[:, i]
                    feature_j = X[:, j]
                    
                    # Skip se features costanti
                    if np.std(feature_i) == 0 or np.std(feature_j) == 0:
                        continue
                        
                    corr, _ = pearsonr(feature_i, feature_j)
                    abs_corr = abs(corr)
                    
                    if abs_corr > 0.95:  # Quasi identiche
                        alerts.append(LeakageAlert(
                            leakage_type="perfect_multicollinearity",
                            severity="warning",
                            correlation_score=abs_corr,
                            features_involved=[feature_names[i], feature_names[j]],
                            description=f"Perfect correlation between features: {abs_corr:.4f}",
                            recommendation=f"Consider removing one of: {feature_names[i]}, {feature_names[j]}"
                        ))
                        
                except Exception:
                    continue
                    
        return alerts
    
    def _detect_temporal_leakage(self, X: np.ndarray, feature_names: List[str], stage: str) -> List[LeakageAlert]:
        """Rileva potenziale temporal leakage (placeholder)"""
        alerts = []
        
        # Placeholder per temporal leakage detection
        # In implementazione reale, controllerebbe:
        # - Features che hanno timestamp futuri
        # - Features che sono derivate da informazioni future
        # - Ordinamento temporale dei dati
        return alerts
    
    def _detect_constant_features(self, X: np.ndarray, feature_names: List[str], stage: str) -> List[LeakageAlert]:
        """Rileva features costanti o quasi-costanti"""
        alerts = []
        
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
            
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
            
        for i in range(n_features):
            feature_col = X[:, i]
            unique_vals = np.unique(feature_col)
            
            if len(unique_vals) == 1:
                alerts.append(LeakageAlert(
                    leakage_type="constant_feature",
                    severity="warning",
                    correlation_score=0.0,
                    features_involved=[feature_names[i]],
                    description=f"Constant feature with single value: {unique_vals[0]}",
                    recommendation=f"Remove constant feature {feature_names[i]}"
                ))
                
            elif len(unique_vals) / len(feature_col) < 0.01:  # Meno dell'1% di valori unici
                alerts.append(LeakageAlert(
                    leakage_type="quasi_constant_feature",
                    severity="warning", 
                    correlation_score=0.0,
                    features_involved=[feature_names[i]],
                    description=f"Quasi-constant feature with {len(unique_vals)} unique values",
                    recommendation=f"Consider removing quasi-constant feature {feature_names[i]}"
                ))
                
        return alerts
    
    def _alert_to_dict(self, alert: LeakageAlert) -> Dict[str, Any]:
        """Converte LeakageAlert a dictionary"""
        return {
            "leakage_type": alert.leakage_type,
            "severity": alert.severity,
            "correlation_score": alert.correlation_score,
            "features_involved": alert.features_involved,
            "description": alert.description,
            "recommendation": alert.recommendation
        }
    
    def get_correlation_report(self) -> Dict[str, Any]:
        """Genera report delle correlazioni rilevate"""
        return {
            "target_correlations": dict(self.target_correlations),
            "blocked_features": list(self.blocked_features),
            "high_risk_features": [
                feat for feat, corr in self.target_correlations.items() 
                if corr > 0.8
            ],
            "moderate_risk_features": [
                feat for feat, corr in self.target_correlations.items()
                if 0.6 < corr <= 0.8
            ]
        }