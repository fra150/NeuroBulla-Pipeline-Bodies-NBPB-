# nbpb/utils/metrics.py
import numpy as np
from typing import Dict, Any, List
import time
NBPBEvent = Any  
EventSeverity = Any
DataSnapshot = Any
RECENT_EVENTS_TIMEFRAME_SECONDS = 300  # 5 minutes
HEALTH_SCORE_CRITICAL_PENALTY = 0.7
HEALTH_SCORE_WARNING_PENALTY = 0.9
STABILITY_SNAPSHOT_WINDOW = 10

def compute_health_score(events: List[NBPBEvent]) -> float:
    """
    Computa score di salute basato sugli eventi recenti
    Args:
        events: Lista di eventi NBPB 
    Returns:
        float: Score da 0.0 (critico) a 1.0 (perfetto)
    """
    if not events:
        return 1.0
        
    current_time = time.time()
    recent_events = [
        e for e in events 
        if (current_time - getattr(e, 'timestamp', float('-inf'))) < RECENT_EVENTS_TIMEFRAME_SECONDS
    ]
    if not recent_events:
        return 1.0
    score = 1.0
    for event in recent_events:
        severity = getattr(event, 'severity', None)
        if severity == "critical":
            score *= HEALTH_SCORE_CRITICAL_PENALTY
        elif severity == "warning":
            score *= HEALTH_SCORE_WARNING_PENALTY    
    return max(0.0, score)

def compute_data_stability_score(snapshots: List[DataSnapshot]) -> float:
    """
    Computa score di stabilita dei dati basato sui snapshot
    Returns:
        float: Score da 0.0 (instabile) a 1.0 (stabile)
    """
    if len(snapshots) < 2:
        return 1.0
    recent_snapshots = snapshots[-STABILITY_SNAPSHOT_WINDOW:]
    means = [getattr(s, 'mean', 0.0) for s in recent_snapshots]
    stds = [getattr(s, 'std', 0.0) for s in recent_snapshots]
    mean_cv = np.std(means) / np.mean(means) if np.mean(means) != 0 else 0
    std_cv = np.std(stds) / np.mean(stds) if np.mean(stds) != 0 else 0
    stability = 1.0 - min(1.0, (mean_cv + std_cv) / 2)
    return max(0.0, stability)