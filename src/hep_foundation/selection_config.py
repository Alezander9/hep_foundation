from typing import Dict, Union, Optional, Tuple
import numpy as np

class SelectionConfig:
    """Configuration for track and event selections"""
    def __init__(self,
                 # Required parameters
                 max_tracks_per_event: int,
                 min_tracks_per_event: int,
                 # Optional selection criteria
                 track_selections: Optional[Dict] = None,
                 event_selections: Optional[Dict] = None):
        # Required parameters
        self.max_tracks_per_event = max_tracks_per_event
        self.min_tracks_per_event = min_tracks_per_event
        
        # Optional selections
        self.track_selections = track_selections or {}
        self.event_selections = event_selections or {}
        
    def apply_track_selections(self, track_features: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply selections to individual tracks"""
        # Start with all tracks
        mask = np.ones(len(next(iter(track_features.values()))), dtype=bool)
        
        # Map feature names to indices
        feature_map = {
            'pt': track_features['pt'],
            'eta': track_features['eta'],
            'phi': track_features['phi'],
            'd0': track_features['d0'],
            'z0': track_features['z0'],
            'chi2_per_ndof': track_features['chi2_per_ndof']
        }
        
        # Apply each selection
        for feature, criteria in self.track_selections.items():
            if feature not in feature_map:
                continue
                
            feature_values = feature_map[feature]
            if isinstance(criteria, tuple):
                min_val, max_val = criteria
                if min_val is not None:
                    mask &= (feature_values >= min_val)
                if max_val is not None:
                    mask &= (feature_values <= max_val)
            else:
                mask &= (feature_values >= criteria)
                
        return mask
    
    def apply_event_selections(self, event_features: Dict[str, float]) -> bool:
        """Apply event-level selections"""
        # print(f"\nApplying event selections:")
        # print(f"Configured selections: {self.event_selections}")
        # print(f"Event features: {event_features}")
        
        if not self.event_selections:
            # print("No event selections configured - passing event")
            return True
            
        for feature, (min_val, max_val) in self.event_selections.items():
            if feature not in event_features:
                # print(f"Warning: Feature {feature} not found in event")
                continue
            value = event_features[feature]
            if min_val is not None and value < min_val:
                # print(f"Failed {feature} min cut: {value} < {min_val}")
                return False
            if max_val is not None and value > max_val:
                # print(f"Failed {feature} max cut: {value} > {max_val}")
                return False
        # print("Passed all event selections")
        return True