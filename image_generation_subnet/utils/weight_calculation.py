from datetime import datetime, timezone
import numpy as np

class WeightTransitionConfig:
    TRANSITIONS = [
        {
            "deadline": datetime(2025, 2, 19, 0, 0, 0, 0, tzinfo=timezone.utc),
            "miner_weight": 1.0,
            "stake_weight": 0.0
        },
        {
            "deadline": datetime(2025, 2, 24, 0, 0, 0, 0, tzinfo=timezone.utc),
            "miner_weight": 0.7,
            "stake_weight": 0.3
        },
        {
            "deadline": datetime(2025, 2, 26, 0, 0, 0, 0, tzinfo=timezone.utc),
            "miner_weight": 0.4,
            "stake_weight": 0.6
        },
    ]
    
    # Default weights after all transitions
    DEFAULT_MINER_WEIGHT = 0.1
    DEFAULT_STAKE_WEIGHT = 0.9 

class WeightCalculationService:
    @staticmethod
    def calculate_transition_weights(
        miner_raw_weights: np.ndarray,
        alpha_raw_weights: np.ndarray,
        current_time: datetime = None
    ) -> np.ndarray:
        """
        Calculate weights based on time-based transitions
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        for transition in WeightTransitionConfig.TRANSITIONS:
            if current_time < transition["deadline"]:
                return (transition["miner_weight"] * miner_raw_weights + 
                       transition["stake_weight"] * alpha_raw_weights)
                
        # After all transitions, use default weights
        return (WeightTransitionConfig.DEFAULT_MINER_WEIGHT * miner_raw_weights + 
                WeightTransitionConfig.DEFAULT_STAKE_WEIGHT * alpha_raw_weights)