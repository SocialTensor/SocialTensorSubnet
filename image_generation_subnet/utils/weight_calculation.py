from datetime import datetime, timezone
import numpy as np

class WeightTransitionConfig:
    TRANSITIONS = [
        {
            "deadline": datetime(2025, 2, 19, 0, 0, 0, 0, tzinfo=timezone.utc),
            "miner_weight": 1.0,
            "stake_weight": 0.0,
            "specific_model_weight": 1.0,
            "recycle_weight": 0.0,
        },
        {
            "deadline": datetime(2025, 2, 20, 0, 0, 0, 0, tzinfo=timezone.utc),
            "miner_weight": 0.85,
            "stake_weight": 0.15,
            "specific_model_weight": 0.76,
            "recycle_weight": 0.24,
        },
    ]
    
    # Default weights after all transitions
    DEFAULT_MINER_WEIGHT = 0.7
    DEFAULT_STAKE_WEIGHT = 0.3 
    DEFAULT_SPECIFIC_MODEL_WEIGHT = 0.52
    DEFAULT_RECYCLE_WEIGHT = 0.48

class WeightCalculationService:
    @staticmethod
    def calculate_transition_weights(
        alpha_raw_weights: np.ndarray,
        specific_model_raw_weights: np.ndarray,
        recycle_raw_weights: np.ndarray,
        current_time: datetime = None
    ) -> np.ndarray:
        """
        Calculate weights based on time-based transitions
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)
            
        for transition in WeightTransitionConfig.TRANSITIONS:
            if current_time < transition["deadline"]:
                miner_component = (
                    transition["specific_model_weight"] * specific_model_raw_weights +
                    transition["recycle_weight"] * recycle_raw_weights
                )
                # Normalize miner component
                miner_component_sum = np.sum(miner_component)
                if miner_component_sum != 0:
                    miner_component = miner_component / miner_component_sum
                miner_component = miner_component * transition["miner_weight"]
                
                stake_component = alpha_raw_weights
                stake_component_sum = np.sum(stake_component)
                if stake_component_sum != 0:
                    stake_component = stake_component / stake_component_sum
                stake_component = stake_component * transition["stake_weight"]
                return miner_component + stake_component
                
        # After all transitions, use default weights
        miner_component = (
            WeightTransitionConfig.DEFAULT_SPECIFIC_MODEL_WEIGHT 
            * specific_model_raw_weights +
            WeightTransitionConfig.DEFAULT_RECYCLE_WEIGHT 
            * recycle_raw_weights
        )
        miner_component_sum = np.sum(miner_component)
        if miner_component_sum != 0:
            miner_component = miner_component / miner_component_sum
        miner_component = miner_component * WeightTransitionConfig.DEFAULT_MINER_WEIGHT
        
        stake_component = alpha_raw_weights
        stake_component_sum = np.sum(stake_component)
        if stake_component_sum != 0:
            stake_component = stake_component / stake_component_sum
        stake_component = stake_component * WeightTransitionConfig.DEFAULT_STAKE_WEIGHT
        
        return miner_component + stake_component