import sys
sys.path.append("..")
import pytest
from datetime import datetime, timezone
import numpy as np
from image_generation_subnet.utils.weight_calculation import WeightCalculationService

class TestWeightCalculationService:
    def test_weight_calculation_before_first_transition(self):
        service = WeightCalculationService()
        test_time = datetime(2025, 2, 16, tzinfo=timezone.utc)
        miner_weights = np.array([0.5, 0.5])
        stake_weights = np.array([0.7, 0.3])
        
        result = service.calculate_transition_weights(
            miner_weights,
            stake_weights,
            test_time
        )
        
        expected = 1.0 * miner_weights + 0.0 * stake_weights
        np.testing.assert_array_almost_equal(result, expected)

    def test_weight_calculation_during_first_transition(self):
        service = WeightCalculationService()
        test_time = datetime(2025, 2, 20, tzinfo=timezone.utc)
        miner_weights = np.array([0.5, 0.5])
        stake_weights = np.array([0.7, 0.3])
        
        result = service.calculate_transition_weights(
            miner_weights,
            stake_weights,
            test_time
        )
        
        expected = 0.7 * miner_weights + 0.3 * stake_weights
        np.testing.assert_array_almost_equal(result, expected)

    def test_weight_calculation_before_second_transition(self):
        service = WeightCalculationService()
        test_time = datetime(2025, 2, 25, tzinfo=timezone.utc)
        miner_weights = np.array([0.5, 0.5])
        stake_weights = np.array([0.7, 0.3])
        
        result = service.calculate_transition_weights(
            miner_weights,
            stake_weights,
            test_time
        )
        
        expected = 0.4 * miner_weights + 0.6 * stake_weights
        np.testing.assert_array_almost_equal(result, expected)

    def test_weight_calculation_before_third_transition(self):
        service = WeightCalculationService()
        test_time = datetime(2025, 2, 27, tzinfo=timezone.utc)
        miner_weights = np.array([0.5, 0.5])
        stake_weights = np.array([0.7, 0.3])
        
        result = service.calculate_transition_weights(
            miner_weights,
            stake_weights,
            test_time
        )
        
        expected = 0.1 * miner_weights + 0.9 * stake_weights
        np.testing.assert_array_almost_equal(result, expected) 