import sys
sys.path.append("..")
import pytest
from datetime import datetime, timezone
import numpy as np
from image_generation_subnet.utils.weight_calculation import WeightCalculationService

class TestWeightCalculationService:
    @pytest.fixture
    def service(self):
        return WeightCalculationService()

    @pytest.fixture
    def sample_weights(self):
        return {
            'alpha': np.array([0.7, 0.3]),
            'specific_model': np.array([0.6, 0.4]),
            'recycle': np.array([0.8, 0.2])
        }

    def test_before_first_transition(self, service, sample_weights):
        test_time = datetime(2025, 2, 18, tzinfo=timezone.utc)
        
        result = service.calculate_transition_weights(
            alpha_raw_weights=sample_weights['alpha'],
            specific_model_raw_weights=sample_weights['specific_model'],
            recycle_raw_weights=sample_weights['recycle'],
            current_time=test_time
        )
        
        # Before first transition: 100% specific_model, 0% recycle, 100% miner, 0% stake
        expected = (
            1.0 * sample_weights['specific_model'] + 
            0.0 * sample_weights['recycle']
        ) * 1.0 + 0.0 * sample_weights['alpha']
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_during_first_transition(self, service, sample_weights):
        test_time = datetime(2025, 2, 19, 12, tzinfo=timezone.utc)
        
        result = service.calculate_transition_weights(
            alpha_raw_weights=sample_weights['alpha'],
            specific_model_raw_weights=sample_weights['specific_model'],
            recycle_raw_weights=sample_weights['recycle'],
            current_time=test_time
        )
        
        # During first transition: 76% specific_model, 24% recycle, 85% miner, 15% stake
        expected = (
            0.76 * sample_weights['specific_model'] + 
            0.24 * sample_weights['recycle']
        ) * 0.85 + 0.15 * sample_weights['alpha']
        
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_after_second_transition(self, service, sample_weights):
        test_time = datetime(2025, 2, 20, 12, tzinfo=timezone.utc)
        
        result = service.calculate_transition_weights(
            alpha_raw_weights=sample_weights['alpha'],
            specific_model_raw_weights=sample_weights['specific_model'],
            recycle_raw_weights=sample_weights['recycle'],
            current_time=test_time
        )
        
        # After second transition: 76% specific_model, 24% recycle, 70% miner, 30% stake
        expected = (
            0.52 * sample_weights['specific_model'] + 
            0.48 * sample_weights['recycle']
        ) * 0.7 + 0.3 * sample_weights['alpha']
        
        np.testing.assert_array_almost_equal(result, expected)
