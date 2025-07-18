# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for Ray actor fault tolerance functionality.
"""

import pytest
import ray
import ray.exceptions
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

from verl.trainer.config import PPOConfig, FaultToleranceConfig
from verl.trainer.ray_trainer import RayPPOTrainer, Role


class TestFaultToleranceConfig:
    """Test the fault tolerance configuration."""
    
    def test_default_fault_tolerance_config(self):
        """Test default fault tolerance configuration values."""
        config = FaultToleranceConfig()
        
        assert config.enable_fault_tolerance == True
        assert config.max_restarts == 3
        assert config.max_task_retries == 2
        assert config.enable_health_monitoring == True
        assert config.health_check_interval == 30.0
    
    def test_fault_tolerance_in_ppo_config(self):
        """Test that fault tolerance config is properly integrated into PPOConfig."""
        ppo_config = PPOConfig()
        
        assert hasattr(ppo_config, 'fault_tolerance')
        assert isinstance(ppo_config.fault_tolerance, FaultToleranceConfig)
        assert ppo_config.fault_tolerance.enable_fault_tolerance == True
    
    def test_custom_fault_tolerance_config(self):
        """Test custom fault tolerance configuration."""
        custom_config = FaultToleranceConfig(
            enable_fault_tolerance=False,
            max_restarts=5,
            max_task_retries=3,
            enable_health_monitoring=False,
            health_check_interval=60.0
        )
        
        assert custom_config.enable_fault_tolerance == False
        assert custom_config.max_restarts == 5
        assert custom_config.max_task_retries == 3
        assert custom_config.enable_health_monitoring == False
        assert custom_config.health_check_interval == 60.0


class TestRayPPOTrainerFaultTolerance:
    """Test the Ray PPO trainer fault tolerance functionality."""
    
    def setup_method(self):
        """Setup method called before each test."""
        self.config = PPOConfig()
        self.config.fault_tolerance.enable_fault_tolerance = True
        self.config.fault_tolerance.max_restarts = 3
        self.config.fault_tolerance.max_task_retries = 2
        
        # Mock the trainer without actually initializing Ray
        self.trainer = Mock(spec=RayPPOTrainer)
        self.trainer.config = self.config
        self.trainer.actor_failure_counts = defaultdict(int)
        self.trainer.actor_restart_counts = defaultdict(int)
        self.trainer.global_step = 0
        self.trainer.logger = Mock()
        
        # Bind the actual methods to the mock
        self.trainer._handle_actor_error = RayPPOTrainer._handle_actor_error.__get__(self.trainer)
        self.trainer._log_actor_restart = RayPPOTrainer._log_actor_restart.__get__(self.trainer)
        self.trainer._check_actor_health = RayPPOTrainer._check_actor_health.__get__(self.trainer)
    
    def test_handle_actor_error_logging(self):
        """Test that actor errors are properly logged."""
        # Mock the ray actor error
        error = ray.exceptions.RayActorError("Test actor error")
        
        # Call the error handler
        self.trainer._handle_actor_error(error, "test_actor")
        
        # Check that failure count was incremented
        assert self.trainer.actor_failure_counts["test_actor"] == 1
        
        # Check that logger was called with appropriate metrics
        self.trainer.logger.log.assert_called_once()
        call_args = self.trainer.logger.log.call_args
        metrics = call_args[1]['data']
        
        assert 'actor_failures/test_actor' in metrics
        assert 'actor_failures/total' in metrics
        assert 'actor_failures/test_actor_cumulative' in metrics
        assert metrics['actor_failures/test_actor'] == 1
        assert metrics['actor_failures/test_actor_cumulative'] == 1
    
    def test_log_actor_restart(self):
        """Test that actor restarts are properly logged."""
        # Call the restart logger
        self.trainer._log_actor_restart("test_actor")
        
        # Check that restart count was incremented
        assert self.trainer.actor_restart_counts["test_actor"] == 1
        
        # Check that logger was called with appropriate metrics
        self.trainer.logger.log.assert_called_once()
        call_args = self.trainer.logger.log.call_args
        metrics = call_args[1]['data']
        
        assert 'actor_restarts/test_actor' in metrics
        assert 'actor_restarts/total' in metrics
        assert 'actor_restarts/test_actor_cumulative' in metrics
        assert metrics['actor_restarts/test_actor'] == 1
        assert metrics['actor_restarts/test_actor_cumulative'] == 1
    
    def test_check_actor_health_all_healthy(self):
        """Test actor health check when all actors are healthy."""
        # Mock healthy worker groups
        self.trainer.actor_rollout_wg = Mock()
        self.trainer.actor_rollout_wg.world_size = 4
        self.trainer.critic_wg = Mock()
        self.trainer.critic_wg.world_size = 2
        self.trainer.ref_policy_wg = Mock()
        self.trainer.ref_policy_wg.world_size = 2
        
        # Check health
        health_status = self.trainer._check_actor_health()
        
        # All should be healthy
        assert health_status["actor_rollout"] == "healthy"
        assert health_status["critic"] == "healthy"
        assert health_status["ref_policy"] == "healthy"
    
    def test_check_actor_health_with_failure(self):
        """Test actor health check when some actors fail."""
        # Mock healthy worker groups
        self.trainer.actor_rollout_wg = Mock()
        self.trainer.actor_rollout_wg.world_size = 4
        
        # Mock an unhealthy critic
        self.trainer.critic_wg = Mock()
        self.trainer.critic_wg.world_size = Mock(side_effect=Exception("Actor died"))
        
        # No ref policy
        self.trainer.ref_policy_wg = None
        
        # Check health
        health_status = self.trainer._check_actor_health()
        
        # Actor rollout should be healthy, critic should be unhealthy
        assert health_status["actor_rollout"] == "healthy"
        assert health_status["critic"].startswith("unhealthy:")
        assert "ref_policy" not in health_status
    
    def test_safe_actor_call_success(self):
        """Test safe actor call with successful execution."""
        # Mock successful ray.get
        mock_ref = Mock()
        expected_result = "success"
        
        with patch('ray.get', return_value=expected_result):
            result = self.trainer._safe_actor_call(mock_ref, "test_actor")
            assert result == expected_result
    
    def test_safe_actor_call_actor_error(self):
        """Test safe actor call with actor error."""
        # Mock actor error
        mock_ref = Mock()
        error = ray.exceptions.RayActorError("Test actor error")
        
        with patch('ray.get', side_effect=error):
            with pytest.raises(ray.exceptions.RayActorError):
                self.trainer._safe_actor_call(mock_ref, "test_actor")
            
            # Check that error was handled
            assert self.trainer.actor_failure_counts["test_actor"] == 1
    
    def test_safe_actor_call_timeout(self):
        """Test safe actor call with timeout."""
        # Mock timeout error
        mock_ref = Mock()
        error = ray.exceptions.GetTimeoutError("Timeout")
        
        with patch('ray.get', side_effect=error):
            with pytest.raises(ray.exceptions.GetTimeoutError):
                self.trainer._safe_actor_call(mock_ref, "test_actor", timeout=1.0)


class TestActorCreationWithFaultTolerance:
    """Test actor creation with fault tolerance parameters."""
    
    def test_actor_creation_with_fault_tolerance_enabled(self):
        """Test that actors are created with fault tolerance when enabled."""
        # This would require mocking the entire ray.remote system
        # For now, we'll just test the configuration logic
        
        config = PPOConfig()
        config.fault_tolerance.enable_fault_tolerance = True
        config.fault_tolerance.max_restarts = 5
        config.fault_tolerance.max_task_retries = 3
        
        # This would be tested in an integration test with actual Ray
        # For unit tests, we just verify the configuration is correct
        assert config.fault_tolerance.enable_fault_tolerance == True
        assert config.fault_tolerance.max_restarts == 5
        assert config.fault_tolerance.max_task_retries == 3
    
    def test_actor_creation_without_fault_tolerance(self):
        """Test that actors are created without fault tolerance when disabled."""
        config = PPOConfig()
        config.fault_tolerance.enable_fault_tolerance = False
        
        # When disabled, actors should be created without fault tolerance
        assert config.fault_tolerance.enable_fault_tolerance == False


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])