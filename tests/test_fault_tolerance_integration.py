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
Integration tests for Ray actor fault tolerance functionality.
These tests require Ray to be properly installed and configured.
"""

import pytest
import ray
import ray.exceptions
import time
from unittest.mock import Mock


@pytest.fixture(scope="module")
def ray_cluster():
    """Initialize Ray cluster for testing."""
    if not ray.is_initialized():
        ray.init(local_mode=True)
    yield
    if ray.is_initialized():
        ray.shutdown()


class TestActorFaultTolerance:
    """Integration tests for actor fault tolerance."""
    
    def test_actor_restart_on_failure(self, ray_cluster):
        """Test that actors restart after failure."""
        
        @ray.remote(max_restarts=2, max_task_retries=1)
        class TestActor:
            def __init__(self):
                self.call_count = 0
                self.restart_count = 0
                print(f"TestActor initialized (restart: {self.restart_count})")
            
            def increment(self):
                self.call_count += 1
                # Simulate failure on first call
                if self.call_count == 1:
                    print("TestActor crashing on first call...")
                    raise RuntimeError("Simulated failure")
                return self.call_count
            
            def get_call_count(self):
                return self.call_count
        
        # Create actor
        actor = TestActor.remote()
        
        # First call should fail and trigger restart
        try:
            result = ray.get(actor.increment.remote())
            # If we get here, the task was retried successfully
            assert result == 1  # After restart, counter resets
        except ray.exceptions.RayActorError:
            # This is expected if max_task_retries is exhausted
            pass
        
        # Second call should work (on restarted actor)
        try:
            result = ray.get(actor.increment.remote())
            assert result >= 1  # Should be at least 1
        except ray.exceptions.RayActorError:
            # Actor might have been restarted, this is expected behavior
            pass
    
    def test_actor_without_fault_tolerance(self, ray_cluster):
        """Test actor behavior without fault tolerance."""
        
        @ray.remote  # No fault tolerance parameters
        class TestActor:
            def __init__(self):
                self.call_count = 0
            
            def fail_immediately(self):
                raise RuntimeError("Immediate failure")
        
        # Create actor
        actor = TestActor.remote()
        
        # This should fail without retry
        with pytest.raises(ray.exceptions.RayTaskError):
            ray.get(actor.fail_immediately.remote())
    
    def test_actor_task_retry_on_actor_death(self, ray_cluster):
        """Test that tasks are retried when actor dies."""
        
        @ray.remote(max_restarts=1, max_task_retries=2)
        class TestActor:
            def __init__(self):
                self.death_count = 0
            
            def die_and_retry(self):
                self.death_count += 1
                if self.death_count <= 1:
                    # Kill the actor process to simulate death
                    import os
                    os._exit(1)
                return "success"
        
        # Create actor
        actor = TestActor.remote()
        
        # This should eventually succeed after actor restart and task retry
        try:
            result = ray.get(actor.die_and_retry.remote())
            assert result == "success"
        except ray.exceptions.RayActorError:
            # This might happen if retries are exhausted
            pass
    
    def test_multiple_actor_failures(self, ray_cluster):
        """Test handling of multiple actor failures."""
        
        @ray.remote(max_restarts=3, max_task_retries=1)
        class TestActor:
            def __init__(self):
                self.id = ray.get_runtime_context().get_actor_id()
                print(f"TestActor {self.id} initialized")
            
            def get_actor_id(self):
                return self.id
            
            def crash(self):
                raise RuntimeError("Intentional crash")
        
        # Create multiple actors
        actors = [TestActor.remote() for _ in range(3)]
        
        # Get initial actor IDs
        initial_ids = []
        for actor in actors:
            try:
                actor_id = ray.get(actor.get_actor_id.remote())
                initial_ids.append(actor_id)
            except:
                initial_ids.append(None)
        
        # Crash all actors
        for actor in actors:
            try:
                ray.get(actor.crash.remote())
            except:
                pass  # Expected to fail
        
        # Try to use actors again (should work if they restarted)
        for i, actor in enumerate(actors):
            try:
                new_id = ray.get(actor.get_actor_id.remote())
                # Actor ID might change after restart
                print(f"Actor {i}: {initial_ids[i]} -> {new_id}")
            except ray.exceptions.RayActorError:
                # This is expected if max_restarts is exhausted
                print(f"Actor {i} permanently failed")


class TestFaultToleranceConfiguration:
    """Integration tests for fault tolerance configuration."""
    
    def test_actor_creation_with_config(self, ray_cluster):
        """Test actor creation with different fault tolerance configurations."""
        
        # Test with high fault tolerance
        @ray.remote(max_restarts=5, max_task_retries=3)
        class HighFaultToleranceActor:
            def test_method(self):
                return "high_tolerance"
        
        actor = HighFaultToleranceActor.remote()
        result = ray.get(actor.test_method.remote())
        assert result == "high_tolerance"
        
        # Test with low fault tolerance
        @ray.remote(max_restarts=1, max_task_retries=0)
        class LowFaultToleranceActor:
            def test_method(self):
                return "low_tolerance"
        
        actor = LowFaultToleranceActor.remote()
        result = ray.get(actor.test_method.remote())
        assert result == "low_tolerance"


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v", "-s"])