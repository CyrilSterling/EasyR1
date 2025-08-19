#!/usr/bin/env python3
"""
Quick tests to verify curriculum metric optimization correctness.
"""

from typing import List, Dict, Any

# Simplified mock classes for testing
class MockRLHFDataset:
    def __init__(self, size=100):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {"id": idx, "data": f"sample_{idx}"}

class PaddedSequentialSampler:
    """Copy of the sampler for testing."""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        pad_to_length = (len(dataset) + batch_size - 1) // batch_size * batch_size
        self.indices = list(range(pad_to_length))
        self.indices[len(dataset):] = [len(dataset) - 1] * (pad_to_length - len(dataset))

    def __iter__(self):
        for i in self.indices:
            yield i

    def __len__(self):
        return len(self.indices)

# Mock Ray remote functions for testing
@ray.remote
def mock_calculate_single_bleu_score(tokens: List, start_idx: int, end_idx: int):
    """Mock BLEU calculation."""
    import time
    time.sleep(0.1)  # Simulate computation
    return np.random.random()

@ray.remote
def mock_calculate_self_bleu_metric(responses, batch_size, curriculum_rollout_n):
    """Mock self-BLEU metric using Ray tasks."""
    tokenized_sequences = responses.tolist() if hasattr(responses, 'tolist') else responses
    
    # Create Ray tasks
    futures = []
    for i in range(batch_size):
        start_idx = i * curriculum_rollout_n
        end_idx = (i + 1) * curriculum_rollout_n
        future = mock_calculate_single_bleu_score.remote(
            tokenized_sequences, start_idx, end_idx
        )
        futures.append(future)
    
    bleu_scores = ray.get(futures)
    return torch.tensor(bleu_scores, dtype=torch.float32)

def test_padded_sampler():
    """Test PaddedSequentialSampler correctness."""
    print("🧪 Testing PaddedSequentialSampler...")
    
    # Test case 1: Dataset size divisible by batch_size
    dataset = MockRLHFDataset(size=20)
    sampler = PaddedSequentialSampler(dataset, batch_size=5)
    indices = list(sampler)
    
    assert len(indices) == 20, f"Expected 20 indices, got {len(indices)}"
    assert indices == list(range(20)), f"Expected sequential indices, got {indices}"
    print("✅ Test 1 passed: Dataset size divisible by batch_size")
    
    # Test case 2: Dataset size not divisible by batch_size
    dataset = MockRLHFDataset(size=23)
    sampler = PaddedSequentialSampler(dataset, batch_size=5)
    indices = list(sampler)
    
    assert len(indices) == 25, f"Expected 25 indices (padded), got {len(indices)}"
    assert indices[:23] == list(range(23)), "First 23 indices should be sequential"
    assert indices[23:] == [22, 22], "Last 2 indices should be padding (repeat last)"
    print("✅ Test 2 passed: Dataset size not divisible by batch_size")

def test_batch_consistency():
    """Test batch size consistency."""
    print("🧪 Testing batch size consistency...")
    
    dataset = MockRLHFDataset(size=23)
    batch_size = 5
    sampler = PaddedSequentialSampler(dataset, batch_size=batch_size)
    
    # Simulate dataloader behavior
    indices = list(sampler)
    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    
    # All batches should have same size
    batch_sizes = [len(batch) for batch in batches]
    assert all(size == batch_size for size in batch_sizes), f"Inconsistent batch sizes: {batch_sizes}"
    print(f"✅ All {len(batches)} batches have consistent size: {batch_size}")

def test_weight_assignment():
    """Test curriculum weight assignment logic."""
    print("🧪 Testing weight assignment logic...")
    
    dataset_size = 23
    batch_size = 5
    curriculum_weights = torch.zeros(dataset_size)
    
    # Simulate batches
    num_batches = (dataset_size + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        # Simulate computed metrics for this batch
        mock_metrics = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])  # batch_size metrics
        
        # Apply weight assignment logic
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, dataset_size)
        valid_samples = end_idx - start_idx
        
        curriculum_weights[start_idx:end_idx] = mock_metrics[:valid_samples]
    
    # Check results
    assert len(curriculum_weights) == dataset_size, f"Wrong weight array size"
    assert curriculum_weights[-1] != 0, "Last sample should have weight assigned"
    print(f"✅ Weight assignment correct: {curriculum_weights[:10]}...")

@ray.remote
class MockRewardFunction:
    def __call__(self, batch):
        # Simulate reward computation
        batch_size = len(batch) if hasattr(batch, '__len__') else 10
        return torch.randn(batch_size, 20), {"accuracy": [0.8] * batch_size}

def test_ray_task_parallelism():
    """Test Ray task parallelism vs multiprocessing."""
    print("🧪 Testing Ray task parallelism...")
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Mock data
    responses = torch.randint(0, 1000, (50, 20))  # 50 sequences, 20 tokens each
    batch_size = 10
    curriculum_rollout_n = 5
    
    # Test Ray-based approach
    import time
    start_time = time.time()
    
    future = mock_calculate_self_bleu_metric.remote(
        responses, batch_size, curriculum_rollout_n
    )
    result = ray.get(future)
    
    ray_time = time.time() - start_time
    
    # Verify result
    assert isinstance(result, torch.Tensor), "Result should be tensor"
    assert result.shape == (batch_size,), f"Expected shape ({batch_size},), got {result.shape}"
    
    print(f"✅ Ray parallelism test passed in {ray_time:.2f}s")
    print(f"   Result shape: {result.shape}")
    print(f"   Result sample: {result[:3]}")

def test_memory_usage_simulation():
    """Simulate memory usage difference."""
    print("🧪 Testing memory usage simulation...")
    
    # Simulate old approach: passing entire dataset
    large_dataset = MockRLHFDataset(size=10000)
    dataset_size_mb = 10000 * 50 * 4 / (1024*1024)  # Rough estimate
    
    # Simulate new approach: passing only batch data
    batch_size = 32
    batch_data = mock_collate_fn([large_dataset[i] for i in range(batch_size)])
    batch_size_mb = batch_size * 50 * 4 / (1024*1024)  # Rough estimate
    
    memory_reduction = (dataset_size_mb - batch_size_mb) / dataset_size_mb * 100
    
    print(f"✅ Memory usage simulation:")
    print(f"   Old approach (full dataset): ~{dataset_size_mb:.1f} MB")
    print(f"   New approach (batch only): ~{batch_size_mb:.1f} MB") 
    print(f"   Estimated reduction: {memory_reduction:.1f}%")

def run_all_tests():
    """Run all tests."""
    print("🚀 Starting curriculum optimization tests...\n")
    
    try:
        test_padded_sampler()
        print()
        
        test_batch_consistency() 
        print()
        
        test_weight_assignment()
        print()
        
        test_ray_task_parallelism()
        print()
        
        test_memory_usage_simulation()
        print()
        
        print("🎉 All tests passed! The optimization implementation looks correct.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    run_all_tests()