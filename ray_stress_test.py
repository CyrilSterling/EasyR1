#!/usr/bin/env python
"""
Ray-based stress test for vLLM connection fixes.
This script simulates multiple nodes submitting requests simultaneously
to test the robustness of the improved timeout and circuit breaker mechanisms.
"""
import asyncio
import logging
import time
import ray
from typing import List, Dict, Any
import random
import json
from dataclasses import dataclass
from verl.utils.reward_score.gpt_as_judge import openai_llm
from verl.utils.reward_score.resilience import CircuitBreakerConfig

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    num_workers: int = 8
    requests_per_worker: int = 10
    batch_size: int = 5
    healthy_endpoints: List[str] = None
    unhealthy_endpoints: List[str] = None
    model_name: str = "Qwen2.5-7B-Instruct"
    api_key: str = "EMPTY"
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.healthy_endpoints is None:
            self.healthy_endpoints = [
                "http://10.244.174.221:30000/v1",
                "http://10.244.174.221:30001/v1",
                "http://10.244.174.221:30002/v1",
                "http://10.244.174.221:30003/v1",
                "http://10.244.174.221:30004/v1",
                "http://10.244.174.221:30005/v1",
                "http://10.244.174.221:30006/v1",
                "http://10.244.174.221:30007/v1",
            ]
        if self.unhealthy_endpoints is None:
            self.unhealthy_endpoints = [
                "http://invalid-endpoint-1:8000/v1",
                "http://invalid-endpoint-2:8000/v1",
                "http://10.244.174.221:99999/v1",  # Invalid port
            ]

@ray.remote
class StressTestWorker:
    """Ray actor that simulates a worker node making vLLM requests."""
    
    def __init__(self, worker_id: int, config: StressTestConfig):
        self.worker_id = worker_id
        self.config = config
        self.results = []
        self.client = None
        
    async def setup_client(self):
        """Setup vLLM client with endpoint manager."""
        # Mix healthy and unhealthy endpoints to test circuit breaker
        all_endpoints = self.config.healthy_endpoints + self.config.unhealthy_endpoints
        
        # Each worker uses a different primary endpoint
        primary_endpoint = self.config.healthy_endpoints[self.worker_id % len(self.config.healthy_endpoints)]
        
        self.client = openai_llm(
            provider="vllm",
            base_url=primary_endpoint,
            model_name=self.config.model_name,
            api_key=self.config.api_key
        )
        
        # Setup endpoint manager with circuit breaker
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=15.0,
            window_size=10
        )
        
        self.client.setup_endpoint_manager(all_endpoints, circuit_config)
        logger.info(f"Worker {self.worker_id} setup with primary endpoint: {primary_endpoint}")
        
    def generate_test_messages(self, batch_size: int) -> List[List[Dict[str, str]]]:
        """Generate test messages for evaluation."""
        test_prompts = [
            "What is 2+2?",
            "Explain quantum computing in simple terms.",
            "Write a short poem about programming.",
            "What are the benefits of using Python?",
            "How does machine learning work?",
            "Describe the process of photosynthesis.",
            "What is the capital of France?",
            "Explain the concept of recursion.",
            "What is the difference between AI and ML?",
            "How do neural networks learn?",
        ]
        
        messages = []
        for i in range(batch_size):
            prompt = random.choice(test_prompts)
            message = [{"role": "user", "content": prompt}]
            messages.append(message)
        
        return messages
    
    async def run_batch_test(self, batch_id: int) -> Dict[str, Any]:
        """Run a batch of requests and measure performance."""
        batch_size = self.config.batch_size
        messages = self.generate_test_messages(batch_size)
        
        start_time = time.time()
        results = []
        errors = []
        
        try:
            # Generate outputs for the batch
            batch_results = await self.client.generate_outputs_async(
                messages, 
                timeout=self.config.timeout_seconds
            )
            
            # Process results
            for i, (idx, result) in enumerate(batch_results):
                if result == "<judge>1</judge>":
                    errors.append(f"Request {i} got default response")
                else:
                    results.append({
                        "request_id": i,
                        "response_length": len(result),
                        "success": True
                    })
                    
        except Exception as e:
            error_msg = f"Batch {batch_id} failed: {str(e)}"
            errors.append(error_msg)
            logger.error(error_msg)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Get endpoint health stats
        endpoint_stats = self.client.get_endpoint_health()
        
        return {
            "worker_id": self.worker_id,
            "batch_id": batch_id,
            "batch_size": batch_size,
            "duration": duration,
            "successful_requests": len(results),
            "failed_requests": len(errors),
            "errors": errors,
            "endpoint_stats": endpoint_stats,
            "avg_response_time": duration / batch_size if batch_size > 0 else 0
        }
    
    async def run_stress_test(self) -> List[Dict[str, Any]]:
        """Run the full stress test for this worker."""
        await self.setup_client()
        
        logger.info(f"Worker {self.worker_id} starting stress test with {self.config.requests_per_worker} batches")
        
        batch_results = []
        
        for batch_id in range(self.config.requests_per_worker):
            logger.info(f"Worker {self.worker_id} running batch {batch_id + 1}/{self.config.requests_per_worker}")
            
            batch_result = await self.run_batch_test(batch_id)
            batch_results.append(batch_result)
            
            # Small delay between batches to avoid overwhelming the system
            await asyncio.sleep(0.1)
        
        logger.info(f"Worker {self.worker_id} completed stress test")
        return batch_results

async def run_distributed_stress_test(config: StressTestConfig) -> Dict[str, Any]:
    """Run distributed stress test using Ray."""
    logger.info(f"Starting distributed stress test with {config.num_workers} workers")
    
    # Create Ray actors
    workers = [StressTestWorker.remote(i, config) for i in range(config.num_workers)]
    
    # Run stress test on all workers simultaneously
    start_time = time.time()
    
    try:
        # Run all workers in parallel
        worker_results = await asyncio.gather(
            *[worker.run_stress_test.remote() for worker in workers],
            return_exceptions=True
        )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Process results
        all_batch_results = []
        worker_summaries = []
        
        for worker_id, worker_result in enumerate(worker_results):
            if isinstance(worker_result, Exception):
                logger.error(f"Worker {worker_id} failed: {worker_result}")
                worker_summaries.append({
                    "worker_id": worker_id,
                    "status": "failed",
                    "error": str(worker_result)
                })
            else:
                # Process successful worker results
                all_batch_results.extend(worker_result)
                
                # Calculate worker summary
                total_requests = sum(batch["successful_requests"] + batch["failed_requests"] for batch in worker_result)
                successful_requests = sum(batch["successful_requests"] for batch in worker_result)
                failed_requests = sum(batch["failed_requests"] for batch in worker_result)
                avg_batch_time = sum(batch["duration"] for batch in worker_result) / len(worker_result)
                
                worker_summaries.append({
                    "worker_id": worker_id,
                    "status": "completed",
                    "total_requests": total_requests,
                    "successful_requests": successful_requests,
                    "failed_requests": failed_requests,
                    "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
                    "avg_batch_time": avg_batch_time
                })
        
        # Calculate overall statistics
        total_requests = sum(batch["successful_requests"] + batch["failed_requests"] for batch in all_batch_results)
        successful_requests = sum(batch["successful_requests"] for batch in all_batch_results)
        failed_requests = sum(batch["failed_requests"] for batch in all_batch_results)
        
        return {
            "config": config,
            "total_duration": total_duration,
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / total_requests if total_requests > 0 else 0,
            "requests_per_second": total_requests / total_duration if total_duration > 0 else 0,
            "worker_summaries": worker_summaries,
            "batch_results": all_batch_results
        }
        
    except Exception as e:
        logger.error(f"Stress test failed: {e}")
        return {
            "error": str(e),
            "total_duration": time.time() - start_time
        }

def print_stress_test_results(results: Dict[str, Any]):
    """Print formatted stress test results."""
    print("\n" + "="*80)
    print("STRESS TEST RESULTS")
    print("="*80)
    
    if "error" in results:
        print(f"❌ Test failed: {results['error']}")
        return
    
    config = results["config"]
    print(f"Configuration:")
    print(f"  Workers: {config.num_workers}")
    print(f"  Requests per worker: {config.requests_per_worker}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Timeout: {config.timeout_seconds}s")
    print(f"  Healthy endpoints: {len(config.healthy_endpoints)}")
    print(f"  Unhealthy endpoints: {len(config.unhealthy_endpoints)}")
    
    print(f"\nOverall Results:")
    print(f"  Total duration: {results['total_duration']:.2f}s")
    print(f"  Total requests: {results['total_requests']}")
    print(f"  Successful requests: {results['successful_requests']}")
    print(f"  Failed requests: {results['failed_requests']}")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Requests per second: {results['requests_per_second']:.2f}")
    
    print(f"\nWorker Results:")
    for worker_summary in results["worker_summaries"]:
        if worker_summary["status"] == "completed":
            print(f"  Worker {worker_summary['worker_id']}: "
                  f"{worker_summary['successful_requests']}/{worker_summary['total_requests']} success "
                  f"({worker_summary['success_rate']:.1%}), "
                  f"avg batch time: {worker_summary['avg_batch_time']:.2f}s")
        else:
            print(f"  Worker {worker_summary['worker_id']}: FAILED - {worker_summary['error']}")
    
    # Check for any connection issues
    connection_issues = []
    for batch in results["batch_results"]:
        if batch["errors"]:
            connection_issues.extend(batch["errors"])
    
    if connection_issues:
        print(f"\nConnection Issues Found:")
        for issue in connection_issues[:10]:  # Show first 10
            print(f"  - {issue}")
        if len(connection_issues) > 10:
            print(f"  ... and {len(connection_issues) - 10} more")
    else:
        print(f"\n✅ No connection issues detected!")
    
    print("="*80)

async def main():
    """Main function to run the stress test."""
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Configure stress test
    config = StressTestConfig(
        num_workers=4,          # 4 workers simulating different nodes
        requests_per_worker=5,  # 5 batches per worker
        batch_size=8,           # 8 requests per batch
        timeout_seconds=30
    )
    
    print(f"Starting Ray-based stress test...")
    print(f"This will simulate {config.num_workers} nodes making {config.requests_per_worker * config.batch_size} requests each")
    print(f"Total requests: {config.num_workers * config.requests_per_worker * config.batch_size}")
    
    # Run the stress test
    results = await run_distributed_stress_test(config)
    
    # Print results
    print_stress_test_results(results)
    
    # Save results to file
    with open("stress_test_results.json", "w") as f:
        # Convert config to dict for JSON serialization
        results_copy = results.copy()
        if "config" in results_copy:
            results_copy["config"] = {
                "num_workers": config.num_workers,
                "requests_per_worker": config.requests_per_worker,
                "batch_size": config.batch_size,
                "timeout_seconds": config.timeout_seconds,
                "healthy_endpoints": config.healthy_endpoints,
                "unhealthy_endpoints": config.unhealthy_endpoints
            }
        json.dump(results_copy, f, indent=2)
    
    print(f"\nResults saved to stress_test_results.json")
    
    # Shutdown Ray
    ray.shutdown()

if __name__ == "__main__":
    asyncio.run(main())