#!/usr/bin/env python3
"""
优化的Ray任务提交示例
"""

import ray
import torch
import time
from typing import List


@ray.remote
def calculate_single_bleu_score_optimized(tokens: List, start_idx: int, end_idx: int):
    """优化的单个BLEU计算任务"""
    # 模拟BLEU计算
    import time
    time.sleep(0.1)  # 模拟计算时间
    return 0.5


# 方法1: 当前的串行创建方式
@ray.remote(max_retries=3)
def calculate_self_bleu_metric_current(responses, batch_size, curriculum_rollout_n):
    """当前的实现 - 串行创建任务"""
    tokenized_sequences = responses.tolist()
    
    futures = []
    start_time = time.time()
    
    for i in range(batch_size):
        start_idx = i * curriculum_rollout_n
        end_idx = (i + 1) * curriculum_rollout_n
        future = calculate_single_bleu_score_optimized.remote(
            tokenized_sequences, start_idx, end_idx
        )
        futures.append(future)
    
    creation_time = time.time() - start_time
    print(f"任务创建时间: {creation_time:.3f}s")
    
    bleu_scores = ray.get(futures)
    return torch.tensor(bleu_scores, dtype=torch.float32)


# 方法2: 批量创建优化
@ray.remote(max_retries=3) 
def calculate_self_bleu_metric_batch_optimized(responses, batch_size, curriculum_rollout_n):
    """优化版本 - 批量创建任务"""
    tokenized_sequences = responses.tolist()
    
    start_time = time.time()
    
    # 批量创建所有任务参数
    task_params = [
        (tokenized_sequences, i * curriculum_rollout_n, (i + 1) * curriculum_rollout_n)
        for i in range(batch_size)
    ]
    
    # 使用列表推导式批量创建futures（更快的Python执行）
    futures = [
        calculate_single_bleu_score_optimized.remote(tokens, start_idx, end_idx)
        for tokens, start_idx, end_idx in task_params
    ]
    
    creation_time = time.time() - start_time
    print(f"批量任务创建时间: {creation_time:.3f}s")
    
    bleu_scores = ray.get(futures)
    return torch.tensor(bleu_scores, dtype=torch.float32)


# 方法3: 分块提交（适用于大量任务）
@ray.remote(max_retries=3)
def calculate_self_bleu_metric_chunked(responses, batch_size, curriculum_rollout_n, chunk_size=50):
    """分块提交版本 - 适用于非常大的batch_size"""
    tokenized_sequences = responses.tolist()
    
    all_futures = []
    start_time = time.time()
    
    # 分块处理，避免一次性创建过多任务
    for chunk_start in range(0, batch_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, batch_size)
        
        # 创建这个chunk的任务
        chunk_futures = [
            calculate_single_bleu_score_optimized.remote(
                tokenized_sequences,
                i * curriculum_rollout_n,
                (i + 1) * curriculum_rollout_n
            )
            for i in range(chunk_start, chunk_end)
        ]
        
        all_futures.extend(chunk_futures)
        
        # 可选：小延迟以避免瞬时过载Ray调度器
        # time.sleep(0.001)
    
    creation_time = time.time() - start_time
    print(f"分块任务创建时间: {creation_time:.3f}s")
    
    bleu_scores = ray.get(all_futures)
    return torch.tensor(bleu_scores, dtype=torch.float32)


# 方法4: 预热+批量提交
@ray.remote(max_retries=3)
def calculate_self_bleu_metric_warmed(responses, batch_size, curriculum_rollout_n):
    """预热版本 - 减少冷启动开销"""
    tokenized_sequences = responses.tolist()
    
    # 预热：创建一个小任务确保worker已就绪
    warmup_future = calculate_single_bleu_score_optimized.remote(
        tokenized_sequences[:10] if len(tokenized_sequences) > 10 else tokenized_sequences, 
        0, min(2, len(tokenized_sequences))
    )
    
    start_time = time.time()
    
    # 批量创建主任务
    main_futures = [
        calculate_single_bleu_score_optimized.remote(
            tokenized_sequences,
            i * curriculum_rollout_n,
            (i + 1) * curriculum_rollout_n
        )
        for i in range(batch_size)
    ]
    
    creation_time = time.time() - start_time
    print(f"预热+批量创建时间: {creation_time:.3f}s")
    
    # 等待预热完成（通常很快）
    ray.get(warmup_future)
    
    # 获取主结果
    bleu_scores = ray.get(main_futures)
    return torch.tensor(bleu_scores, dtype=torch.float32)


def benchmark_task_creation():
    """基准测试不同的任务创建方式"""
    print("🚀 Ray任务创建优化基准测试")
    
    # 模拟数据
    batch_size = 100
    curriculum_rollout_n = 5
    sequence_length = 50
    responses = torch.randint(0, 1000, (batch_size * curriculum_rollout_n, sequence_length))
    
    print(f"\n测试参数:")
    print(f"  batch_size: {batch_size}")
    print(f"  curriculum_rollout_n: {curriculum_rollout_n}")
    print(f"  总任务数: {batch_size}")
    
    methods = [
        ("当前方法（串行创建）", calculate_self_bleu_metric_current),
        ("优化方法（批量创建）", calculate_self_bleu_metric_batch_optimized),  
        ("分块方法（chunk=25）", lambda r, b, n: calculate_self_bleu_metric_chunked.remote(r, b, n, 25)),
        ("预热方法", calculate_self_bleu_metric_warmed),
    ]
    
    results = {}
    
    for method_name, method_func in methods:
        print(f"\n🧪 测试: {method_name}")
        
        start_time = time.time()
        future = method_func.remote(responses, batch_size, curriculum_rollout_n)
        result = ray.get(future)
        total_time = time.time() - start_time
        
        results[method_name] = {
            "total_time": total_time,
            "result_shape": result.shape
        }
        
        print(f"  总执行时间: {total_time:.3f}s")
        print(f"  结果形状: {result.shape}")
        
    print(f"\n📊 性能对比:")
    baseline_time = results["当前方法（串行创建）"]["total_time"]
    
    for method_name, metrics in results.items():
        speedup = baseline_time / metrics["total_time"]
        print(f"  {method_name}: {speedup:.2f}x 加速")
        
    return results


if __name__ == "__main__":
    # 确保Ray已初始化
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        benchmark_task_creation()
    finally:
        ray.shutdown()