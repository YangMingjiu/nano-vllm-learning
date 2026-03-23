import os
import time
import json
from random import randint, seed
from nanovllm import LLM, SamplingParams

def run_benchmark(llm, num_seqs, max_input_len=512, max_output_len=512):
    seed(42)
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_output_len)) for _ in range(num_seqs)]

    total_tokens = sum(sp.max_tokens for sp in sampling_params)

    # 重置计数器
    llm.scheduler.reset_stats()

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.time() - t

    throughput = total_tokens / elapsed

    # 收集统计数据
    stats = {
        "num_seqs": num_seqs,
        "total_tokens": total_tokens,
        "time": round(elapsed, 2),
        "throughput": round(throughput, 2),
        "prefill_steps": llm.scheduler.num_prefill,
        "decode_steps": llm.scheduler.num_decode,
        "preemptions": llm.scheduler.num_preempt,
        "cache_hits": llm.scheduler.block_manager.num_cache_hit,
    }
    return stats

def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # warmup
    print("Warming up...")
    llm.generate(["Warmup"], SamplingParams())

    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    results = []

    for bs in batch_sizes:
        print(f"\n--- Testing batch_size = {bs} ---")
        result = run_benchmark(llm, num_seqs=bs)
        results.append(result)
        print(f"  Seqs: {bs}, Throughput: {result['throughput']} tok/s, "
              f"Prefill: {result['prefill_steps']}, Decode: {result['decode_steps']}, "
              f"Preempt: {result['preemptions']}, Cache Hit: {result['cache_hits']}")

    with open("bench_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n\n=== 汇总 ===")
    print(f"{'Batch':>6} | {'Tok/s':>8} | {'Prefill':>7} | {'Decode':>7} | {'Preempt':>7} | {'Cache Hit':>9}")
    print("-" * 60)
    for r in results:
        print(f"{r['num_seqs']:>6} | {r['throughput']:>8} | {r['prefill_steps']:>7} | "
              f"{r['decode_steps']:>7} | {r['preemptions']:>7} | {r['cache_hits']:>9}")

    print("\n结果已保存到 bench_results.json")

if __name__ == "__main__":
    main()
