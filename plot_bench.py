import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

def main():
    with open("bench_results.json", "r") as f:
        results = json.load(f)

    batch_sizes = [r["num_seqs"] for r in results]
    throughputs = [r["throughput"] for r in results]
    per_token_latency = [r["time"] / r["total_tokens"] * 1000 for r in results]
    preemptions = [r.get("preemptions", 0) for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    # 图 1: Throughput vs Batch Size
    ax1.plot(batch_sizes, throughputs, 'o-', color='#2196F3', linewidth=2, markersize=8)
    peak_idx = throughputs.index(max(throughputs))
    ax1.plot(batch_sizes[peak_idx], throughputs[peak_idx], 'o', color='#E53935', markersize=12, zorder=5)
    ax1.set_xlabel('Batch Size', fontsize=11)
    ax1.set_ylabel('Throughput (tokens/s)', fontsize=11)
    ax1.set_title('Throughput vs Batch Size', fontsize=12, fontweight='bold')
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=9)
    ax1.grid(True, alpha=0.3)
    for bs, tp in zip(batch_sizes, throughputs):
        ax1.annotate(f'{tp:.0f}', (bs, tp), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8)
    ax1.annotate('peak', (batch_sizes[peak_idx], throughputs[peak_idx]),
                textcoords="offset points", xytext=(18, -5), ha='left',
                fontsize=9, color='#E53935', fontweight='bold')

    # 图 2: Per-Token Latency vs Batch Size
    ax2.plot(batch_sizes, per_token_latency, 's-', color='#FF9800', linewidth=2, markersize=8)
    min_idx = per_token_latency.index(min(per_token_latency))
    ax2.plot(batch_sizes[min_idx], per_token_latency[min_idx], 's', color='#E53935', markersize=12, zorder=5)
    ax2.set_xlabel('Batch Size', fontsize=11)
    ax2.set_ylabel('Per-Token Latency (ms)', fontsize=11)
    ax2.set_title('Per-Token Latency vs Batch Size', fontsize=12, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_xticks(batch_sizes)
    ax2.set_xticklabels([str(bs) for bs in batch_sizes], fontsize=9)
    ax2.grid(True, alpha=0.3)
    for bs, lat in zip(batch_sizes, per_token_latency):
        ax2.annotate(f'{lat:.2f}', (bs, lat), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=8)
    ax2.annotate('best', (batch_sizes[min_idx], per_token_latency[min_idx]),
                textcoords="offset points", xytext=(18, -5), ha='left',
                fontsize=9, color='#E53935', fontweight='bold')

    # 图 3: Preemptions vs Batch Size
    colors = ['#2196F3' if p == 0 else '#E53935' for p in preemptions]
    ax3.bar([str(bs) for bs in batch_sizes], preemptions, color=colors, alpha=0.85)
    ax3.set_xlabel('Batch Size', fontsize=11)
    ax3.set_ylabel('Preemption Count', fontsize=11)
    ax3.set_title('Preemptions vs Batch Size', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, p in enumerate(preemptions):
        if p > 0:
            ax3.annotate(f'{p}', (i, p), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=10, fontweight='bold', color='#E53935')

    fig.suptitle('Nano-vLLM Benchmark (Qwen3-0.6B, RTX 3070 Laptop 8GB)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('bench_throughput.png', dpi=150, bbox_inches='tight')
    print("图表已保存到 bench_throughput.png")

if __name__ == "__main__":
    main()
