"""
Real-time training monitor.
Reads train_log.csv and displays progress as text + optional matplotlib plot.

Usage:
    uv run python scripts/monitor_training.py --log checkpoints/vision_policy_v4/train_log.csv
    uv run python scripts/monitor_training.py --log checkpoints/vision_policy_v4/train_log.csv --plot
    uv run python scripts/monitor_training.py --log checkpoints/vision_policy_v4/train_log.csv --watch 10
"""
import argparse
import os
import sys
import time
import csv


def read_log(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'epoch': int(row['epoch']),
                'train_loss': float(row['train_loss']),
                'val_loss': float(row['val_loss']),
                'best_val': float(row['best_val']),
                'lr': float(row['lr']),
                'time_s': float(row['time_s']),
            })
    return rows


def print_table(rows, last_n=20):
    if not rows:
        print("No data yet.")
        return

    total_epochs = rows[-1]['epoch']
    total_time = sum(r['time_s'] for r in rows)
    best = min(rows, key=lambda r: r['val_loss'])

    print(f"\n{'='*70}")
    print(f"  Epochs: {total_epochs}/200 | "
          f"Time: {total_time/3600:.1f}h | "
          f"Best: {best['best_val']:.6f} (epoch {best['epoch']})")
    print(f"{'='*70}")
    print(f"  {'Epoch':>5} | {'Train':>10} | {'Val':>10} | {'Best':>10} | {'LR':>10} | {'Time':>5}")
    print(f"  {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*5}")

    display = rows[-last_n:]
    for r in display:
        marker = ' *' if r['val_loss'] == r['best_val'] else '  '
        print(f"  {r['epoch']:>5} | {r['train_loss']:>10.6f} | {r['val_loss']:>10.6f} | "
              f"{r['best_val']:>10.6f} | {r['lr']:>10.2e} | {r['time_s']:>4.0f}s{marker}")

    if len(rows) > last_n:
        print(f"  ... ({len(rows) - last_n} earlier epochs hidden)")

    # ETA
    if len(rows) >= 2:
        avg_time = total_time / len(rows)
        remaining = (200 - total_epochs) * avg_time
        print(f"\n  Avg: {avg_time:.0f}s/epoch | ETA: {remaining/3600:.1f}h")
    print()


def plot_losses(rows, save_path=None):
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Use --no-plot or install: pip install matplotlib")
        return

    epochs = [r['epoch'] for r in rows]
    train = [r['train_loss'] for r in rows]
    val = [r['val_loss'] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale
    ax1.plot(epochs, train, 'b-', alpha=0.7, label='Train')
    ax1.plot(epochs, val, 'r-', alpha=0.7, label='Val')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Progress (linear)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Log scale
    ax2.plot(epochs, train, 'b-', alpha=0.7, label='Train')
    ax2.plot(epochs, val, 'r-', alpha=0.7, label='Val')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log)')
    ax2.set_title('Training Progress (log scale)')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"  Plot saved to {save_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--log", required=True, help="Path to train_log.csv")
    parser.add_argument("--plot", action="store_true", help="Show matplotlib plot")
    parser.add_argument("--save-plot", default=None, help="Save plot to file (e.g. loss.png)")
    parser.add_argument("--watch", type=int, default=0,
                        help="Auto-refresh every N seconds (0=once)")
    parser.add_argument("--last", type=int, default=20, help="Show last N epochs")
    args = parser.parse_args()

    if args.watch > 0:
        try:
            while True:
                os.system('clear')
                rows = read_log(args.log)
                print_table(rows, args.last)
                if rows and rows[-1]['epoch'] >= 200:
                    print("  Training complete!")
                    break
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
    else:
        rows = read_log(args.log)
        print_table(rows, args.last)
        if args.plot or args.save_plot:
            if rows:
                plot_losses(rows, args.save_plot)
            else:
                print("No data to plot.")


if __name__ == "__main__":
    main()
