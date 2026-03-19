#!/usr/bin/env python3
"""Benchmark policy rollout performance."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-rollouts", type=int, default=20)
    args = parser.parse_args()
    print(f"Benchmarking {args.num_rollouts} rollouts")


if __name__ == "__main__":
    main()
