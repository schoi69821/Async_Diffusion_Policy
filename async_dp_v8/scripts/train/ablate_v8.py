#!/usr/bin/env python3
"""Ablation study runner for v8 components."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation", choices=["no_phase", "no_contact", "no_crop", "full"], default="full")
    args = parser.parse_args()
    print(f"Ablation: {args.ablation}")
    print("Placeholder for ablation study implementation")


if __name__ == "__main__":
    main()
