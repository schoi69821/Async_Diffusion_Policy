#!/usr/bin/env python3
"""Offline evaluation of v8 policy on recorded episodes."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/processed/test")
    args = parser.parse_args()
    print(f"Offline eval with {args.checkpoint}")


if __name__ == "__main__":
    main()
