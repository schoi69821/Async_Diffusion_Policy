#!/usr/bin/env python3
"""Export processed data in LeRobot-compatible format."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/train")
    parser.add_argument("--output-dir", default="data/processed/lerobot")
    args = parser.parse_args()

    print("LeRobot export: placeholder for future implementation")
    print(f"Would export from {args.data_dir} to {args.output_dir}")


if __name__ == "__main__":
    main()
