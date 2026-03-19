#!/usr/bin/env python3
"""Replay a recorded episode on the robot."""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    print(f"Replaying {args.episode} at {args.speed}x")


if __name__ == "__main__":
    main()
