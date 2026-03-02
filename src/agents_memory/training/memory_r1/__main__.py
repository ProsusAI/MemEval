"""Allow running training subcommands via python -m agents_memory.training.memory_r1."""

import sys


def main():
    print("Memory-R1 Training Pipeline")
    print()
    print("Available commands:")
    print("  python -m agents_memory.training.memory_r1.prepare_data  - Prepare SFT data")
    print("  python -m agents_memory.training.memory_r1.sft           - Run SFT training")
    print("  python -m agents_memory.training.memory_r1.grpo          - Run GRPO RL training")
    print()
    print("Or use the convenience scripts:")
    print("  ./scripts/prepare_memory_r1_data.sh")
    print("  ./scripts/train_memory_r1_sft.sh")
    print("  ./scripts/train_memory_r1_grpo.sh")


if __name__ == "__main__":
    main()
