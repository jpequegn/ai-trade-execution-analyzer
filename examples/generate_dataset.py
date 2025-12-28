#!/usr/bin/env python3
"""Generate JSON dataset from sample FIX messages.

This script parses the sample FIX messages and exports them to JSON format
for use in testing and evaluation.
"""

import json
from pathlib import Path

from src.parsers import parse_fix_message


def load_fix_messages(file_path: Path) -> list[str]:
    """Load FIX messages from text file, ignoring comments and blank lines."""
    messages = []
    with file_path.open() as f:
        for line in f:
            line = line.strip()
            # Skip comments and blank lines
            if not line or line.startswith("#"):
                continue
            messages.append(line)
    return messages


def generate_json_dataset(input_path: Path, output_path: Path) -> dict:
    """Parse FIX messages and generate JSON dataset."""
    messages = load_fix_messages(input_path)

    dataset = {
        "metadata": {
            "total_messages": len(messages),
            "source_file": str(input_path.name),
            "description": "Sample FIX execution report messages for testing",
            "categories": {
                "good_executions": "Messages 1-15 (expected score 8-10)",
                "average_executions": "Messages 16-35 (expected score 5-7)",
                "poor_executions": "Messages 36-50 (expected score 1-4)",
                "edge_cases": "Messages 51-55 (special scenarios)",
            },
        },
        "executions": [],
    }

    for i, raw_message in enumerate(messages, start=1):
        try:
            execution = parse_fix_message(raw_message)
            # Determine category based on message number
            if i <= 15:
                category = "good"
                expected_score_range = "8-10"
            elif i <= 35:
                category = "average"
                expected_score_range = "5-7"
            elif i <= 50:
                category = "poor"
                expected_score_range = "1-4"
            else:
                category = "edge_case"
                expected_score_range = "varies"

            entry = {
                "id": f"MSG{i:03d}",
                "category": category,
                "expected_score_range": expected_score_range,
                "raw_message": raw_message,
                "parsed": {
                    "order_id": execution.order_id,
                    "symbol": execution.symbol,
                    "side": execution.side,
                    "quantity": execution.quantity,
                    "price": execution.price,
                    "venue": execution.venue,
                    "timestamp": execution.timestamp.isoformat(),
                    "fill_type": execution.fill_type,
                    "fix_version": execution.fix_version,
                    "exec_type": execution.exec_type,
                    "cum_qty": execution.cum_qty,
                    "avg_px": execution.avg_px,
                },
            }
            dataset["executions"].append(entry)
        except Exception as e:
            print(f"Error parsing message {i}: {e}")
            dataset["executions"].append(
                {
                    "id": f"MSG{i:03d}",
                    "error": str(e),
                    "raw_message": raw_message,
                }
            )

    # Write JSON output
    with output_path.open("w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset['executions'])} entries to {output_path}")
    return dataset


if __name__ == "__main__":
    examples_dir = Path(__file__).parent
    input_file = examples_dir / "sample_fix_messages.txt"
    output_file = examples_dir / "sample_fix_messages.json"

    generate_json_dataset(input_file, output_file)
