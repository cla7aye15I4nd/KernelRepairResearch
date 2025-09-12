#!/usr/bin/env python3

import statistics
from pathlib import Path

import yaml
from transformers import RobertaTokenizer

VULN_DIR = Path(__file__).parent.parent / "vuln"


def analyze_token_lengths() -> None:
    """Analyze token length distribution of vulnerability reports"""

    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    token_lengths = []
    positive_lengths = []
    negative_lengths = []

    print("Processing reports...")

    for vuln_dir in VULN_DIR.iterdir():
        if not vuln_dir.is_dir():
            continue

        config_yaml = vuln_dir / "config.yaml"
        report_txt = vuln_dir / "report.txt"

        if config_yaml.exists() and report_txt.exists():
            # Read report text
            text = report_txt.read_text()

            # Tokenize and get length
            tokens = tokenizer.tokenize(text)
            length = len(tokens)
            token_lengths.append(length)

            # Check label for class-specific statistics
            config_data = yaml.safe_load(config_yaml.read_text())
            hunk_count = config_data["hunk_count"]
            covered_count = config_data["covered_count"]
            is_positive = (hunk_count == 1) and (covered_count == 1)

            if is_positive:
                positive_lengths.append(length)
            else:
                negative_lengths.append(length)

    print(f"\nProcessed {len(token_lengths)} reports")
    print(f"Positive samples: {len(positive_lengths)}")
    print(f"Negative samples: {len(negative_lengths)}")

    # Overall statistics
    print("\n=== OVERALL TOKEN LENGTH STATISTICS ===")
    print(f"Total reports: {len(token_lengths)}")
    print(f"Min length: {min(token_lengths):,} tokens")
    print(f"Max length: {max(token_lengths):,} tokens")
    print(f"Mean length: {statistics.mean(token_lengths):,.1f} tokens")
    print(f"Median length: {statistics.median(token_lengths):,} tokens")
    print(f"Standard deviation: {statistics.stdev(token_lengths):,.1f} tokens")

    # Percentiles
    sorted_lengths = sorted(token_lengths)
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print("\nPercentiles:")
    for p in percentiles:
        idx = int(len(sorted_lengths) * p / 100) - 1
        print(f"{p:2d}th percentile: {sorted_lengths[idx]:,} tokens")

    # Length ranges
    print("\n=== LENGTH DISTRIBUTION ===")
    ranges = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 16384), (16384, 32768), (32768, float("inf"))]

    for start, end in ranges:
        if end == float("inf"):
            count = sum(1 for length in token_lengths if length >= start)
            print(f"{start:,}+ tokens: {count:,} reports ({count/len(token_lengths)*100:.1f}%)")
        else:
            count = sum(1 for length in token_lengths if start <= length < end)
            print(f"{start:,}-{end-1:,} tokens: {count:,} reports ({count/len(token_lengths)*100:.1f}%)")

    # Class-specific statistics
    if positive_lengths and negative_lengths:
        print("\n=== CLASS-SPECIFIC STATISTICS ===")
        print("Positive class (single hunk, fully covered):")
        print(f"  Count: {len(positive_lengths)}")
        print(f"  Mean: {statistics.mean(positive_lengths):,.1f} tokens")
        print(f"  Median: {statistics.median(positive_lengths):,} tokens")
        print(f"  Min: {min(positive_lengths):,} tokens")
        print(f"  Max: {max(positive_lengths):,} tokens")

        print("Negative class:")
        print(f"  Count: {len(negative_lengths)}")
        print(f"  Mean: {statistics.mean(negative_lengths):,.1f} tokens")
        print(f"  Median: {statistics.median(negative_lengths):,} tokens")
        print(f"  Min: {min(negative_lengths):,} tokens")
        print(f"  Max: {max(negative_lengths):,} tokens")

    # Reports exceeding common model limits
    print("\n=== MODEL CONTEXT LENGTH ANALYSIS ===")
    limits = [512, 1024, 2048, 4096, 8192, 16384]
    for limit in limits:
        exceeding = sum(1 for length in token_lengths if length > limit)
        print(f"Reports exceeding {limit:,} tokens: {exceeding:,} ({exceeding/len(token_lengths)*100:.1f}%)")

    # Top 10 longest reports
    print("\n=== TOP 10 LONGEST REPORTS ===")
    longest_reports = []
    for vuln_dir in VULN_DIR.iterdir():
        if not vuln_dir.is_dir():
            continue

        config_yaml = vuln_dir / "config.yaml"
        report_txt = vuln_dir / "report.txt"

        if config_yaml.exists() and report_txt.exists():
            text = report_txt.read_text()
            tokens = tokenizer.tokenize(text)
            length = len(tokens)
            longest_reports.append((vuln_dir.name, length))

    longest_reports.sort(key=lambda x: x[1], reverse=True)
    for i, (name, length) in enumerate(longest_reports[:10], 1):
        print(f"{i:2d}. {name}: {length:,} tokens")


if __name__ == "__main__":
    analyze_token_lengths()
