#!/usr/bin/env python3
"""Rebuild descriptive reports from saved events, without model calls or changing logs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motiverooms.evaluation import summarize
from motiverooms.experiment import compute_metrics, read_events


def summarize_batch(folder):
    folder = Path(folder)
    records = []
    for path in sorted(folder.glob("*.jsonl")):
        events = read_events(path)
        start = next(e for e in events if e["type"] == "start")
        metadata = dict(start["metadata"])
        metadata.setdefault("observation_version", start["config"].get("observation_version", "v1"))
        records.append(dict(metadata=metadata, metrics=compute_metrics(events), path=str(path.resolve())))
    if not records:
        raise ValueError(f"No episode JSONL logs in {folder}")
    summarize(records, folder)
    return folder / "summary.md"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    print(summarize_batch(args.batch))
