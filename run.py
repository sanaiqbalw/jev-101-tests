import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Jev vs Claude use-case comparison.")
    parser.add_argument("config")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Checked before importing usecases: clients.py builds a boto3 client at import.
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        sys.exit(2)

    import usecases

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.generate:
        if Path(cfg["dataset"]).exists() and not args.force:
            print(f"{cfg['dataset']} exists; pass --force to overwrite", file=sys.stderr)
            sys.exit(1)
        getattr(usecases, f"generate_{cfg['usecase']}")(cfg)
        sys.exit(0)

    # Imported before the run so a broken report.py fails before any money is spent.
    from report import summarize

    lines = Path(cfg["dataset"]).read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()][: cfg["max_rows"]]
    run_dir = Path("results") / cfg["usecase"] / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    rows = getattr(usecases, f"run_{cfg['usecase']}")(records, cfg, run_dir)
    (run_dir / "summary.md").write_text(summarize(rows, cfg["usecase"], cfg["confidence_threshold"]))
    print(run_dir)


if __name__ == "__main__":
    main()
