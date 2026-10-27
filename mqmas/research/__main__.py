"""``python -m mqmas.research [root] [lab|swing|season|all]`` — run the labs whose report is missing or outdated."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


def _current(path: Path, version: int) -> bool:
    try:
        return bool(json.loads(path.read_text()).get("version") == version)
    except (OSError, ValueError):
        return False


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s research %(levelname)s %(message)s", stream=sys.stdout)
    log = logging.getLogger("research")
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/research")
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("lab", "all"):
        from mqmas.research.lab import VERSION, Lab, save, summarize

        if which == "lab" or not _current(root / "report.json", VERSION):
            report = Lab(root).run()
            path = save(report, root)
            for line in summarize(report):
                log.info(line)
            log.info("report saved to %s", path)
    if which in ("swing", "all"):
        from mqmas.research import swing

        path = root / "swing_report.json"
        if which == "swing" or not _current(path, swing.VERSION):
            rep = swing.SwingLab(root).run()
            root.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in swing.summarize(rep):
                log.info(line)
            log.info("swing report saved to %s", path)
    if which in ("season", "all"):
        from mqmas.research import seasonality

        path = root / "season_report.json"
        if which == "season" or not _current(path, seasonality.VERSION):
            rep = seasonality.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in seasonality.summarize(rep):
                log.info(line)
            log.info("seasonality report saved to %s", path)


if __name__ == "__main__":
    main()
