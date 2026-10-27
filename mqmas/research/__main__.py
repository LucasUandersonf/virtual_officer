"""``python -m mqmas.research [root] [lab|swing|season|ta|conf|port|all]`` — run the labs whose report is missing or outdated."""

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
    if which in ("ta", "all"):
        from mqmas.research import ta_study

        path = root / "ta_report.json"
        if which == "ta" or not _current(path, ta_study.VERSION):
            rep = ta_study.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in ta_study.summarize(rep):
                log.info(line)
            log.info("ta study saved to %s", path)
    if which in ("conf", "all"):
        from mqmas.research import confluence_study

        path, ta_path = root / "confluence_report.json", root / "ta_report.json"
        stale = not _current(path, confluence_study.VERSION) or (
            ta_path.exists() and path.exists() and ta_path.stat().st_mtime > path.stat().st_mtime)
        if ta_path.exists() and (which == "conf" or stale):
            rep = confluence_study.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in confluence_study.summarize(rep):
                log.info(line)
            log.info("confluence study saved to %s", path)
    if which in ("port", "all"):
        from mqmas.research import pattern_portfolio

        path, ta_path = root / "pattern_portfolio.json", root / "ta_report.json"
        stale = not _current(path, pattern_portfolio.VERSION) or (
            ta_path.exists() and path.exists() and ta_path.stat().st_mtime > path.stat().st_mtime)
        if ta_path.exists() and (which == "port" or stale):
            rep = pattern_portfolio.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in pattern_portfolio.summarize(rep):
                log.info(line)
            log.info("pattern portfolio saved to %s", path)


if __name__ == "__main__":
    main()
