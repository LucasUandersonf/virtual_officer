"""``python -m mqmas.research [root] [lab|swing|season|ta|conf|port|acct|swingbook|classic|tcc|livro|cross|arena|talib|
trend|waves|all]``.

Runs the labs whose report is missing or outdated."""

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
    if which in ("acct", "all"):
        from mqmas.config import Settings
        from mqmas.research import account

        path = root / "account_report.json"
        if which == "acct" or not _current(path, account.VERSION):
            key = Settings().book_strategies.split(";")[0].strip()
            rep = account.run(root, key)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in account.summarize(rep):
                log.info(line)
            log.info("account report saved to %s", path)
    if which in ("swingbook", "all"):
        from mqmas.research import account

        path = root / "swingbook_report.json"
        if which == "swingbook" or not _current(path, account.SWING_VERSION):
            rep = account.run_swing(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in account.summarize_swing(rep):
                log.info(line)
            log.info("swing book report saved to %s", path)
    if which in ("classic", "all"):
        from mqmas.research import classic_study

        path = root / "classic_report.json"
        if which == "classic" or not _current(path, classic_study.VERSION):
            rep = classic_study.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in classic_study.summarize(rep):
                log.info(line)
            log.info("classic report saved to %s", path)
    if which in ("tcc", "all"):
        from mqmas.research import tcc

        path = root / "tcc_report.json"
        if which == "tcc" or not _current(path, tcc.VERSION):
            rep = tcc.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in tcc.summarize(rep):
                log.info(line)
            log.info("tcc report saved to %s", path)
    if which in ("livro", "all"):
        from mqmas.research import livro

        path = root / "livro_report.json"
        if which == "livro" or not _current(path, livro.VERSION):
            rep = livro.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in livro.summarize(rep):
                log.info(line)
            log.info("livro report saved to %s", path)
    if which in ("cross", "all"):
        from mqmas.research import crosscheck

        path = root / "crosscheck_report.json"
        if which == "cross" or not _current(path, crosscheck.VERSION):
            rep = crosscheck.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in crosscheck.summarize(rep):
                log.info(line)
            log.info("crosscheck report saved to %s", path)
    if which in ("arena", "all"):
        from mqmas.research import arena

        path = root / "arena_report.json"
        if which == "arena" or not _current(path, arena.VERSION):
            rep = arena.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in arena.summarize(rep):
                log.info(line)
            log.info("arena report saved to %s", path)
    if which in ("talib", "all"):
        from mqmas.research import talib_study

        path = root / "talib_report.json"
        if which == "talib" or not _current(path, talib_study.VERSION):
            rep = talib_study.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in talib_study.summarize(rep):
                log.info(line)
            log.info("talib report saved to %s", path)
    if which in ("trend", "all"):
        from mqmas.research import trend_study

        path = root / "trend_report.json"
        if which == "trend" or not _current(path, trend_study.VERSION):
            rep = trend_study.TrendStudy(root).run()
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in trend_study.summarize(rep):
                log.info(line)
            log.info("trend report saved to %s", path)
    if which in ("waves", "all"):
        from mqmas.research import waves

        path = root / "waves_report.json"
        if which == "waves" or not _current(path, waves.VERSION):
            rep = waves.run(root)
            path.write_text(json.dumps(rep, default=str, ensure_ascii=False))
            for line in waves.summarize(rep):
                log.info(line)
            log.info("waves report saved to %s", path)


if __name__ == "__main__":
    main()
