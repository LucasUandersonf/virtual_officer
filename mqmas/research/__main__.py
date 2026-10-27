"""``python -m mqmas.research [root]`` — run the lab, write report.json, log a summary."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from mqmas.research.lab import Lab, save, summarize


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s research %(levelname)s %(message)s", stream=sys.stdout)
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/research")
    report = Lab(root).run()
    path = save(report, root)
    for line in summarize(report):
        logging.getLogger("research").info(line)
    logging.getLogger("research").info("report saved to %s", path)


if __name__ == "__main__":
    main()
