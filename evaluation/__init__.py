"""Accuracy evaluation of TeachLens against blind reference labels (report/evaluation_plan.md).

The pipeline code and the saved lesson folders are read from the pipeline checkout
(TEACHLENS_PIPELINE_ROOT, default: this repository); nothing there is changed.
Results go to outputs/evaluation/<YYYY-MM-DD_HHMM>_<name>/ in this checkout (ignored by Git).
"""

import os
from pathlib import Path
import sys

PIPELINE_ROOT = Path(os.environ.get("TEACHLENS_PIPELINE_ROOT", Path(__file__).resolve().parents[1]))
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "evaluation"

if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
