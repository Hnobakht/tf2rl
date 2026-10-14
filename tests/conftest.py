"""Keep all test artifacts within this checkout, including temporary files."""
from pathlib import Path
import os
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
_root = Path(__file__).resolve().parents[1]
_local_tmp = _root / ".tmp"
_local_tmp.mkdir(exist_ok=True)
os.environ["TMPDIR"] = str(_local_tmp)
tempfile.tempdir = str(_local_tmp)
os.environ.setdefault("MPLCONFIGDIR", str(_root / ".cache" / "matplotlib"))
