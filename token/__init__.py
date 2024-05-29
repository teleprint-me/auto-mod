import sys
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent / "gguf"))
    import gguf
