"""Run comprehensive evaluation of the already fitted Weeks 4-7 pipelines."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.week8 import run

if __name__ == '__main__':
    run(ROOT)
