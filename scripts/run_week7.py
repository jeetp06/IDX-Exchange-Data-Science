"""Run the complete April-only Week 7 analysis in the existing environment."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.week7 import run

if __name__ == '__main__':
    result = run(ROOT)
    print('Week 7 complete:', result['selected'], result['selected_R2'])
