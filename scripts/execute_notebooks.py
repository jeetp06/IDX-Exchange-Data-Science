"""Execute every notebook locally with the current Python interpreter."""
import json
import os
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ['IPYTHONDIR'] = str(ROOT / 'outputs' / 'ipython')
    os.environ['JUPYTER_RUNTIME_DIR'] = str(ROOT / 'outputs' / 'jupyter_runtime')
    # Use an explicit kernel so a global Jupyter install cannot select another Python.
    kernel_dir = ROOT / "outputs" / "local_kernel" / "idx-local"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "IDX local environment", "language": "python",
    }), encoding="utf-8")
    manager = KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)])
    paths = [ROOT / 'notebooks' / name for name in sys.argv[1:]] if len(sys.argv)>1 else sorted((ROOT / "notebooks").glob("*.ipynb"))
    for path in paths:
        print(f"Executing {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=3600, kernel_name="idx-local",
                                resources={"metadata": {"path": str(ROOT)}})
        client.km = client.create_kernel_manager()
        client.km.kernel_spec_manager = manager
        client.execute()
        # Retain the portable default kernel name for interactive use.
        nbformat.write(notebook, path)
        print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
