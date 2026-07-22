import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_import_smoke(code: str, cwd: Path = ROOT):
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_package_imports_from_repository_root():
    result = _run_import_smoke(
        "import BitFlip.Run_PSRL; import GridWorld.Run_PSRL; print('PACKAGE_IMPORTS_OK')"
    )

    assert "PACKAGE_IMPORTS_OK" in result.stdout


def test_gridworld_script_style_imports_from_subdirectory():
    result = _run_import_smoke(
        "import GridWorld, PSRL_Agents, Run_PSRL; print('GRIDWORLD_SCRIPT_IMPORTS_OK')",
        cwd=ROOT / "GridWorld",
    )

    assert "GRIDWORLD_SCRIPT_IMPORTS_OK" in result.stdout


def test_bitflip_script_style_imports_from_subdirectory():
    result = _run_import_smoke(
        "import BitFlip, PSRL_Agents, Run_PSRL; print('BITFLIP_SCRIPT_IMPORTS_OK')",
        cwd=ROOT / "BitFlip",
    )

    assert "BITFLIP_SCRIPT_IMPORTS_OK" in result.stdout
