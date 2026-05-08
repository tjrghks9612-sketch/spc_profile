from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qt_runtime import prepare_qt_runtime, qt_diagnostic_text


def main() -> int:
    added = prepare_qt_runtime()
    print(qt_diagnostic_text())
    if added:
        print("- DLL search dirs:")
        for path in added:
            print(f"  {path}")

    try:
        from PySide6 import QtCore, QtWidgets
    except Exception as exc:
        print(f"Qt import failed: {exc}")
        return 1

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    print(f"Qt import OK: Qt {QtCore.qVersion()}")
    print(f"QApplication OK: {app.platformName()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
