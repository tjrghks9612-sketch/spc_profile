from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import platform
import sys
from pathlib import Path


_DLL_DIRECTORY_HANDLES = []


def _package_dir(package_name: str) -> Path | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin:
        return Path(spec.origin).resolve().parent
    return None


def prepare_qt_runtime() -> list[Path]:
    os.environ.setdefault("QT_API", "pyside6")
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return []

    added: list[Path] = []
    for package_name in ("shiboken6", "PySide6"):
        package_dir = _package_dir(package_name)
        if package_dir is None:
            continue
        for candidate in (package_dir, package_dir / "lib"):
            if candidate.exists():
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(candidate)))
                added.append(candidate)
    return added


def qt_diagnostic_text(error: BaseException | None = None) -> str:
    lines = [
        "Qt/PySide6 런타임 진단",
        f"- Python: {sys.version.split()[0]} ({sys.executable})",
        f"- Platform: {platform.platform()}",
    ]
    for package_name in ("PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6"):
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"- {package_name}: {version}")
    if error is not None:
        lines.append(f"- Import error: {error}")
    return "\n".join(lines)


def qt_failure_help(error: BaseException) -> str:
    return f"""
필수 GUI 의존성을 불러오지 못했습니다.

{qt_diagnostic_text(error)}

Windows에서 "DLL load failed while importing QtCore"가 나오면 보통 다음 중 하나입니다.
1. PySide6 최신 버전이 PC의 Windows/Python 조합과 맞지 않음
2. 기존 site-packages에 서로 다른 PySide6/shiboken6 버전이 섞임
3. PATH에 다른 Qt DLL이 먼저 잡힘
4. Microsoft Visual C++ Redistributable이 없거나 오래됨

권장 복구 절차:
  powershell -ExecutionPolicy Bypass -File .\\install_windows.ps1

수동 복구:
  python -m pip uninstall -y PySide6 PySide6_Addons PySide6_Essentials shiboken6
  python -m pip install --no-cache-dir --force-reinstall -r requirements.txt
  python tools\\diagnose_qt.py

그래도 실패하면 Microsoft Visual C++ Redistributable 2015-2022 x64를 설치한 뒤 다시 실행하세요.
다운로드: https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist
""".strip()
