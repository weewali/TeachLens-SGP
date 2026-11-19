"""Resolve project paths without depending on the caller's working directory."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value: str | Path) -> Path:
    """Return an absolute project-relative path, preserving absolute inputs."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def file_in_use(path: Path) -> bool:
    """True if another program holds path open so it cannot be replaced.

    Windows refuses to replace a file that a program (for example a video
    player showing a review MP4) opened without delete sharing. Asking for
    delete access with full sharing fails exactly then. Always False on
    other systems and for missing files.
    """
    if os.name != "nt" or not path.exists():
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                     wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    delete_access, share_all, open_existing, normal = 0x00010000, 0x7, 3, 0x80
    handle = kernel32.CreateFileW(str(path), delete_access, share_all, None, open_existing, normal, None)
    if handle is None or handle == wintypes.HANDLE(-1).value:
        return ctypes.get_last_error() == 32  # ERROR_SHARING_VIOLATION
    kernel32.CloseHandle(handle)
    return False
