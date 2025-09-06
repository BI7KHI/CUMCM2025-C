import os
import sys
import subprocess
import shutil
import urllib.request
from typing import List, Optional


def run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def has_cmd(name: str) -> bool:
    return shutil.which(name) is not None


def project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_fonts_dir() -> str:
    fonts_dir = os.path.join(project_root(), "fonts")
    os.makedirs(fonts_dir, exist_ok=True)
    return fonts_dir


def install_cjk_fonts_systemwide() -> bool:
    """
    Attempt to install CJK fonts using the system package manager (non-interactive).
    Returns True if a package manager command was executed and succeeded.
    """
    try:
        # Prefer Noto and WenQuanYi families
        if has_cmd("apt-get"):
            cmd = [
                "sudo", "apt-get", "update"
            ] if has_cmd("sudo") else ["apt-get", "update"]
            run(cmd)
            install_cmd = (
                ["sudo"] if has_cmd("sudo") else []
            ) + [
                "apt-get", "install", "-y",
                "fonts-noto-cjk",
                "fonts-noto-color-emoji",
                "fonts-wqy-zenhei",
                "fonts-wqy-microhei",
            ]
            result = run(install_cmd)
            return result.returncode == 0

        if has_cmd("dnf"):
            install_cmd = (
                ["sudo"] if has_cmd("sudo") else []
            ) + [
                "dnf", "install", "-y",
                "google-noto-sans-cjk-ttc",
                "google-noto-serif-cjk-ttc",
                "wqy-zenhei-fonts",
            ]
            result = run(install_cmd)
            return result.returncode == 0

        if has_cmd("yum"):
            install_cmd = (
                ["sudo"] if has_cmd("sudo") else []
            ) + [
                "yum", "install", "-y",
                "google-noto-sans-cjk-ttc",
                "google-noto-serif-cjk-ttc",
                "wqy-zenhei-fonts",
            ]
            result = run(install_cmd)
            return result.returncode == 0

        if has_cmd("pacman"):
            install_cmd = (
                ["sudo"] if has_cmd("sudo") else []
            ) + [
                "pacman", "-S", "--noconfirm",
                "noto-fonts-cjk",
                "wqy-zenhei",
            ]
            result = run(install_cmd)
            return result.returncode == 0

        if has_cmd("zypper"):
            install_cmd = (
                ["sudo"] if has_cmd("sudo") else []
            ) + [
                "zypper", "--non-interactive", "install",
                "fonts-noto-cjk",
            ]
            result = run(install_cmd)
            return result.returncode == 0
    except Exception:
        return False

    return False


def download_with_fallback(urls: List[str], dest_path: str) -> Optional[str]:
    for url in urls:
        try:
            print(f"Downloading: {url}")
            urllib.request.urlretrieve(url, dest_path)
            return dest_path
        except Exception as e:
            print(f"Failed: {e}")
    return None


def install_local_cjk_font(fonts_dir: str) -> Optional[str]:
    """
    Download a single reliable CJK font locally into fonts/.
    Returns the local font file path if successful.
    """
    candidate_urls = [
        # Primary: Google Noto CJK (SC) regular
        "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf",
        # Alternative path naming used in some branches
        "https://raw.githubusercontent.com/googlefonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansSC-Regular.otf",
    ]
    target_file = os.path.join(fonts_dir, "NotoSansCJKsc-Regular.otf")
    return download_with_fallback(candidate_urls, target_file)


def refresh_font_cache():
    # Rebuild fontconfig cache (user-level)
    if has_cmd("fc-cache"):
        run(["fc-cache", "-f", "-v"])  # ignore result; best-effort
    # Rebuild matplotlib cache
    try:
        from matplotlib import font_manager as fm
        fm._load_fontmanager(try_read_cache=False)
    except Exception:
        pass


def verify_cjk_available() -> bool:
    try:
        from matplotlib import font_manager as fm
        families = {
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "Source Han Sans CN",
            "SimHei",
            "WenQuanYi Zen Hei",
            "Microsoft YaHei",
            "PingFang SC",
        }
        installed = set(f.name for f in fm.fontManager.ttflist)
        return any(name in installed for name in families)
    except Exception:
        return False


def main() -> int:
    print("=== Chinese Font Setup ===")
    fonts_dir = ensure_fonts_dir()

    print("1) Trying system package installation (non-interactive)...")
    system_ok = install_cjk_fonts_systemwide()
    if system_ok:
        print("System-wide CJK fonts installed.")
    else:
        print("System install not available or failed. Falling back to local download...")

    print("2) Ensuring at least one local CJK font exists under ./fonts ...")
    local_paths = [f for f in os.listdir(fonts_dir) if f.lower().endswith((".ttf", ".otf"))]
    if not local_paths:
        downloaded = install_local_cjk_font(fonts_dir)
        if downloaded:
            print(f"Downloaded: {downloaded}")
        else:
            print("Failed to download a local font. You may need to install fonts manually.")

    print("3) Refreshing font caches ...")
    refresh_font_cache()

    print("4) Verifying availability ...")
    ok = verify_cjk_available()
    if ok:
        print("Success: CJK fonts detected. Matplotlib should render Chinese correctly.")
        print("Tip: Our scripts auto-load fonts from ./fonts if present.")
        return 0
    else:
        print("Warning: No CJK fonts detected by Matplotlib.")
        print("You can place .ttf/.otf files in the ./fonts directory and rerun.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


