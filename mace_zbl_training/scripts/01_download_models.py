#!/usr/bin/env python3
"""Download MACE-MH-1 and ensure MH-0 symlink. Idempotent."""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parents[1]
PRE = WS / "pretrained"

URLS = {
    "mace-mh-0.model": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model",
    "mace-mh-1.model": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model",
}

MH0_SHARED = WS.parent / "gfnff_delta" / "pretrained" / "mace-mh-0.model"


def ensure(name: str, url: str) -> Path:
    target = PRE / name
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"  ok      {name:18s} {target.stat().st_size/1e6:6.1f} MB  ({'symlink' if target.is_symlink() else 'file'})")
        return target

    if name == "mace-mh-0.model" and MH0_SHARED.exists():
        rel = os.path.relpath(MH0_SHARED, PRE)
        if target.is_symlink() or target.exists():
            target.unlink()
        os.symlink(rel, target)
        print(f"  link    {name:18s} -> {rel}")
        return target

    print(f"  fetch   {name}  <- {url}")
    PRE.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    print(f"          {target.stat().st_size/1e6:6.1f} MB")
    return target


def main() -> None:
    PRE.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        ensure(name, url)


if __name__ == "__main__":
    main()
