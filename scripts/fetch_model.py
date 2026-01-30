from __future__ import annotations

import os
import shutil
import tarfile
import zipfile
from pathlib import Path

import requests


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def _extract(archive_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    name = archive_path.name.lower()
    if name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            tf.extractall(out_dir)
        return

    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(out_dir)
        return

    raise ValueError("Unsupported archive format. Use .tar.gz, .tgz, or .zip")


def ensure_model_present() -> Path:
    """
    Ensures a model artefact exists under models/latest.

    Strategy:
    - If models/latest exists: do nothing.
    - Else, if MODEL_URL is set: download and extract.
    - The extracted content must include an MLflow sklearn model directory (MLmodel file).
    """
    model_dir = Path("models/latest")
    if model_dir.exists():
        print(f"Model already present at: {model_dir}")
        return model_dir

    model_url = os.getenv("MODEL_URL", "").strip()
    if not model_url:
        raise RuntimeError(
            "MODEL_URL is not set and models/latest does not exist. "
            "Provide MODEL_URL as an environment variable (public URL to the model artefact)."
        )

    tmp_dir = Path("/tmp/model_download")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    archive_path = tmp_dir / "model_artifact"
    # Keep the extension if present (helps extraction)
    if model_url.lower().endswith(".tar.gz"):
        archive_path = tmp_dir / "model_latest.tar.gz"
    elif model_url.lower().endswith(".tgz"):
        archive_path = tmp_dir / "model_latest.tgz"
    elif model_url.lower().endswith(".zip"):
        archive_path = tmp_dir / "model_latest.zip"

    print(f"Downloading model artefact from: {model_url}")
    _download(model_url, archive_path)

    extract_dir = tmp_dir / "extracted"
    _extract(archive_path, extract_dir)

    # The GitHub Actions artefact often contains model_latest.tar.gz inside the zip.
    # If we extracted a zip and we still have a tar.gz inside, handle it.
    inner_tars = list(extract_dir.rglob("*.tar.gz")) + list(extract_dir.rglob("*.tgz"))
    if inner_tars:
        inner = inner_tars[0]
        inner_dir = tmp_dir / "inner_extracted"
        _extract(inner, inner_dir)
        extract_dir = inner_dir

    # Find the MLflow model directory (contains MLmodel)
    candidates = [p.parent for p in extract_dir.rglob("MLmodel")]
    if not candidates:
        raise RuntimeError("Could not find an MLflow model directory (MLmodel file not found).")

    source_model_dir = candidates[0]
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    if model_dir.exists():
        shutil.rmtree(model_dir)
    shutil.copytree(source_model_dir, model_dir)

    print(f"Model extracted to: {model_dir}")
    return model_dir


if __name__ == "__main__":
    ensure_model_present()
