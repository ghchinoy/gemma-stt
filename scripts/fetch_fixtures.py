#!/usr/bin/env python3
"""Download/regenerate test fixture audio for gemma-stt.

Fixture .wav files are NOT committed to git (see .gitignore) to keep the
repo lean -- only the manifests (with ground truth + source citations) and
this script are. Run this to (re)create the actual audio before running
tests, the domain showcase, or a smoke check.

Subcommands:
  domains   Fetch the legal/medical/financial clips in
            tests/fixtures/domains/ via direct HTTP download + ffmpeg
            slicing. No extra Python dependencies -- just needs `ffmpeg`
            on PATH.
  minds14   Fetch the 5 PolyAI/MInDS-14 clips in tests/fixtures/ used for
            the E2B-vs-E4B comparison in docs/FINDINGS.md. Requires the
            `datasets` package (install with:
            `uv pip install -e '.[fixtures]'`).
  all       Both of the above.

Usage:
    .venv/bin/python scripts/fetch_fixtures.py domains
    .venv/bin/python scripts/fetch_fixtures.py minds14
    .venv/bin/python scripts/fetch_fixtures.py all [--force]
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent.parent / "tests" / "fixtures"
DOMAINS_DIR = FIXTURES_DIR / "domains"


def _require_tools():
    missing = [t for t in ("ffmpeg", "curl") if shutil.which(t) is None]
    if missing:
        print(
            f"Error: required tool(s) not found on PATH: {', '.join(missing)}. "
            "Install with e.g. `brew install ffmpeg` (curl ships with macOS).",
            file=sys.stderr,
        )
        sys.exit(1)


def _download(url: str, dest: Path):
    print(f"  downloading {url}")
    # Uses curl (system trust store) rather than Python's urllib, which can
    # fail with CERTIFICATE_VERIFY_FAILED on some python.org framework
    # builds that don't bundle/find a CA cert path.
    subprocess.run(
        ["curl", "-sL", "-f", url, "-o", str(dest)],
        check=True,
    )


def fetch_domains(force: bool = False):
    _require_tools()
    manifest = json.loads((DOMAINS_DIR / "manifest.json").read_text())

    # Cache each unique source recording once, slice per clip, then discard.
    with tempfile.TemporaryDirectory(prefix="gemma-stt-fixtures-") as tmp:
        tmp_path = Path(tmp)
        cache: dict[str, Path] = {}

        for domain, info in manifest.items():
            print(f"=== {domain} ===")
            for clip in info["clips"]:
                out_path = DOMAINS_DIR / clip["file"]
                if out_path.exists() and not force:
                    print(f"  {clip['file']} already exists, skipping (use --force to re-fetch)")
                    continue

                url = clip["audio_url"]
                if url not in cache:
                    local_name = tmp_path / (str(len(cache)) + Path(url).suffix or ".mp3")
                    _download(url, local_name)
                    cache[url] = local_name
                source_file = cache[url]

                out_path.parent.mkdir(parents=True, exist_ok=True)
                print(f"  slicing {clip['file']} [{clip['start']}s - {clip['end']}s]")
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        str(source_file),
                        "-ss",
                        str(clip["start"]),
                        "-to",
                        str(clip["end"]),
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-loglevel",
                        "error",
                        str(out_path),
                    ],
                    check=True,
                )
    print("Done.")


def fetch_minds14(force: bool = False):
    try:
        from datasets import Audio, load_dataset
        import soundfile as sf
    except ImportError:
        print(
            "Error: the `datasets` package is required for this. Install with:\n"
            "  uv pip install -e '.[fixtures]'",
            file=sys.stderr,
        )
        sys.exit(1)
    import io

    manifest = json.loads((FIXTURES_DIR / "manifest.json").read_text())
    src = manifest["source"]

    missing = [
        s
        for s in manifest["samples"]
        if force or not (FIXTURES_DIR / s["file"]).exists()
    ]
    if not missing:
        print("All MInDS-14 fixtures already present, skipping (use --force to re-fetch).")
        return

    print(f"Loading {src['hf_dataset']} ({src['hf_config']}, {src['hf_split']}) ...")
    ds = load_dataset(src["hf_dataset"], src["hf_config"], split=src["hf_split"])
    ds = ds.cast_column("audio", Audio(decode=False))

    for sample in missing:
        row = ds[sample["hf_index"]]
        audio_bytes = row["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(audio_bytes))
        out_path = FIXTURES_DIR / sample["file"]
        sf.write(out_path, data, sr, subtype="PCM_16")
        print(f"  wrote {sample['file']} ({len(data) / sr:.1f}s @ {sr}Hz)")

        # Sanity check against the committed ground truth -- catches dataset
        # version drift.
        if row["transcription"] != sample["ground_truth"]:
            print(
                f"  WARNING: transcription for {sample['file']} does not match "
                f"the manifest's ground_truth. Dataset may have changed since "
                f"this fixture was created."
            )
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("target", choices=["domains", "minds14", "all"])
    parser.add_argument("--force", action="store_true", help="Re-fetch even if the file already exists")
    args = parser.parse_args()

    if args.target in ("domains", "all"):
        fetch_domains(force=args.force)
    if args.target in ("minds14", "all"):
        fetch_minds14(force=args.force)


if __name__ == "__main__":
    main()
