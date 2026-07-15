#!/usr/bin/env python3
"""Ad-hoc script (not part of the shipped CLI) used to generate the
comparison data in docs/DOMAIN_SHOWCASE.md. Runs each domain clip through
gemma-stt twice: once with the CLI's generic default prompt, once with a
domain-informed prompt, so the two can be compared side by side.

Usage: run from the repo root with the project venv active:
    .venv/bin/python tests/run_domain_showcase.py --model e4b
"""

import argparse
import json
from pathlib import Path

from gemma_stt.models import resolve_model
from gemma_stt.transcribe import DEFAULT_PROMPT, GemmaSTT

DOMAIN_PROMPTS = {
    "legal": (
        "Transcribe the following audio verbatim. This is a recording of a "
        "U.S. Supreme Court oral argument. Use correct legal terminology, "
        "case names, and citations."
    ),
    "medical": (
        "Transcribe the following audio verbatim. This is a recording of a "
        "doctor-patient medical consultation. Use correct clinical, "
        "anatomical, and pharmaceutical terminology, including exact drug "
        "names."
    ),
    "financial": (
        "Transcribe the following audio verbatim. This is a recording "
        "discussing financial regulation and government appropriations. "
        "Use correct financial, legal, and monetary terminology, and "
        "transcribe dollar figures exactly."
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="e4b")
    parser.add_argument(
        "--fixtures-dir",
        default=str(Path(__file__).parent / "fixtures" / "domains"),
    )
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    fixtures_dir = Path(args.fixtures_dir)
    manifest = json.loads((fixtures_dir / "manifest.json").read_text())

    resolved = resolve_model(args.model)
    print(f"Loading {resolved.alias} from {resolved.path} ...")
    engine = GemmaSTT(resolved.path, resolved.alias).load()
    print(f"Loaded in {engine.load_seconds:.1f}s\n")

    results = []
    for domain, info in manifest.items():
        domain_prompt = DOMAIN_PROMPTS[domain]
        for clip in info["clips"]:
            audio_path = str(fixtures_dir / clip["file"])
            print(f"=== {domain}: {clip['file']} ===")

            generic = engine.transcribe(
                audio_path, prompt=DEFAULT_PROMPT, max_tokens=args.max_tokens
            )
            domain_specific = engine.transcribe(
                audio_path, prompt=domain_prompt, max_tokens=args.max_tokens
            )

            print(f"  ground truth : {clip['ground_truth']}")
            print(f"  generic      : {generic.text}")
            print(f"  domain-aware : {domain_specific.text}")
            print()

            results.append(
                {
                    "domain": domain,
                    "file": clip["file"],
                    "ground_truth": clip["ground_truth"],
                    "generic_prompt": DEFAULT_PROMPT,
                    "generic_output": generic.text,
                    "domain_prompt": domain_prompt,
                    "domain_output": domain_specific.text,
                }
            )

    out_path = fixtures_dir / f"showcase_results_{resolved.alias}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
