import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_REPO = "hv-123/QA-Engine"
MODEL_REVISION = "7fda22dce2a8855a49e55e7016c1547dc3d0cd94"
ROOT = Path(__file__).resolve().parent
REQUIRED = [
    "models_fp16/extractive/model.safetensors",
    "models_fp16/extractive/model_config.json",
    "models_fp16/extractive/tokenizer.json",
    "models_fp16/generative/best.pt",
    "models_fp16/generative/tokenizer.json",
]


def main():
    parser = argparse.ArgumentParser(description="Download the published QA inference checkpoints at a fixed revision")
    parser.add_argument("--check", action="store_true", help="Check local files without accessing the network")
    args = parser.parse_args()
    if not args.check:
        snapshot_download(
            repo_id=MODEL_REPO, repo_type="space", revision=MODEL_REVISION,
            allow_patterns=["models_fp16/extractive/*", "models_fp16/generative/*"],
            local_dir=str(ROOT), token=False,
        )
    missing = [name for name in REQUIRED if not (ROOT / name).is_file() or (ROOT / name).stat().st_size == 0]
    if missing:
        raise SystemExit("Missing checkpoint files: " + ", ".join(missing))
    print(f"Checkpoints ready: {MODEL_REPO}@{MODEL_REVISION}")


if __name__ == "__main__":
    main()
