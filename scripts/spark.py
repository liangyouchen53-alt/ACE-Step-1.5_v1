"""Portable DGX Spark launch and LoRA training entry points."""

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def check():
    """Require ARM64 Linux and execute a CUDA operation before loading models."""
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError("此入口需要 Linux ARM64 的 DGX Spark 主機。")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 無法使用，請確認 Spark 驅動及 CUDA 版 PyTorch。")
    x = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
    (x @ x).sum().item()
    torch.cuda.synchronize()
    print(f"GPU: {torch.cuda.get_device_name(0)}；PyTorch {torch.__version__}；CUDA {torch.version.cuda}")


def run(arguments):
    """Run the project's interpreter and propagate command failures."""
    subprocess.run([sys.executable, *arguments], cwd=ROOT, check=True)


def normalize_manifests(data):
    """Convert legacy Windows relative separators for training on Linux."""
    for split in ("train", "heldout"):
        source = data / f"{split}.json"
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
        for sample in payload["samples"]:
            relative = sample["audio_path"].replace("\\", "/")
            if ":" in relative or relative.startswith("/"):
                raise ValueError(f"{source} 包含絕對路徑；請重新執行 prepare.py。")
            path = (data / relative).resolve()
            if not path.is_relative_to(data) or not path.is_file():
                raise ValueError(f"缺少音訊或路徑超出資料集：{path}")
            sample["audio_path"] = relative
        source.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    """Configure local caches and dispatch web, preprocessing or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "web", "preprocess", "train", "resume"])
    parser.add_argument("--dataset", default="singing_bowl_manual")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--resume-from")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault("HF_HOME", str(ROOT / ".hf-cache"))
    os.environ.setdefault("HF_MODULES_CACHE", str(ROOT / ".hf-cache/modules"))
    os.environ["ACESTEP_GENERATION_ONLY"] = "true"
    os.environ["ACESTEP_COMPILE_MODEL"] = "false"
    os.environ["PYTHONUTF8"] = "1"
    check()
    if args.action == "check":
        return
    if args.action == "web":
        run(["-m", "acestep.acestep_v15_pipeline", "--server-name", args.host,
             "--port", str(args.port), "--language", "zh-TW", "--batch_size", "1",
             "--config_path", "acestep-v15-turbo", "--init_service", "true",
             "--init_llm", "true", "--lm_model_path", "acestep-5Hz-lm-1.7B",
             "--backend", "pt", "--offload_to_cpu", "true", "--offload_dit_to_cpu", "true",
             "--quantization", "none", "--use_flash_attention", "false"])
        return
    data = (ROOT / "training" / args.dataset).resolve()
    if not data.is_relative_to(ROOT / "training") or not data.is_dir():
        raise ValueError("資料集必須是 training/ 內的現有目錄。")
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("請先在網站終端按 Ctrl+C，釋放模型記憶體後再訓練。")
    common = ["train.py", "--yes", "--plain", "fixed", "--checkpoint-dir", "checkpoints",
              "--model-variant", "turbo", "--device", "cuda", "--precision", "bf16",
              "--output-dir", str(data / "lora")]
    if args.action == "preprocess":
        normalize_manifests(data)
        for split in ("train", "heldout"):
            run(common + ["--dataset-dir", str(data / "tensors" / split), "--preprocess",
                          "--dataset-json", str(data / f"{split}.json"),
                          "--audio-dir", str(data / "clips" / split),
                          "--tensor-output", str(data / "tensors" / split), "--max-duration", "20"])
        return
    if args.epochs < 1:
        raise ValueError("epochs 必須大於 0。")
    if args.action == "train" and (data / "lora").exists():
        raise ValueError("lora 已存在，請使用 resume 或先手動封存。")
    run([str(data / "verify.py")])
    extra = []
    if args.action == "resume":
        if not args.resume_from or not (Path(args.resume_from) / "training_state.pt").is_file():
            raise ValueError("請指定含 training_state.pt 的 --resume-from。")
        extra = ["--resume-from", args.resume_from]
    run(common + ["--dataset-dir", str(data / "tensors/train"), "--rank", "8", "--alpha", "16",
                  "--batch-size", "1", "--gradient-accumulation", "4", "--epochs", str(args.epochs),
                  "--lr", "0.00005", "--warmup-steps", "2", "--optimizer-type", "adafactor",
                  "--gradient-checkpointing", "--offload-encoder", "--num-workers", "0",
                  "--save-every", "1", "--log-every", "1", "--sample-every-n-epochs", "0"] + extra)


if __name__ == "__main__":
    main()
