"""Paired heldout flow-MSE evaluation with identical noise and fixed timesteps."""

import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent
os.environ["HF_HOME"] = str(ROOT / ".hf-cache")
os.environ["HF_MODULES_CACHE"] = str(ROOT / ".hf-cache/modules")

import torch
from peft import PeftModel

from acestep.training_v2.model_loader import load_decoder_for_training


def score(decoder, paths):
    """Measure conditional flow error; no CFG dropout or stochastic training mode."""
    decoder.eval()
    rows = []
    for index, path in enumerate(paths):
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        batch = {k: tensor[k].unsqueeze(0).to("cuda", dtype=torch.bfloat16) for k in (
            "target_latents", "attention_mask", "encoder_hidden_states",
            "encoder_attention_mask", "context_latents")}
        x0 = batch.pop("target_latents")
        losses = []
        for noise_seed in (20261117, 20261118):
            torch.manual_seed(noise_seed + index)
            noise = torch.randn_like(x0)
            for value in (0.2, 0.5, 0.8):
                t = torch.tensor([value], device="cuda", dtype=torch.bfloat16)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    prediction = decoder(hidden_states=value * noise + (1-value) * x0,
                                         timestep=t, timestep_r=t, **batch)[0]
                loss = (prediction.float() - (noise - x0).float()).square().mean().item()
                if not torch.isfinite(torch.tensor(loss)):
                    raise ValueError("Non-finite evaluation loss")
                losses.append(loss)
        rows.append({"clip": path.stem, "mse": sum(losses) / len(losses), "trials": losses})
    return {"mean_mse": sum(row["mse"] for row in rows) / len(rows), "clips": rows}


def main():
    """Evaluate original, earlier pilot and new adapter on the same excluded clips."""
    paths = sorted((DATA / "tensors/heldout").glob("*.pt"))
    if not paths:
        raise RuntimeError("Preprocess heldout tensors first")
    report = {"method": "Conditional flow MSE; 2 fixed noises x 3 fixed timesteps per clip.",
              "limitation": "Proxy metric, not perceptual quality or proof of calmness.",
              "heldout_count": len(paths), "results": {}}
    adapters = {
        "original": None,
        "pilot": ROOT / "training/piano_pilot/lora/final",
        "expanded": DATA / "lora/final",
    }
    for label, adapter in adapters.items():
        model = load_decoder_for_training(ROOT / "checkpoints", "turbo", "cpu", "bf16")
        model.decoder.to("cuda")
        if adapter is not None:
            model.decoder = PeftModel.from_pretrained(model.decoder, str(adapter))
        report["results"][label] = score(model.decoder, paths)
        print(label, report["results"][label]["mean_mse"], flush=True)
        (DATA / "evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
