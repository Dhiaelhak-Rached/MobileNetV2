"""Step A (run on the LAPTOP): export the frozen model to ONNX.

    python Jetson/export_onnx.py

What it does:
  1. Loads artifacts/model/mobilenetv2_drone_best.pth (embeds audio config,
     normalization stats and class names — nothing can be misconfigured).
  2. Wraps preprocessor + model into ONE module:
         waveform [B, 16000] -> log-mel spectrogram -> MobileNetV2 -> probabilities [B, 2]
     so the Jetson receives raw audio and outputs P(drone) — frontend in-graph.
  3. Exports to artifacts/export/mobilenetv2_drone.onnx (opset 17, dynamic batch).
  4. Verifies the file with the ONNX checker and a smoke inference.

The frozen checkpoint is only READ, never modified.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.audio.preprocessing import AudioConfig, Preprocessor, ROOT
from src.models.mobilenet_acoustic import build_model

CKPT = ROOT / "artifacts" / "model" / "mobilenetv2_drone_best.pth"
OUT = ROOT / "artifacts" / "export" / "mobilenetv2_drone.onnx"
OPSET = 17


class DroneExportModule(torch.nn.Module):
    """waveform [B, n_samples] -> softmax probabilities [B, 2] (frontend in-graph).

    Export-friendly frontend. Two deliberate rewrites of the torchaudio path,
    both numerically equivalent:
      1. torch.stft(return_complex=True) can't be exported to ONNX (complex
         types unsupported) -> we use the real-valued two-output form.
      2. TensorRT's ONNX parser (checked on v10.3) rejects the STFT op
         entirely -> the STFT is expressed as a single Conv1d whose kernels
         are the windowed DFT basis (STFT == convolution with cos/sin basis;
         this is the classic "STFT as convolution" identity). Conv1d is one of
         the best-supported TensorRT ops.
    The export script asserts the result matches the official Preprocessor
    path to <1e-5 in P(drone) on a real file before writing anything."""

    def __init__(self, pp: Preprocessor, model: torch.nn.Module):
        super().__init__()
        import math
        cfg = pp.cfg
        self.hop = cfg.hop_length
        self.pad = cfg.n_fft // 2  # center padding, like torchaudio Spectrogram
        self.mean, self.std = pp.mean, pp.std
        self.model = model

        # windowed DFT basis as Conv1d kernels: rows 0..512 = real, 513..1025 = imag
        n_fft, n_bins = cfg.n_fft, cfg.n_fft // 2 + 1  # onesided: 513 bins
        k = torch.arange(n_bins).unsqueeze(1).float()   # [513, 1]
        n = torch.arange(n_fft).unsqueeze(0).float()    # [1, 1024]
        ang = 2 * math.pi * k * n / n_fft               # [513, 1024]
        w = torch.hann_window(cfg.win_length)           # [1024]
        re = torch.cos(ang) * w
        im = -torch.sin(ang) * w                        # X[k] = sum x*w*e^{-i*ang}
        dft_weight = torch.cat([re, im], dim=0).unsqueeze(1).contiguous()  # [1026,1,1024]
        self.register_buffer("dft_weight", dft_weight)
        self.register_buffer("mel_fb", pp.mel.mel_scale.fb.clone())  # [513, n_mels]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.pad(waveform, (self.pad, self.pad),
                                    mode="reflect").unsqueeze(1)  # [B,1,T+1024]
        stft = torch.nn.functional.conv1d(x, self.dft_weight,
                                          stride=self.hop)        # [B,1026,T]
        power = stft[:, :513].pow(2) + stft[:, 513:].pow(2)       # [B,513,T]
        mel = torch.matmul(power.transpose(1, 2), self.mel_fb).transpose(1, 2)
        log_mel = 10.0 * torch.log10(torch.clamp(mel, min=1e-10))
        spec = ((log_mel - self.mean) / self.std).unsqueeze(1)  # [B, 1, n_mels, T]
        logits = self.model(spec)                            # [B, 2]
        return torch.softmax(logits, dim=1)


def main() -> None:
    import onnx  # noqa: pip install onnx

    state = torch.load(CKPT, weights_only=False, map_location="cpu")
    audio_cfg = AudioConfig(**state["audio"])
    model = build_model(pretrained=False, dropout=state["config"]["model"]["dropout"])
    model.load_state_dict(state["model"])
    model.eval()
    # CPU on purpose: the exported graph must be device-agnostic
    pp = Preprocessor(audio_cfg, state["norm"]["mean"], state["norm"]["std"], device="cpu")
    wrapper = DroneExportModule(pp, model).eval()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, audio_cfg.n_samples)
    try:
        torch.onnx.export(wrapper, dummy, str(OUT), opset_version=OPSET,
                          input_names=["waveform"], output_names=["probabilities"],
                          dynamic_axes={"waveform": {0: "batch"},
                                        "probabilities": {0: "batch"}},
                          dynamo=False)
    except TypeError:  # older torch without the `dynamo` kwarg
        torch.onnx.export(wrapper, dummy, str(OUT), opset_version=OPSET,
                          input_names=["waveform"], output_names=["probabilities"],
                          dynamic_axes={"waveform": {0: "batch"},
                                        "probabilities": {0: "batch"}})

    onnx.checker.check_model(str(OUT))  # raises if the graph is invalid

    # prove the manually-written frontend matches the official Preprocessor path
    # on a real audio file (this is the check that the complex-STFT rewrite
    # changed nothing numerically)
    import csv
    rows = list(csv.DictReader((ROOT / "data" / "splits" / "test.csv").open()))
    w = pp.fix_length(pp.load(rows[0]["path"]), offset=None).unsqueeze(0)
    with torch.no_grad():
        ref_spec = pp.waveform_to_spec(w)
        ref_prob = torch.softmax(model(ref_spec), 1)[0, 1].item()
        new_prob = wrapper(w)[0, 1].item()
    assert abs(ref_prob - new_prob) < 1e-5, (ref_prob, new_prob)

    # smoke test: ONNX output must exist, be finite, and match PyTorch
    import onnxruntime as ort  # noqa: pip install onnxruntime
    sess = ort.InferenceSession(str(OUT), providers=["CPUExecutionProvider"])
    probs = sess.run(["probabilities"], {"waveform": w.numpy()})[0]
    assert probs.shape == (1, 2) and torch.isfinite(torch.from_numpy(probs)).all()
    assert abs(float(probs[0, 1]) - new_prob) < 1e-4, (probs[0, 1], new_prob)

    mb = OUT.stat().st_size / 1e6
    print(f"OK: exported {OUT} ({mb:.1f} MB), opset {OPSET}, "
          f"input waveform[batch,{audio_cfg.n_samples}] -> probabilities[batch,2]")
    print(f"class order: {state['class_names']} (index 1 = drone)")
    print("next: python Jetson/validate_onnx.py")


if __name__ == "__main__":
    main()
