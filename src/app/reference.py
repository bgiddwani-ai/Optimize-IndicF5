"""Pinned IndicF5 FP32 runtime with request-scoped conditioning memoization."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from pydub import AudioSegment, silence
from safetensors.torch import load_file

from f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder, preprocess_ref_audio_text
from f5_tts.model import DiT
from f5_tts.model.utils import convert_char_to_pinyin
from app.workload import REFERENCE_TEXT, TARGETS, seed_for
from app.config import settings

REVISION = "fb652a96e508091045890dae5bb99fe511facbcf"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

class Runtime:
    def __init__(self):
        assert sys.version_info[:2] == (3, 10), sys.version
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        torch.set_num_threads(settings()["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.snapshot = Path(snapshot_download("ai4bharat/IndicF5", revision=REVISION,
            allow_patterns=["model.safetensors", "config.json", "checkpoints/vocab.txt", "prompts/*"]))
        self.model = load_model(DiT, dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            vocab_file=str(self.snapshot / "checkpoints/vocab.txt"), device="cuda")
        vocos_revision = "0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21"
        vocos_path = snapshot_download("charactr/vocos-mel-24khz", revision=vocos_revision,
            allow_patterns=["config.yaml", "pytorch_model.bin"])
        self.vocoder = load_vocoder(is_local=True, local_path=vocos_path, device="cuda").eval()
        state = load_file(str(self.snapshot / "model.safetensors"), device="cpu")
        model_state, vocoder_state = {}, {}
        for key, value in state.items():
            clean = key.replace("._orig_mod.", ".")
            if clean.startswith("ema_model."):
                model_state[clean[len("ema_model."):]] = value
            elif clean.startswith("vocoder."):
                vocoder_state[clean[len("vocoder."):]] = value
            else:
                raise RuntimeError(f"Unrecognized checkpoint key: {key}")
        self.model.load_state_dict(model_state, strict=True)
        assert model_state and vocoder_state
        self.vocoder.load_state_dict(vocoder_state, strict=True)
        del state, model_state, vocoder_state
        self.model = self.model.float().eval()
        self.vocoder = self.vocoder.float().eval()
        self.ref_path, self.ref_text = preprocess_ref_audio_text(str(self.snapshot / "prompts/PAN_F_HAPPY_00001.wav"), REFERENCE_TEXT)
        self.preflight = self.check_tokens(TARGETS)
        self.metadata = {"model": "ai4bharat/IndicF5", "revision": REVISION,
            "checkpoint_sha256": sha256(self.snapshot / "model.safetensors"),
            "vocab_sha256": sha256(self.snapshot / "checkpoints/vocab.txt"),
            "vocos_config_revision": vocos_revision,
            "reference_sha256": sha256(self.snapshot / "prompts/PAN_F_HAPPY_00001.wav"),
            "gpu_name": torch.cuda.get_device_name(), "gpu_capability": list(torch.cuda.get_device_capability()),
            "python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "nfe": settings()["nfe"], "cfg": 2.0, "sway": -1.0, "speed": 1.0,
            "parameter_dtype": "float32", "compute_dtype": "float32", "tf32": False,
            "tokenizer": "IndicF5 upstream custom character/pinyin", "model_boundary": "serialized",
            "streaming": False, "preflight": self.preflight}

    def check_tokens(self, texts):
        prefix = self.ref_text + (" " if len(self.ref_text[-1].encode()) == 1 else "")
        rows = []
        for text in texts:
            tokens = convert_char_to_pinyin([prefix + text])[0]
            missing = sorted(set(t for t in tokens if t not in self.model.vocab_char_map))
            rows.append({"text": text, "token_count": len(tokens), "oov": missing})
        assert all(not row["oov"] for row in rows), rows
        return {"passed": True, "rows": rows}

    @contextlib.contextmanager
    def prepared(self, enabled):
        if not enabled:
            yield
            return
        dit = self.model.transformer
        text_original = dit.text_embed.forward
        rope_original = dit.rotary_embed.forward_from_seq_len
        text_cache, rope_cache = {}, {}
        def text_forward(text, seq_len, drop_text=False):
            key = (text.data_ptr(), tuple(text.shape), seq_len, drop_text)
            if key not in text_cache:
                text_cache[key] = text_original(text, seq_len, drop_text=drop_text)
            return text_cache[key]
        def rope_forward(seq_len, *args, **kwargs):
            key = (seq_len, args, tuple(sorted(kwargs.items())))
            if key not in rope_cache:
                rope_cache[key] = rope_original(seq_len, *args, **kwargs)
            return rope_cache[key]
        dit.text_embed.forward = text_forward
        dit.rotary_embed.forward_from_seq_len = rope_forward
        try:
            yield
        finally:
            dit.text_embed.forward = text_original
            dit.rotary_embed.forward_from_seq_len = rope_original

    def synthesize(self, text, backend="prepared"):
        assert backend in ("eager", "prepared")
        self.check_tokens([text])
        torch.manual_seed(seed_for(text))
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode(), self.prepared(backend == "prepared"):
            audio, sr, _ = infer_process(self.ref_path, self.ref_text, text, self.model, self.vocoder,
                nfe_step=settings()["nfe"], cfg_strength=2.0, sway_sampling_coef=-1.0, speed=1.0,
                device="cuda", show_info=lambda *a: None)
        torch.cuda.synchronize()
        # Match the official HF wrapper's PCM conversion, silence removal and -20 dBFS gain.
        assert audio.size and np.isfinite(audio).all()
        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        segment = AudioSegment.from_file(buffer, format="wav")
        parts = silence.split_on_silence(segment, min_silence_len=1000, silence_thresh=-50, keep_silence=500, seek_step=10)
        segment = sum(parts, AudioSegment.silent(duration=0))
        assert len(segment) > 0 and np.isfinite(segment.dBFS)
        segment = segment.apply_gain(-20.0 - segment.dBFS)
        result = np.asarray(segment.get_array_of_samples(), dtype=np.float32) / 32768.0
        assert np.isfinite(result).all() and np.sqrt(np.mean(result**2)) >= 0.001
        out = io.BytesIO()
        sf.write(out, result, sr, format="WAV", subtype="PCM_16")
        return out.getvalue(), {"process_s": time.perf_counter() - start, "audio_s": len(result) / sr}
