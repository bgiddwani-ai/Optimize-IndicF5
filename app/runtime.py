"""FP32 control and matched prepared BF16 eager/AOTI inference; no fallback."""
import contextlib, json
from pathlib import Path
import torch
from app.reference import Runtime as ReferenceRuntime, sha256
from app.step import BF16Step
from app.config import settings

BACKENDS = ('fp32_eager', 'bf16_eager', 'bf16_aoti')


class Runtime(ReferenceRuntime):
    def __init__(self):
        super().__init__()
        assert self.metadata['checkpoint_sha256'] == 'ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c'
        self.step = BF16Step(self.model.transformer).eval()
        self.aoti = None
        built = {}
        if Path('artifacts/dit_bf16.pt2').exists():
            built = json.loads(Path('evidence/aoti-build.json').read_text())
            assert sha256('artifacts/dit_bf16.pt2') == built['package_sha256']
            assert sha256('app/step.py') == built['source_sha256']
            assert built['torch'] == torch.__version__
            assert built['gpu_capability'] == list(torch.cuda.get_device_capability()), 'Rebuild on this GPU architecture.'
            self.aoti = torch._inductor.aoti_load_package('artifacts/dit_bf16.pt2')
        self.active_backend = None
        self.aoti_calls = 0
        self.observed_frames = set()
        self.completed = {b: 0 for b in BACKENDS}
        self.metadata.update(backends=list(BACKENDS),
            compute_dtype='BF16 autocast DiT; FP32 weights, mel frontend, sampler state and Vocos',
            bf16_eager_optimizations='request-scoped text/RoPE/zero-condition preparation; same graph as AOTI',
            aoti_package_sha256=built.get('package_sha256'), aoti_compile_s=built.get('compile_s'),
            dynamic_frames=[128,settings()['max_frames']], fallback=False, aoti_format=built.get('format'))

    def stats(self):
        return {'compiled_dit_calls': self.aoti_calls, 'observed_frames': sorted(self.observed_frames),
                'completed_requests': dict(self.completed)}

    @contextlib.contextmanager
    def prepared(self, enabled):
        if not enabled:
            yield
            return
        assert self.active_backend in ('bf16_eager', 'bf16_aoti')
        dit = self.model.transformer
        original = dit.forward
        sample_original = self.model.sample
        mel_original = self.model.mel_spec.forward
        texts, ropes, zeros, times, held_tokens, held_times = {}, {}, {}, {}, [], []

        def mel_fp32(*args, **kwargs):
            with torch.autocast('cuda', enabled=False):
                return mel_original(*args, **kwargs)

        def sample_bf16(*args, **kwargs):
            # Keep the autocast weight cache alive throughout all 64 DiT calls.
            # The mel frontend has an explicit FP32 boundary; Vocos runs after
            # sample returns, outside this autocast scope.
            with torch.autocast('cuda', dtype=torch.bfloat16):
                return sample_original(*args, **kwargs)

        def forward(x, cond, text, time, drop_audio_cond, drop_text, mask=None):
            assert mask is None and x.shape[0] == 1 and 128 <= x.shape[1] <= settings()['max_frames']
            length = x.shape[1]
            self.observed_frames.add(length)
            key = (text.data_ptr(), tuple(text.shape), length, bool(drop_text))
            if key not in texts:
                held_tokens.append(text)
                texts[key] = dit.text_embed(text, length, drop_text=drop_text).to(torch.bfloat16).contiguous()
            if length not in ropes:
                freq, scale = dit.rotary_embed.forward_from_seq_len(length)
                assert scale == 1.0
                ropes[length] = freq.contiguous()
            if drop_audio_cond:
                if length not in zeros:
                    zeros[length] = torch.zeros_like(cond)
                cond = zeros[length]
            timestep = time.repeat(1) if time.ndim == 0 else time
            time_key = (time.data_ptr(), tuple(time.shape))
            if time_key not in times:
                held_times.append(time)
                times[time_key] = dit.time_embed(timestep).contiguous()
            args = (x.contiguous(), cond.contiguous(), texts[key], times[time_key], ropes[length])
            if self.active_backend == 'bf16_eager':
                return self.step(*args)
            assert self.aoti is not None, 'Compile the AOTI package before using bf16_aoti.'
            result = self.aoti(*args)
            self.aoti_calls += 1
            assert result.dtype == torch.bfloat16 and result.is_cuda
            return result

        dit.forward = forward
        self.model.sample = sample_bf16
        self.model.mel_spec.forward = mel_fp32
        try:
            yield
        finally:
            dit.forward = original
            self.model.sample = sample_original
            self.model.mel_spec.forward = mel_original

    def synthesize(self, text, backend='bf16_aoti'):
        assert backend in BACKENDS
        self.active_backend = backend
        before = self.aoti_calls
        try:
            body, metrics = super().synthesize(text, 'eager' if backend == 'fp32_eager' else 'prepared')
            calls = self.aoti_calls - before
            assert (calls >= 2*settings()['nfe'] and calls % (2*settings()['nfe']) == 0) if backend == 'bf16_aoti' else calls == 0
            self.completed[backend] += 1
            return body, {**metrics, 'compiled_dit_calls': calls}
        finally:
            self.active_backend = None
