import json, time
from pathlib import Path
import torch
from app.reference import Runtime, sha256
from app.step import BF16Step
from app.config import settings


def example(dit, length, time_value=0.37):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        time_embedding = dit.time_embed(torch.tensor([time_value], device='cuda')).contiguous()
    return (
        torch.randn(1, length, 100, device='cuda'),
        torch.randn(1, length, 100, device='cuda'),
        torch.randn(1, length, 512, device='cuda', dtype=torch.bfloat16),
        time_embedding,
        dit.rotary_embed.forward_from_seq_len(length)[0].contiguous(),
    )


def main():
    root = Path('artifacts'); root.mkdir(exist_ok=True)
    evidence = Path('evidence'); evidence.mkdir(exist_ok=True)
    assert torch.cuda.is_bf16_supported(), "A BF16-capable NVIDIA GPU is required."
    runtime = Runtime()
    dit = runtime.model.transformer
    step = BF16Step(dit).eval()
    package = root/'dit_bf16.pt2'
    metadata = {**runtime.metadata, 'compute_dtype': 'bfloat16 autocast DiT; FP32 sampler and Vocos',
                'source_sha256': sha256(Path('app/step.py')), 'builder_sha256': sha256(Path(__file__)),
                'upstream_commit': '13f7c4d627cc10111aea8fe9c0039462cacacdc7',
                'dynamic_frames': [128, settings()['max_frames']], 'batch': 1,
                'format': 'AOTInductor compiled package, torch.export + aoti_compile_and_package',
                'compiler_config': {'emulate_precision_casts': False},
                'native_external_ops': 'standard AOTInductor operator selection',
                'time_embedding': 'upstream eager BF16 autocast outside compiled graph, cached once per timestep'}
    (evidence/'runtime.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    torch.manual_seed(741)
    started = time.perf_counter()
    with torch.no_grad():
        args = example(dit, 512)
        frames = torch.export.Dim('frames', min=128, max=settings()['max_frames'])
        exported = torch.export.export(step, args, dynamic_shapes=({1: frames}, {1: frames}, {1: frames}, None, {1: frames}), strict=False)
        (evidence/'export-graph.txt').write_text(str(exported.graph))
        result = torch._inductor.aoti_compile_and_package(exported, package_path=str(package),
                    inductor_configs={'emulate_precision_casts': False})
    torch.cuda.synchronize()
    metadata.update(package=str(result), package_sha256=sha256(package), package_bytes=package.stat().st_size,
                    compile_s=time.perf_counter()-started)
    (evidence/'aoti-build.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(json.dumps({'built': str(result), 'compile_s': metadata['compile_s'], 'sha256': metadata['package_sha256']}), flush=True)


if __name__ == '__main__':
    main()
