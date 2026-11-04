"""Validate a package loaded in a fresh process against eager BF16 and FP32."""
import json, time
from pathlib import Path
import numpy as np
import torch
from app.reference import Runtime
from app.step import BF16Step
from app.build import example
from app.config import settings


def main():
    r = Runtime(); step = BF16Step(r.model.transformer).eval()
    compiled = torch._inductor.aoti_load_package('artifacts/dit_bf16.pt2')
    rows = []
    torch.manual_seed(741)
    for length,time_value in [(length,t) for length in settings()['probe_frames'] for t in (0.0,0.37,0.95)]:
        with torch.inference_mode():
            args = example(r.model.transformer, length, time_value)
            expected = step(*args)
            actual = compiled(*args)
            assert actual.dtype == torch.bfloat16 and actual.is_cuda
            difference = actual.float()-expected.float()
            relative = float(torch.linalg.vector_norm(difference)/torch.linalg.vector_norm(expected.float()))
            cosine = float(torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(),dim=0))
            row = {'frames':length,'time':time_value,'relative_l2':relative,'cosine':cosine,
                   'max_abs':float(difference.abs().max()),'finite':bool(torch.isfinite(actual).all()),
                   'output_dtype':str(actual.dtype)}
            rows.append(row)
            assert row['finite'], row
            row['passed'] = relative <= settings()['numerical_l2_max'] and cosine >= settings()['numerical_cosine_min']
            print(json.dumps(row),flush=True)
    Path('evidence/numerical-validation.json').write_text(json.dumps({'passed':all(r['passed'] for r in rows),'finite_passed':True,
        'performance_policy':'Exploratory performance allowed for finite candidates; failed numerical gates remain failed and prevent qualification.',
        'gate':{'relative_l2_max':settings()['numerical_l2_max'],'cosine_min':settings()['numerical_cosine_min']},'comparison':'AOTI versus same prepared eager BF16 graph','rows':rows},indent=2))
    timings=[]
    for length in [v for v in (256,1024,2048) if v <= settings()['max_frames']]:
        with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
            args=example(r.model.transformer,length)
            for name,call in [('bf16_eager',step),('bf16_aoti',compiled)]:
                for _ in range(5):call(*args)
                torch.cuda.synchronize()
                gpu_ms=[];wall_ms=[]
                for _ in range(20):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    t=time.perf_counter();start.record();call(*args);end.record();end.synchronize()
                    gpu_ms.append(start.elapsed_time(end));wall_ms.append((time.perf_counter()-t)*1000)
                timings.append({'frames':length,'backend':name,'n':20,'warmups':5,
                    'gpu_ms_mean':float(np.mean(gpu_ms)),'gpu_ms_p50':float(np.median(gpu_ms)),
                    'gpu_ms_p95':float(np.percentile(gpu_ms,95)),'wall_ms_mean':float(np.mean(wall_ms))})
    Path('evidence/step-benchmark.json').write_text(json.dumps(timings,indent=2))


if __name__ == '__main__':main()
