"""Closed-loop HTTP benchmark; first byte means a real response-body byte."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import requests
import soundfile as sf
from app.workload import TARGETS, GROUPS
from app.config import settings

def request(endpoint, backend, i, save=None):
    start = time.perf_counter()
    row = {"target_index": i, "backend": backend}
    try:
        with requests.post(endpoint + "/v1/audio/speech", json={"text": TARGETS[i], "backend": backend},
                           stream=True, timeout=1800, headers={"Connection": "close"}) as r:
            r.raise_for_status()
            first = r.raw.read(1)
            ttfb = time.perf_counter() - start
            assert first, "empty response body"
            body = first + r.raw.read()
            elapsed = time.perf_counter() - start
            data, sr = sf.read(io.BytesIO(body))
            assert data.size and np.isfinite(data).all()
            rms = float(np.sqrt(np.mean(data**2)))
            assert rms >= 0.001, rms
            row.update(ok=True, latency_s=elapsed, ttfb_s=ttfb, audio_s=len(data)/sr,
                       process_s=float(r.headers["X-Process-Seconds"]), queue_s=float(r.headers["X-Queue-Seconds"]),
                       sample_rate=sr, samples=len(data), rms=rms, wav_sha256=hashlib.sha256(body).hexdigest())
            if save:
                save.write_bytes(body)
    except Exception as e:
        row.update(ok=False, latency_s=time.perf_counter()-start, error=str(e))
    return row

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--endpoint",default="http://127.0.0.1:8000")
    p.add_argument("--output",default="evidence/matrix")
    p.add_argument("--backends",nargs="+",default=["fp32_eager","bf16_eager","bf16_aoti"])
    p.add_argument("--concurrencies",type=int,nargs="+",default=settings()["concurrencies"])
    args=p.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    health=requests.get(args.endpoint+"/healthz",timeout=10).json()
    (root/"server_health.json").write_text(json.dumps(health,ensure_ascii=False,indent=2), encoding="utf-8")
    summaries=[]
    # Alternate baseline/candidate per matrix cell to reduce order bias.
    for group in settings()["groups"]:
        indices=GROUPS[group]
        for c in args.concurrencies:
            for backend in args.backends:
                case=root/f"{group}_{backend}_c{c}";case.mkdir(exist_ok=True)
                for _ in range(settings()["warmups"]):
                    warm=request(args.endpoint,backend,indices[0]);assert warm["ok"],warm
                n=max(settings()["minimum_requests"],c)
                scheduled=[indices[i%len(indices)] for i in range(n)]
                start=time.perf_counter()
                with ThreadPoolExecutor(max_workers=c) as pool:
                    rows=list(pool.map(lambda pair: request(args.endpoint,backend,pair[1],case/f"{pair[0]:03d}.wav"),enumerate(scheduled)))
                wall=time.perf_counter()-start
                (case/"requests.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
                valid=[row for row in rows if row["ok"]]
                summary={"group":group,"backend":backend,"concurrency":c,"requests":n,
                         "success":len(valid),"failures":n-len(valid),"wall_s":wall,
                         "generated_audio_s":sum(row["audio_s"] for row in valid),
                         "throughput_req_s":len(valid)/wall,
                         "system_rtf":wall/sum(row["audio_s"] for row in valid) if valid else None}
                for metric in ("latency_s","ttfb_s","process_s","queue_s"):
                    values=[row[metric] for row in valid]
                    if values:
                        summary[metric+"_mean"]=float(np.mean(values))
                        for q in (50,90,95):summary[metric+f"_p{q}"]=float(np.percentile(values,q))
                summary["mean_request_rtf"]=float(np.mean([row["latency_s"]/row["audio_s"] for row in valid])) if valid else None
                (case/"summary.json").write_text(json.dumps(summary,indent=2))
                summaries.append(summary)
                (root/"summary.json").write_text(json.dumps(summaries,indent=2))
                print(json.dumps(summary),flush=True)
                assert not summary["failures"],summary

if __name__=="__main__":main()
