"""Reports contain workload, hardware class and measured results, not host identity."""
import base64
import csv
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from app.config import settings


def main():
    root = Path.cwd(); evidence = root/'evidence'; output = root/'reports'
    output.mkdir(exist_ok=True)
    def read(name):
        return json.loads((evidence/name).read_text())
    cfg = settings()
    rows = read('matrix/summary.json')
    numeric = read('numerical-validation.json')
    quality = read('quality/asr.json')
    server = read('server-final.json')
    build = read('aoti-build.json')
    micro = read('step-benchmark.json')
    backends = ['fp32_eager', 'bf16_eager', 'bf16_aoti']
    expected_cells = len(cfg['groups'])*len(cfg['concurrencies'])*len(backends)
    expected_requests = len(cfg['groups'])*len(backends)*sum(max(cfg['minimum_requests'], c) for c in cfg['concurrencies'])
    assert len(rows) == expected_cells and sum(r['success'] for r in rows) == expected_requests
    assert sum(r['failures'] for r in rows) == 0 and numeric['finite_passed']
    count = 0; canonical = {}
    for folder in sorted((evidence/'matrix').glob('*_c*')):
        for index, line in enumerate((folder/'requests.jsonl').read_text().splitlines()):
            row = json.loads(line); data = (folder/f'{index:03d}.wav').read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            key = (row['backend'], row['target_index'])
            assert row['ok'] and digest == row['wav_sha256']
            if not quality.get('skipped'):
                reference = evidence/'quality'/row['backend']/f'{row["target_index"]:02d}.wav'
                assert data == reference.read_bytes(), (folder, index)
            assert digest == canonical.setdefault(key, digest), 'Within-backend output changed'
            count += 1
    assert count == expected_requests and server['compiled_dit_calls'] > 0
    qualified = bool(numeric['passed'] and quality['passed'])
    verdict = {'qualified': qualified, 'numerical_gate_passed': numeric['passed'],
               'asr_gate_passed': quality['passed'], 'asr_skipped': quality.get('skipped', False),
               'finite_passed': True, 'measured_requests': count}
    (evidence/'qualification.json').write_text(json.dumps(verdict, indent=2))
    (evidence/'all-requests-audit.json').write_text(json.dumps({'passed': True, 'requests': count,
        'compiled_dit_calls': server['compiled_dit_calls'], 'within_backend_deterministic': True}, indent=2))
    with (output/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output/'summary.json').write_text(json.dumps(rows, indent=2))
    lookup = {(r['group'], r['backend'], r['concurrency']): r for r in rows}
    first_c = min(cfg['concurrencies'])
    labels = {'fp32_eager': 'FP32 eager', 'bf16_eager': 'BF16 prepared eager', 'bf16_aoti': 'BF16 + AOTI'}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout='constrained')
    for i, (backend, color) in enumerate(zip(backends, ['#63727a', '#2988aa', '#76b900'])):
        axes[0].bar([j+(i-1)*.25 for j in range(len(cfg['groups']))],
            [lookup[g, backend, first_c]['throughput_req_s'] for g in cfg['groups']], width=.25, color=color, label=labels[backend])
        group = cfg['groups'][-1]
        axes[1].plot(cfg['concurrencies'], [lookup[group, backend, c]['latency_s_p95'] for c in cfg['concurrencies']],
                     marker='o', color=color, label=labels[backend])
    axes[0].set_xticks(range(len(cfg['groups'])), cfg['groups']); axes[0].set_ylabel('Requests / second')
    axes[0].set_title(f'Endpoint throughput, concurrency {first_c}')
    axes[1].set_title(f'{cfg["groups"][-1]} input: queue-inclusive P95'); axes[1].set_ylabel('Seconds'); axes[1].set_xlabel('Concurrency')
    for axis in axes:
        axis.legend(fontsize=8); axis.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'IndicF5 · {server["gpu_name"]} · NFE {cfg["nfe"]}')
    fig.savefig(output/'benchmark.png', dpi=160); fig.savefig(output/'benchmark.svg'); plt.close(fig)
    image = base64.b64encode((output/'benchmark.png').read_bytes()).decode()
    def cell(value):
        return f'{value:.4f}' if isinstance(value, float) else html.escape(str(value))
    def table(data, columns):
        return '<div class="scroll"><table><tr>'+''.join(f'<th>{label}</th>' for _, label in columns)+'</tr>'+''.join(
            '<tr>'+''.join(f'<td>{cell(row.get(key, ""))}</td>' for key, _ in columns)+'</tr>' for row in data)+'</table></div>'
    comparisons = []
    for group in cfg['groups']:
        a, b, c = [lookup[group, backend, first_c] for backend in backends]
        comparisons.append({'group': group, 'fp32_s': a['latency_s_mean'], 'bf16_s': b['latency_s_mean'], 'aoti_s': c['latency_s_mean'],
            'vs_fp32_x': c['throughput_req_s']/a['throughput_req_s'], 'vs_bf16_x': c['throughput_req_s']/b['throughput_req_s']})
    (output/'comparison.json').write_text(json.dumps(comparisons, indent=2))
    comparisons_table = table(comparisons, [('group', 'Input'), ('fp32_s', 'FP32 mean s'), ('bf16_s', 'BF16 mean s'),
        ('aoti_s', 'AOTI mean s'), ('vs_fp32_x', 'AOTI / FP32 throughput'), ('vs_bf16_x', 'AOTI / BF16 throughput')])
    matrix = table(rows, [('group','Input'),('backend','Backend'),('concurrency','C'),('requests','N'),
        ('throughput_req_s','req/s'),('generated_audio_s','Audio s'),('system_rtf','System RTF'),('mean_request_rtf','Mean request RTF'),
        ('latency_s_mean','Mean E2E s'),('latency_s_p95','P95 E2E s'),('ttfb_s_p50','TTFB P50 s'),
        ('ttfb_s_p90','TTFB P90 s'),('ttfb_s_p95','TTFB P95 s'),('process_s_mean','Mean process s'),('queue_s_mean','Mean queue s')])
    numeric_table = table(numeric['rows'], [('frames','Frames'),('time','Time'),('relative_l2','Relative L2'),('cosine','Cosine'),('passed','Gate passed')])
    step_table = table(micro, [('frames','Frames'),('backend','Backend'),('n','N'),('gpu_ms_mean','CUDA interval mean ms'),
        ('gpu_ms_p95','CUDA interval P95 ms'),('wall_ms_mean','Host mean ms')])
    quality_html = '<p>ASR was disabled. This run cannot qualify the candidate.</p>'
    if not quality.get('skipped'):
        scores = [{'backend': b, 'wer': quality['results'][b]['wer']*100, 'cer': quality['results'][b]['cer']*100} for b in backends]
        quality_html = table(scores, [('backend','Backend'),('wer','WER %'),('cer','CER %')])
        quality_html += table([{'comparison': k, **v} for k, v in quality['deltas'].items()],
            [('comparison','Candidate minus control'),('wer_delta_pp','WER delta pp'),('cer_delta_pp','CER delta pp'),('no_regression','No regression')])
        samples = []
        for index, target in enumerate(quality['results']['fp32_eager']['rows']):
            cells = f'<td>{index}</td><td>{html.escape(target["target"])}</td>'
            for backend in backends:
                data = base64.b64encode((evidence/'quality'/backend/f'{index:02d}.wav').read_bytes()).decode()
                hypothesis = quality['results'][backend]['rows'][index]['hypothesis']
                cells += f'<td>{html.escape(hypothesis)}<br><audio controls preload="none" src="data:audio/wav;base64,{data}"></audio></td>'
            samples.append('<tr>'+cells+'</tr>')
        quality_html += '<div class="scroll"><table class="audio"><tr><th>ID</th><th>Target</th><th>FP32</th><th>BF16 eager</th><th>AOTI</th></tr>'+''.join(samples)+'</table></div>'
    document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IndicF5 benchmark</title>
<style>body{{font:16px/1.55 system-ui;color:#17252c;background:#f3f5f3;margin:0}}main{{max-width:1500px;margin:auto;padding:32px}}section{{background:white;padding:24px;border-radius:12px;margin:20px 0}}h1{{font-size:36px}}h2{{margin-top:0}}img{{width:100%}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:9px;text-align:right;white-space:nowrap;border-bottom:1px solid #e2e8e0}}td:first-child,th:first-child{{text-align:left}}th{{background:#edf3e5}}.audio td{{white-space:normal;text-align:left}}audio{{width:210px}}code{{overflow-wrap:anywhere}}</style>
<main><h1>IndicF5: BF16 + AOTInductor</h1><p>{count}/{count} successful requests · {len(rows)} cells · {server['compiled_dit_calls']:,} compiled DiT calls</p>
<section><h2>Qualification: {'PASSED' if qualified else 'NOT QUALIFIED'}</h2><p>Numerical gate: {'PASSED' if numeric['passed'] else 'FAILED'}. Hindi ASR gate: {'PASSED' if quality['passed'] else 'FAILED / NOT RUN'}. Finite performance results are exploratory when either gate fails. Pipeline completion does not imply candidate acceptance.</p></section>
<section><h2>Endpoint performance</h2><img src="data:image/png;base64,{image}" alt="Measured throughput and latency">{comparisons_table}
<p>Comparison at concurrency {first_c}. AOTI versus prepared BF16 isolates compilation using the same preparation and precision. AOTI versus original FP32 also includes BF16 and cached conditioning. One serialized model worker handles all requests; client concurrency creates a queue, with no independent-request batching.</p></section>
<section><h2>Complete matrix</h2>{matrix}<p>N=max({cfg['minimum_requests']},C); {cfg['warmups']} excluded warmup(s) per cell. Fixed Hindi targets: small 0–1, medium 2–4, large 5–7. Different concurrency cells can have different cyclic target counts. These small samples are engineering measurements, not production SLO qualification.</p></section>
<section><h2>Paired Hindi quality</h2>{quality_html}<p>ASR is measured outside performance timing using a pinned Whisper large-v3-turbo revision with forced Hindi. NFC normalization, punctuation to space, collapsed whitespace; CER includes spaces. The fixed eight-sentence Hindi workload and upstream Punjabi reference do not qualify all supported languages. No independent human listening was performed. Every measured WAV is finite and within-backend deterministic.</p></section>
<section><h2>Synthetic numerical probes</h2>{numeric_table}<p>Relative L2 limit {cfg['numerical_l2_max']}, minimum cosine {cfg['numerical_cosine_min']}. These compare the loaded AOTI package against the same prepared eager BF16 graph. Finite output alone does not prove numerical equivalence.</p><h2>Isolated prepared DiT steps</h2>{step_table}<p>Five warmups and twenty measured calls. CUDA-event intervals include idle gaps from host dispatch; they are not sums of isolated kernel durations. Synthetic prepared inputs include an eager-computed time embedding. Full endpoint timings include preparation and waveform postprocessing.</p></section>
<section><h2>Runtime and timing contract</h2><ul><li>BF16 autocast applies to eligible DiT operations. Checkpoint parameters, mel frontend, sampler state and Vocos remain FP32; TF32 is disabled.</li><li>Both BF16 paths cache text, RoPE, zero conditioning and per-timestep time embeddings. NFE={cfg['nfe']}, Euler, CFG=2, sway=-1, deterministic seeds, upstream duration and waveform postprocessing are shared.</li><li>TTFB is the first real response-body byte. Responses are buffered WAVs, not streaming PCM. E2E ends at the final response byte. Queue time is measured before the serialized worker starts.</li><li>System RTF = workload wall / generated audio duration. Mean request RTF averages each request's E2E / audio duration. Workload wall includes client validation and saving. Loading, compilation, ASR and warmups are excluded.</li></ul><p>GPU class: {html.escape(server['gpu_name'])}; Torch {html.escape(build['torch'])}; CUDA {html.escape(build['cuda'])}. Build time {build['compile_s']:.1f}s. AOTI package SHA256 <code>{build['package_sha256']}</code>. The package was built and loaded on the allocated GPU; serving has no full-model eager fallback.</p><p>Model revision <code>{build['revision']}</code>; checkpoint SHA256 <code>{build['checkpoint_sha256']}</code>. Configuration, source hashes, raw requests, WAVs and logs accompany this report.</p></section></main></html>'''
    (output/'report.html').write_text(document)
    print(json.dumps({**verdict, 'cells': len(rows), 'comparisons': comparisons}), flush=True)


if __name__ == '__main__':
    main()
