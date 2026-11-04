# Measured results

These are measured **native HTTP-runner** results on one H100 80GB GPU, with
PyTorch 2.6.0/CUDA 12.4 and NFE=32. They are historical results, not measurements
of the new Kubernetes wrapper. Rerun the Kubernetes Job on your target hardware.
No device identity, node name, provider information, credential, or original
filesystem path is included here.

Mean endpoint latency at concurrency 1 (N=4 per cell):

| Input | FP32 eager | BF16 prepared eager | BF16 + AOTI | AOTI / FP32 throughput |
|---|---:|---:|---:|---:|
| Small | 1.259 s | 0.962 s | 0.336 s | 3.74x |
| Medium | 1.750 s | 0.985 s | 0.387 s | 4.52x |
| Large | 1.988 s | 0.987 s | 0.420 s | 4.72x |

All 576 measured requests succeeded across 45 cells. The fixed Hindi sample
scores were 11.22% WER and 2.65% CER for every backend, with zero aggregate deltas.
This is an eight-sentence automated quality check, not a multilingual audit.

**The AOTI candidate did not qualify.** Seven of eighteen synthetic numerical
probes failed the preset relative-L2/cosine gates; maximum relative L2 was 6.85%
against a 2% ceiling, with minimum cosine 0.997660 against 0.999. All outputs were
finite. Performance is exploratory; unchanged ASR scores do not erase this failure.

`summary.csv` and `summary.json` contain the complete latency/TTFB/RTF/concurrency
matrix. `quality.json` contains paired transcripts and signed WER/CER deltas.
`numerical.json` contains every numerical probe. The compiler-only comparison
against the same prepared BF16 graph is in `comparison-c1.json`.

One serialized model worker was used, with no independent-request batching.
AOTI versus FP32 includes precision, preparation and compilation changes.
Responses were buffered WAVs; TTFB is not streaming PCM latency. Metric
boundaries and the workload are documented below.

## Reproduction details

The checked-in `config.json` uses the same eight Hindi targets and fixed Punjabi
reference. All backends share per-text seeds, the upstream character/pinyin
tokenizer, duration calculation, Euler solver, NFE=32, CFG=2, sway=-1, speed=1
and waveform postprocessing. Each cell measures `N=max(4, concurrency)` after
one excluded warmup. Concurrency is 1/2/8/16/32 for small/medium/large inputs.

BF16 autocast applies to eligible DiT operations. Checkpoint parameters, the mel
frontend, sampler state and Vocos remain FP32; TF32 is disabled. Both BF16 paths
cache text embeddings, RoPE, zero conditioning and time embeddings per request.
AOTI compiles the remaining prepared DiT graph into a `.pt2` package. AOTI versus
prepared BF16 isolates compilation; AOTI versus FP32 includes all these changes.
Packages are built on the allocated GPU and loaded in a fresh process. The
compiled frame range is 128–4096; longer inputs fail rather than being truncated.

Endpoint latency includes queueing and buffered WAV transfer. TTFB measures the
first actual response-body byte. System RTF is workload wall time divided by
generated audio duration; mean request RTF averages each request's latency/audio
duration. Workload wall time includes client validation and saving. Compilation,
model loading, ASR and warmups are excluded from endpoint timings.

Quality uses paired audio and a pinned Whisper large-v3-turbo Hindi evaluator.
Qualification requires finite audio, the numerical gates, CER <=10%, and no
aggregate WER/CER regression. This eight-sentence test is not a multilingual
quality audit. A completed Job means measurement completed; inspect its
`evidence/qualification.json` for acceptance.

Source pins:

- IndicF5 commit: `13f7c4d627cc10111aea8fe9c0039462cacacdc7`
- Model revision: `fb652a96e508091045890dae5bb99fe511facbcf`
- Checkpoint SHA256: `ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c`

## Kubernetes execution check

[kubernetes-smoke.json](kubernetes-smoke.json) records an actual successful
Kubernetes Job: 3/3 measured requests across all three backends, 16 compiled DiT
calls, HTML report generation and CPU-only PVC result download. It used NFE=4
and skipped ASR, so it verifies execution and does not qualify speech quality.
The matching short configuration is now at `src/smoke.json`.
