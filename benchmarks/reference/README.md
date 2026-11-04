# Reference measurements

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
Responses were buffered WAVs; TTFB is not streaming PCM latency. Detailed metric
boundaries and the reproducible workload are documented in the main README.
