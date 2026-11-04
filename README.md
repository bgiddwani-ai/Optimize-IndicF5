# Optimize IndicF5

Run an IndicF5 **FP32 eager → BF16 eager → BF16 + AOTInductor** comparison in
Kubernetes, then download one HTML report containing timings, quality scores and
audio samples.

The benchmark is one GPU Job. It builds the AOTI package on the GPU allocated to
that Pod, starts an internal HTTP worker, runs the comparisons sequentially,
saves the evidence to a PVC, and exits. Kubernetes releases the allocated GPU
when the Job finishes. An optional Deployment provides a persistent speech API.

This project uses standard `nvidia.com/gpu` resources. There are no fixed device
identifiers, node names, host directories, or cloud-provider requirements.
Paths such as `/data` and `/opt/indicf5` below are **generic container paths**.

## Requirements

* A Linux x86-64 Kubernetes GPU node with a CUDA 12.4-compatible driver and a
  BF16-capable NVIDIA GPU, such as an Ampere or newer GPU. This GPU workflow is
  not a CPU-only or AMD GPU implementation.
* An installed NVIDIA device plugin or GPU Operator that exposes
  `nvidia.com/gpu`. Use your cluster's configured RuntimeClass if one is required.
* A default dynamic StorageClass, or an explicit `STORAGE_CLASS`. The default PVC
  requests 40 GiB. Allow enough node disk space for the CUDA development image.
* An approved Hugging Face account with access to
  [ai4bharat/IndicF5](https://huggingface.co/ai4bharat/IndicF5).
* On your workstation: Git, Python 3, Bash, Docker, and `kubectl` connected to
  your cluster. Your cluster must be able to pull the image you build.

The container supplies Python 3.10, PyTorch 2.6.0 + CUDA 12.4, the CUDA compiler,
FFmpeg, the pinned IndicF5 source, and the locked Python dependencies. Model
weights download at runtime; neither weights nor credentials are in this repo.

## Quick start

```bash
git clone https://github.com/bgiddwani-ai/Optimize-IndicF5.git
cd Optimize-IndicF5
cp .env.example .env
```

Edit `IMAGE` in `.env` to a registry/repository you can push, for example:

```bash
IMAGE=ghcr.io/your-account/indicf5-aoti:v1
```

Authenticate Docker to that registry using its normal login procedure. Then:

```bash
./run.sh up       # build + push image, prompt for HF token if needed, submit Job
./run.sh logs     # follow pipeline phases; Ctrl-C only stops following logs
./run.sh wait     # wait for the benchmark Job
./run.sh download
```

Open `outputs/run-001/reports/report.html`. The full matrix is also available in
`reports/summary.csv`, with raw requests, WAVs and logs under `evidence/`.
The resolved `config.json` is downloaded alongside these directories.
Read `evidence/qualification.json` separately: a completed Job means that the
measurements finished, not that the candidate passed its numerical/ASR gates.

If the image is already available, use `./run.sh token` once and `./run.sh run`.
The token command prompts without echo and sends a Kubernetes Secret over stdin;
it does not put a token in a command argument, config file or Git commit.

## Two small configuration files

### `.env`: image and Kubernetes resources

```bash
IMAGE=ghcr.io/your-account/indicf5-aoti:v1
NAMESPACE=indicf5
RUN_ID=run-001
STORAGE_SIZE=40Gi
STORAGE_CLASS=
CPU_REQUEST=4
CPU_LIMIT=8
MEMORY_REQUEST=16Gi
MEMORY_LIMIT=32Gi
RUNTIME_CLASS=
NODE_SELECTOR=
PULL_SECRET=
```

Blank `STORAGE_CLASS` uses the default StorageClass. Blank `RUNTIME_CLASS` and
`NODE_SELECTOR` let the cluster choose its normal GPU runtime and a suitable node.
If needed, `NODE_SELECTOR='{"accelerator":"nvidia"}'` selects a **label you
manage**, not a built-in requirement. Set `PULL_SECRET` to an existing image-pull
Secret when your registry requires one. GPU memory requirements depend on input
length and the runtime; the Pod checks that it sees exactly one allocated GPU.

Change `RUN_ID` for each benchmark. With `JOB_NAME` blank, the Job name is derived
from it. Existing evidence is never silently overwritten. `CONFIG_FILE` selects
an alternative JSON configuration. `.env` and downloaded outputs are ignored by Git.

### `config.json`: inference and measurement

```json
{
  "nfe": 32,
  "groups": ["small", "medium", "large"],
  "concurrencies": [1, 2, 8, 16, 32],
  "minimum_requests": 4,
  "warmups": 1,
  "quality_enabled": true,
  "max_frames": 4096,
  "probe_frames": [128, 256, 512, 1024, 2048, 4096],
  "numerical_l2_max": 0.02,
  "numerical_cosine_min": 0.999,
  "cer_ceiling": 0.10,
  "serve_backend": "fp32_eager",
  "allow_unqualified_aoti": false
}
```

Omitted keys use the defaults in `app/config.py`. The checked-in config also sets
CPU and compiler thread counts. Unknown keys and invalid shape ranges fail early.
`max_frames` is a compiled-shape limit, not a request to shorten generated audio;
inputs exceeding it fail instead of silently changing duration. Keep all probes
within that limit and choose workloads that fit it.

For a quick deployment smoke test, use `CONFIG_FILE=configs/smoke.json` and a new
`RUN_ID`. That preset uses NFE=4 and skips ASR; it checks the workflow and **cannot
qualify speech quality or replace the full NFE=32 benchmark**.

## What runs

```text
Kubernetes Job: one allocated GPU + one PVC
  checkpoint preflight
  → torch.export + AOTInductor package build
  → load package in a fresh process, numerical probes and DiT-step timings
  → internal HTTP worker
  → paired speech generation and Hindi ASR
  → sequential FP32 / BF16 eager / BF16 AOTI request matrix
  → WAV audit, qualification.json, CSV/JSON/HTML
  → stop HTTP worker and finish Job
```

The default run has 45 cells and 576 measured requests. Each cell uses
`N=max(minimum_requests, concurrency)` with excluded real-synthesis warmups.
The three paths use the same fixed reference, tokenizer, target sequence,
per-text seed, duration calculation, Euler solver, CFG=2, sway=-1 and waveform
postprocessing. The default is NFE=32.

BF16 autocast applies to eligible DiT operations. Checkpoint parameters, mel
frontend, sampler state and Vocos stay FP32; TF32 is disabled. Both BF16 paths
cache text embeddings, RoPE, zero conditioning and per-timestep time embeddings.
AOTI runs the remaining prepared DiT graph from a compiled `.pt2` package.
There is no full-model eager fallback. Packages are rebuilt for the allocated
GPU architecture/runtime, rather than assumed portable between devices.

The HTTP model worker is serialized for RNG/cache correctness. Higher client
concurrency creates a queue; it is not independent-request batching or CUDA Graph
replay. AOTI versus prepared BF16 isolates compilation. AOTI versus original FP32
also includes reduced precision and cached preparation.

## Results and qualification

The report includes:

* Mean/P95 endpoint latency, actual response-body TTFB P50/P90/P95, throughput,
  mean request RTF, system RTF, processing time and queue time.
* Isolated DiT-step timings, compiled-call counters, configuration and source hashes.
* Finite/nonempty audio checks, per-backend waveform determinism, paired Hindi
  WER/CER and signed candidate-minus-control deltas.
* Synthetic-input numerical probes comparing AOTI with prepared eager BF16.

Responses are buffered WAVs: TTFB is not streaming time to first audible PCM.
System RTF is workload wall time divided by total generated audio duration.
Mean request RTF averages each request's queue-inclusive latency/audio duration.
Workload wall also includes client validation and saving. Loading, compilation,
ASR and warmups are excluded from endpoint timings.

Qualification requires the configured numerical gates, finite audio, an absolute
CER ceiling, and no aggregate WER/CER regression. A finite candidate can still be
benchmarked as **exploratory** when a gate fails. Skipping ASR cannot qualify it.
The fixed eight-sentence Hindi test is not a multilingual or production-SLO audit.

The sanitized [reference results](benchmarks/reference/README.md) preserve measured
timings and the failed numerical qualification. They are historical measurements
from the native HTTP runner, not a claim about performance of a new Kubernetes
deployment. Rerun this Job on your target hardware for comparable cluster results.

## Optional speech API

After the benchmark finishes, set a new `RUN_ID` for serving. The default backend
is FP32 eager:

```bash
./run.sh serve
kubectl -n indicf5 rollout status deployment/indicf5 --timeout=20m
./run.sh forward
```

From another terminal:

```bash
curl -f http://localhost:8000/healthz
curl -f http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"text":"नमस्ते।","backend":"fp32_eager"}' \
  --output speech.wav
```

For an **explicit experimental AOTI deployment**, set `serve_backend` to
`bf16_aoti` and `allow_unqualified_aoti` to `true`, then run `./run.sh serve`.
It builds and checks a package on the allocated GPU before starting the API.
The API does not silently fall back when AOTI fails. The Service is ClusterIP;
the helper forwards a local port, without creating a public endpoint.

The PVC uses ReadWriteOnce. Finish or stop one GPU workload before mounting that
claim in another workload on a different node. Separate namespaces/PVCs support
independent deployments.

## YAML, inspection and cleanup

Ready-to-read manifests are in [`k8s/`](k8s/). Render your configuration before
applying it:

```bash
./run.sh render > rendered.yaml
kubectl apply --dry-run=server -f rendered.yaml
./run.sh render --serve > rendered.yaml   # optional serving Deployment + Service
./run.sh status
./run.sh stop
```

`stop` removes this run's Job and the serving Deployment, releasing their GPU
allocations. The PVC and HF Secret remain available. The Job has no automatic
retry and a configurable time limit; this avoids repeated expensive builds after
a failure. Completed Jobs expire after seven days, while PVC results persist.

If the Pod is Pending, inspect its events for GPU capacity, storage binding,
runtime class or pull-secret errors. If a stage fails, its detailed log is under
`/data/runs/<run-id>/evidence/`. `./run.sh download` uses a temporary CPU-only Pod
to read completed results and removes that reader afterward. It requests no GPU.

## Validation and provenance

```bash
python3 -m unittest discover -s tests -v
./run.sh render > rendered.yaml
```

CPU CI checks configuration, rendering and source syntax. It does not claim to
test CUDA inference. GPU validation is performed by the Kubernetes Job itself;
inspect the downloaded report and `qualification.json` for each run.

An actual Kubernetes smoke run completed all three backends, 3/3 measured
requests, 16 compiled DiT calls, HTML report generation and CPU-only PVC download.
The optional Deployment also returned finite 24 kHz audio with its FP32 default.
The [sanitized validation summary](validation/kubernetes-smoke.json) records the
scope: NFE=4, ASR disabled, and no quality qualification. This checks execution;
the full NFE=32 performance/quality matrix must be rerun on the target hardware.

Source pins: IndicF5 `13f7c4d627cc10111aea8fe9c0039462cacacdc7`, model revision
`fb652a96e508091045890dae5bb99fe511facbcf`, and checkpoint SHA256
`ba7f3671180fb7784e24bd1dafc96e729a38ce02e7f6d3877cdef32525a1865c`.
The model is loaded strictly, including its Vocos weights. The original custom
character/pinyin tokenizer is preserved.

References: [IndicF5](https://github.com/AI4Bharat/IndicF5),
[PyTorch AOTInductor](https://docs.pytorch.org/docs/2.6/torch.compiler_aot_inductor.html),
[Kubernetes GPU scheduling](https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/),
[NVIDIA CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html),
and the [Sooktam2 optimization reference](https://github.com/bgiddwani-ai/Optimize-sooktam2).
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for model/source terms.
