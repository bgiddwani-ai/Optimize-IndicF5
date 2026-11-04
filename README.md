# Optimize IndicF5

Build one container and run **FP32 eager, BF16 eager and BF16 + AOTInductor**
benchmarks as one Kubernetes GPU Job.

You need Docker, Git, Bash, Python 3 and `kubectl`; a Linux x86-64 Kubernetes
node with a BF16-capable NVIDIA GPU, a CUDA 12.4-compatible driver and the NVIDIA
device plugin; dynamic storage; and approved access to
[IndicF5](https://huggingface.co/ai4bharat/IndicF5).

## 1. Create the container

Run these commands in the same terminal. Choose an image registry you can push
to, log in to it, and ensure your Kubernetes nodes can pull that image.

```bash
git clone https://github.com/bgiddwani-ai/Optimize-IndicF5.git
cd Optimize-IndicF5
export IMAGE=ghcr.io/your-account/indicf5-aoti:v1

docker login ghcr.io
docker build --platform linux/amd64 -t "$IMAGE" .
docker push "$IMAGE"
```

## 2. Kubernetes configuration

[k8s.yaml](k8s.yaml) contains the Job, storage and configuration resources.
It requests **1 GPU**, 4 CPUs/16 GiB RAM with limits of 8 CPUs/32 GiB, and a
40 GiB persistent volume. Edit those resource values there if needed.
`run.sh` fills the YAML placeholders automatically.

[config.json](config.json) already selects the full benchmark: NFE=32,
small/medium/large inputs, concurrency 1/2/8/16/32, and Hindi quality checks.
Keep these defaults to repeat the measured workload: **45 cells, 576 requests**.

Only if your cluster requires overrides, set its RuntimeClass, StorageClass or
an existing image-pull Secret before running:

```bash
# Optional — use names configured in your own cluster.
# export RUNTIME_CLASS=nvidia
# export STORAGE_CLASS=your-storage-class
# export PULL_SECRET=your-registry-secret
```

## 3. Run and download results

```bash
./run.sh run       # prompts for an approved HF token on the first run
./run.sh wait
./run.sh download
```

Open **`outputs/run-001/reports/report.html`**. CSV, raw measurements, WAVs and
quality checks are downloaded alongside it. The Job compiles AOTI on its allocated
GPU, measures all three backends, saves results, then exits and releases the GPU.
All paths are generic container or checkout-relative paths.

Use `./run.sh logs` for progress, or `./run.sh stop` to cancel the Job while
retaining its data. For another run, `export RUN_ID=run-002` and repeat step 3.

## Our measured results

H100 80GB · PyTorch 2.6.0/CUDA 12.4 · NFE=32 · concurrency 1 · N=4 per cell.

| Input | FP32 mean latency | BF16 mean latency | BF16+AOTI mean latency | AOTI/FP32 throughput |
|---|---:|---:|---:|---:|
| Small | 1.259 s | 0.962 s | 0.336 s | 3.74× |
| Medium | 1.750 s | 0.985 s | 0.387 s | 4.52× |
| Large | 1.988 s | 0.987 s | 0.420 s | 4.72× |

**576/576 requests succeeded.** All three backends scored 11.22% Hindi WER and
2.65% CER on eight sentences. **AOTI remains exploratory:** 7/18 numerical probes
failed the preset gates. Check `evidence/qualification.json` in each new run.

The [complete results and method](results/README.md) include the
[45-cell CSV](results/summary.csv), quality scores and numerical probes.
These full measurements used the native HTTP runner. The Kubernetes smoke test
separately passed 3/3 requests at NFE=4 with ASR disabled; rerun the full Job for
your target hardware's results. Implementation lives in `src/`.

Model/source terms: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
