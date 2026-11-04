FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUTF8=1 \
    HF_HOME=/data/cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/data/cache/torchinductor \
    TRITON_CACHE_DIR=/data/cache/triton CUDA_CACHE_PATH=/data/cache/cuda \
    NUMBA_CACHE_DIR=/data/cache/numba XDG_CACHE_HOME=/data/cache \
    HOME=/tmp NVIDIA_VISIBLE_DEVICES=void PYTHONPATH=/opt/indicf5:/opt/IndicF5
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv git ffmpeg libsndfile1 g++ ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock.txt /opt/indicf5/requirements.lock.txt
RUN python3 -m pip --isolated install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0+cu124 torchaudio==2.6.0+cu124 \
    && python3 -m pip --isolated install --no-cache-dir --index-url https://pypi.org/simple \
      -r /opt/indicf5/requirements.lock.txt
# AOTInductor/Triton compile native Python extensions on the allocated GPU.
RUN apt-get update && apt-get install -y --no-install-recommends python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -c "import pathlib, sysconfig; assert (pathlib.Path(sysconfig.get_path('include')) / 'Python.h').is_file()"
RUN git clone https://github.com/AI4Bharat/IndicF5.git /opt/IndicF5 \
    && git -C /opt/IndicF5 checkout --detach 13f7c4d627cc10111aea8fe9c0039462cacacdc7 \
    && rm -rf /opt/IndicF5/.git
COPY src/app /opt/indicf5/app
COPY src/benchmark /opt/indicf5/benchmark
COPY src/tools /opt/indicf5/tools
RUN python3 -c "import app.reference, app.runtime, benchmark.report"
RUN mkdir -p /data && chown 1000:1000 /data
USER 1000:1000
WORKDIR /opt/indicf5
EXPOSE 8000
ENTRYPOINT ["python3", "-m", "tools.pipeline"]
