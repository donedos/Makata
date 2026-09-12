# Makata — voice-cloning TTS server
#
# Build : docker build -t makata .
# Run   : docker run --gpus all -p 8300:8300 -v makata-data:/data makata
# (see docker-compose.yml)

FROM python:3.11-slim

# ffmpeg for broad audio decode; git not needed at runtime
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Torch first as its own layer (biggest dep, cached across rebuilds).
# Linux PyPI wheels ship CUDA 12 runtime libs; the NVIDIA *driver* comes
# from the host via --gpus all / nvidia-container-toolkit.
RUN pip install --no-cache-dir "torch>=2.3"

WORKDIR /srv/makata

COPY pyproject.toml README.md ./
COPY app ./app

RUN pip install --no-cache-dir .

# Persist voices, samples, outputs and HuggingFace model downloads
ENV MAKATA_DATA_DIR=/data \
    HF_HOME=/data/hf \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN useradd --create-home --uid 1000 makata \
    && mkdir -p /data \
    && chown -R makata:makata /data /srv/makata
USER makata

VOLUME /data
EXPOSE 8300

ENTRYPOINT ["makata", "serve", "--host", "0.0.0.0"]
CMD ["--port", "8300"]
