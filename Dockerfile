# Single image used by BOTH roles (global server and node). The role is chosen
# at run time by the command (see docker-compose.yml / k8s manifests), exactly
# like the loopback launcher shells out to either uvicorn or `python -m nodes.node`.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# CPU-only torch stack first. This keeps the image ~1GB instead of the ~5GB the
# default CUDA wheels would pull in — the code already falls back to CPU when no
# GPU is present, so nothing else changes.
RUN pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Remaining runtime deps (torchaudio from requirements.txt is unused, so skipped).
RUN pip install --no-cache-dir \
    numpy pandas pyyaml tqdm fastapi uvicorn matplotlib requests pysyncobj

# App code. data/MNIST/raw is copied in so torchvision finds it locally and never
# needs to download at runtime (download=True is a no-op when the files exist).
COPY . /app

# Default role = global server. Node services override this command.
EXPOSE 8000
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
