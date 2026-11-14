FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# CPU-only PyTorch keeps the image small (no CUDA needed for this project)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-medical.txt ./
RUN pip install -r requirements.txt -r requirements-medical.txt

COPY fedswarm ./fedswarm

CMD ["python", "-m", "fedswarm.cli.server"]
