FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY swarm ./swarm
COPY scripts ./scripts

# Unprivileged user: edge devices must not run as root.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin swarm
USER 10001
