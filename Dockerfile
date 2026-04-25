FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/app/.mplconfig

WORKDIR /app

# System deps for common python wheels + fonts for matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    fonts-dejavu-core \
  && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md openenv.yaml /app/
COPY delegation_gauntlet /app/delegation_gauntlet
COPY spaces /app/spaces
COPY public /app/public
COPY training /app/training
COPY tests /app/tests

# Install package + runtime deps
RUN pip install --upgrade pip && pip install -e .

EXPOSE 7860

# Hugging Face Spaces expects the app on port 7860
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

CMD ["python", "spaces/app.py"]

