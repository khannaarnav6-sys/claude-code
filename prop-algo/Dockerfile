FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.yaml .
COPY src/ src/
COPY scripts/ scripts/

# DISCORD_WEBHOOK_URL must be provided at runtime:
#   docker run -e DISCORD_WEBHOOK_URL=... propalgo
CMD ["python", "scripts/cli.py", "live"]
