FROM python:3.11-slim

WORKDIR /app

# System dependencies
# espeak-ng for phonemizer, ffmpeg for pydub
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    espeak-ng \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

# Consider installing torch and torchaudio separately if they cause issues with other packages
# For CPU:
# RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# For CUDA (adjust CUDA version, e.g., cu118, cu121):
# RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app

# Expose port
EXPOSE 8000

# Healthcheck (optional but good)
# HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
#   CMD curl -f http://localhost:8000/health || exit 1 # Add a /health endpoint to your API

# Run FastAPI app
# Use Gunicorn for production for more workers
# CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:8000", "app.main:app"]
# For development:
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]