# Hugging Face Spaces (Docker SDK) image for the TrustPRO API.
# Only Python packages are installed here. Model files are NOT in the image: the app downloads
# them on start-up (scripts/download_models.py), like pip does for requirements.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000; the app folder must be writable for models/ and storage/.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user --force-reinstall --no-deps opencv-contrib-python==5.0.0.93

COPY --chown=user . .

EXPOSE 7860
# Apply database migrations (creates the schema on a fresh Neon database), then serve.
CMD ["sh", "-c", "python -m alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 7860"]
