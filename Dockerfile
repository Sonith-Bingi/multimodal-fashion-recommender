FROM python:3.11-slim

WORKDIR /app

# CPU-only wheel: far smaller/faster than the default CUDA build, and all a
# GPU-less serving container needs to run the two-tower model at inference.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.3"

COPY pyproject.toml README.md ./
COPY src ./src
COPY ui ./ui

# Serving only needs the already-trained model + FAISS index, not the
# sentence-transformers/CLIP encoders used at training time -- the `ci`
# extra (torch + faiss-cpu) plus `api` (fastapi + uvicorn) is the full set.
RUN pip install --no-cache-dir -e ".[ci,api]"

# Trained artifacts (item_index_v11.faiss, item_tower_vecs_v11.npy, the
# catalog cache, the two-tower checkpoint) are expected to be mounted here,
# not baked into the image -- see docker-compose.yml.
ENV RECO_DRIVE_DIR=/data
VOLUME ["/data"]

EXPOSE 8000

CMD ["uvicorn", "recommender.api:app", "--host", "0.0.0.0", "--port", "8000"]
