# Deployment

LandmarkDiff can be deployed in several ways depending on your use case.

## Docker

The included Dockerfile builds on `nvidia/cuda:12.1.1-devel-ubuntu22.04` with Python 3.11.

### Build and Run

```bash
# Build
docker build -t landmarkdiff .

# Run Gradio demo (default)
docker run -p 7860:7860 --gpus all landmarkdiff

# Run with specific GPU
docker run -p 7860:7860 --gpus '"device=0"' landmarkdiff

# CPU-only (TPS mode only)
docker run -p 7860:7860 landmarkdiff
```

The default CMD launches the Gradio demo on port 7860.

### Docker Compose

```bash
docker-compose up
```

The `docker-compose.yml` in the repo root handles GPU passthrough and port mapping.

### Custom Entry Points

Override the default command to run inference directly:

```bash
# Single image inference
docker run --gpus all -v /path/to/images:/data landmarkdiff \
    python -m landmarkdiff infer /data/face.jpg \
    --procedure rhinoplasty --intensity 60 --mode tps \
    --output /data/output/

# Launch CLI
docker run -it --gpus all landmarkdiff bash
```

### Pre-downloading Models

For air-gapped environments, pre-download models during build. Add to your Dockerfile:

```dockerfile
RUN python -c "
from diffusers import StableDiffusionPipeline
StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5')
"
```

This adds ~5 GB to the image but removes runtime downloads.

## Gradio Web Demo

The Gradio demo provides an interactive web interface for all procedures and modes.

### Local Launch

```bash
# Install Gradio deps
pip install -e ".[app]"

# Launch
python -m landmarkdiff demo
# or directly:
python scripts/app.py
```

Opens on `http://localhost:7860` by default.

### HuggingFace Spaces

The project has a live demo at [huggingface.co/spaces/dreamlessx/LandmarkDiff](https://huggingface.co/spaces/dreamlessx/LandmarkDiff). It runs TPS mode on CPU (free tier).

To deploy your own HF Space:

1. Create a new Space on HuggingFace (Gradio SDK)
2. Copy `scripts/app.py` and the `landmarkdiff/` package
3. Add a `requirements.txt` with core dependencies
4. Push to the Space's git repo

For GPU inference on HF Spaces, upgrade to a GPU-enabled runtime (T4, A10G, or A100).

## REST API

LandmarkDiff includes a FastAPI-based REST API server.

### Launch

```bash
pip install -e ".[app]"
python scripts/api_server.py --host 0.0.0.0 --port 8000
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Run inference on uploaded image |
| GET | `/health` | Health check |
| GET | `/procedures` | List supported procedures |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
    -F "image=@face.jpg" \
    -F "procedure=rhinoplasty" \
    -F "intensity=60" \
    -F "mode=tps" \
    --output prediction.png
```

### Production Considerations

For production deployments:

1. **Use Gunicorn/Uvicorn workers:**
   ```bash
   uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --workers 4
   ```
   With GPU, use 1 worker per GPU to avoid VRAM contention.

2. **Put a reverse proxy in front:**
   Nginx or Traefik for TLS, rate limiting, and static file serving.

3. **Add authentication:**
   The API server does not include auth by default. Add middleware for API key or OAuth.

4. **Set resource limits:**
   - Max image upload size (~10 MB is reasonable)
   - Request timeout (~30s for ControlNet, ~5s for TPS)
   - Concurrent request limit matching GPU count

## Cloud Deployment

### GPU Instances

Recommended instance types for ControlNet inference:

| Provider | Instance | GPU | VRAM | Cost/hr |
|----------|----------|-----|------|---------|
| AWS | g5.xlarge | A10G | 24 GB | ~$1.00 |
| GCP | a2-highgpu-1g | A100 | 40 GB | ~$3.50 |
| Lambda | gpu_1x_a10 | A10 | 24 GB | ~$0.75 |

For TPS-only deployment, any CPU instance works. A 2-core instance handles ~20 requests/sec.

### Container Registries

Push the Docker image to your container registry:

```bash
docker tag landmarkdiff your-registry.io/landmarkdiff:latest
docker push your-registry.io/landmarkdiff:latest
```

Then deploy via Kubernetes, ECS, Cloud Run, etc. Make sure to pass GPU resources in your pod spec.

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | For ControlNet | HuggingFace auth token |
| `CUDA_VISIBLE_DEVICES` | No | GPU selection |
| `TORCH_HOME` | No | Model cache directory |
| `GRADIO_SERVER_NAME` | No | Gradio bind address |
| `GRADIO_SERVER_PORT` | No | Gradio port (default 7860) |

## Safety Watermarking

For clinical or public-facing deployments, enable the safety watermark:

```python
from landmarkdiff.safety import SafetyValidator

validator = SafetyValidator(watermark_enabled=True, watermark_text="AI-GENERATED PREDICTION")
watermarked = validator.apply_watermark(result["output"])
```

Or via CLI:
```bash
landmarkdiff infer face.jpg --procedure rhinoplasty --watermark
```

This adds a visible "AI-GENERATED PREDICTION" text overlay to prevent the output from being mistaken for a real clinical photograph.
