# Product shelf recognition

PoC: cooler/shelf photo → detect products → match brands → planogram metrics → Excel/JSON in the browser.

## Run with Docker

Requires [Docker Desktop](https://www.docker.com/products/docker-desktop/) and [Git LFS](https://git-lfs.com).

```bash
git lfs install
git clone <repo-url>
cd Product-shelf-recognition
git lfs pull
docker compose up --build
```

Open http://127.0.0.1:8000

YOLO weights ship via Git LFS (`models/sku110k/`). SigLIP2 downloads from Hugging Face on first analyze.

Stop: `docker compose down`

## Run locally (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn poc.api:app --host 127.0.0.1 --port 8000
```

## Layout

```
catalog/      brand reference images
samples/      demo photos
planograms/   expected shelf schemes
models/       YOLO weights (Git LFS)
poc/          API + pipeline
web/          UI
```
