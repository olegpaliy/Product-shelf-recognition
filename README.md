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

YOLO weights ship via Git LFS (`models/sku110k/`). Brand match uses SigLIP2 `google/siglip2-so400m-patch16-256` from Hugging Face on first analyze.

Stop: `docker compose down`

## Layout

```
catalog/      brand reference images
samples/      demo photos
planograms/   expected shelf schemes
models/       YOLO weights (Git LFS)
poc/          API + pipeline
web/          UI
```
