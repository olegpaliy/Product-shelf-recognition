# Product shelf recognition

PoC: фото холодильника / полиці → QA → рамки → бренд → груба схема викладки → Excel, у браузері.

## Документи
- [План PoC](docs/POC-PLAN.md)

## Швидкий старт

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# еталони брендів (авто-кропи з samples через CLIP text)
python -m poc.seed_catalog
# або синтетичні кольорові рефи (слабше):
# python -m poc.build_catalog

# батч по samples/ → out/ (JSON + Excel + annotated)
python -m poc.run --input samples/ --output out/
python -m poc.index_out

# веб-UI
uvicorn poc.api:app --reload --host 127.0.0.1 --port 8000
```

Відкрий [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Що покажемо замовнику
- Upload або вибір sample-фото
- Annotated результат з брендами
- Якість фото, доля полиці, грубий збіг зі схемою, missing / competitor
- Експорт Excel / JSON

## Стек (PoC)
Python + FastAPI + Ultralytics YOLO + **SigLIP2** (Hugging Face) brand matching + static web UI.  
Дані на диску (`samples/`, `catalog/`, `planograms/`, `out/`) — **без БД**.

Samples за замовчуванням аналізуються **live** (`use_cache=false`). Кеш лише якщо явно `?use_cache=true`.

## CLI

```bash
python -m poc.run --input samples/xo_fridge_01.jpg --output out/ --conf 0.15 --imgsz 640
```

Sample з кешем лише за потреби: `POST /api/analyze?sample=...&use_cache=true`.

Планограма per-photo:
- `planogram=auto` (default) — `planograms/<stem>.json` якщо є, інакше Missing не рахується
- `planogram=none` — тільки детекція
- `planogram=custom&expected=morshynska,coca-cola` — своя схема
- `planogram=xo_napoi_hero` — конкретний файл

## Структура
```
samples/              демо-фото
catalog/<brand>/      еталони брендів
planograms/demo.json  одна демо-схема ХО
poc/                  QA, detect, brand, shelves, planogram, report, api
web/                  UI
out/                  annotated + reports
```

## Статус
Реалізація за `docs/POC-PLAN.md` (варіант 1). TypeScript-прод — після демо.
