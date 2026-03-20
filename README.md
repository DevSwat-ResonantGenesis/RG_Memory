# RG Memory

> **Part of the [ResonantGenesis](https://dev-swat.com) platform** — Core memory engine with Hash Sphere, RAG, and embeddings.

[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen.svg)]()
[![License: RG Source Available](https://img.shields.io/badge/License-RG%20Source%20Available-blue.svg)](LICENSE.txt)

## Features
- Hash Sphere coordinate system with ResonanceHasher PCA
- Embedding generation and vector search
- RAG (Retrieval-Augmented Generation) for context injection
- Memory clustering (Alpha-Zeta layers)
- Memory ingestion from all platform services
- DSID (Digital Soul Identity) creation

## Quick Start
```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deployment
- **Container**: `memory_service` | **Port**: 8000
- **Server path**: `/home/deploy/RG_Memory`

---
**Organization**: [DevSwat-ResonantGenesis](https://github.com/DevSwat-ResonantGenesis) | **Platform**: [dev-swat.com](https://dev-swat.com)
