# SIH26059 — Antarctic Sea-Ice, Iceberg Trajectory, and Navigation Decision Support System

Core Python backend algorithms for Smart India Hackathon 2026 problem statement **SIH26059** (Ministry of Earth Sciences, MoES).

## Architecture Overview

```text
antarctic-nav-ai/
├── data/              # Raw, processed, and external datasets
├── notebooks/         # Analysis and validation notebooks
├── src/
│   ├── data/          # Environmental data loaders and unified abstraction
│   ├── models/        # Iceberg dynamics and physics models
│   ├── forecasting/   # Sea ice spatiotemporal forecasting
│   ├── routing/       # Polar grid, vessel profile, cost, and A* pathfinding
│   └── evaluation/    # Geodesic metrics and validation
├── tests/             # Pytest test suite using synthetic fixtures
├── configs/           # YAML / dictionary configurations
├── requirements.txt   # Virtual environment dependencies
└── README.md
```

## Setup & Testing

```bash
# Create virtual environment and install dependencies
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Run test suite
pytest tests/ -v
```
