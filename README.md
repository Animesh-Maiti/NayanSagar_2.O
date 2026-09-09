# NayanSagar 2.o

NayanSagar 2.o is a hydrographic side-scan sonar hazard detection platform for submerged hazard localization, segmentation, replay, and engineering reporting.

## Architecture

```text
NayanSagar_2.o/
├── app/                      # UI, telemetry parsers, routing, and report consumers
│   ├── core/                 # tiling, geometry, physics, reporting
│   ├── telemetry/           # XTF readers and synthetic sonar stream utilities
│   └── static/samples/      # Sample data and dashboard fixtures
├── ai_engine/                # Decoupled inference runtime and postprocessing
│   └── weights/              # model checkpoints (best.pt ignored by git)
├── training_scripts/        # Data preparation, training, and label processing scripts
└── data/                     # Local rights-managed sonar, image, and ground-truth assets
```

## Quickstart

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app/main.py
```

## Team Workflow

Rule: Teammates must implement UI features and parsers inside `app/`. Do not touch `ai_engine/` or `training_scripts/`.

This decoupled architecture keeps the AI engine and training pipelines isolated from the application implementation.

