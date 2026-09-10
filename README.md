# NayanSagar 2.o

An AI-driven side-scan sonar hydrographic survey platform engineered for real-time acoustic anomaly detection, submerged hazard segmentation, physics-informed acoustic shadow verification, and IHO S-44 compliant clearance reporting.

---

## System Overview

NayanSagar processes raw side-scan sonar telemetry (`.xtf` binary packages and high-resolution acoustic waterfall images) to detect, classify, and verify submerged navigation hazards (e.g., shipwrecks, aircraft debris, structural obstructions) while filtering geomorphological false positives like periodic sand ripples.

### Key Capabilities

* **Dual-Stage Detection Engine:**
  * **Supervised Segmentation:** Uses YOLO11s-seg to isolate bounding boxes and structural contours for cataloged hazards (`shipwreck`, `aircraft_wreck`, `debris_highlight`).
  * **Unsupervised Anomaly Detector:** Employs a convolutional autoencoder evaluating Mean Squared Error (MSE) reconstruction loss to flag novel, uncataloged seafloor hazards.
* **Physics-Informed Acoustic Shadow Verification:**
  * Implements triangular acoustic shadow geometry to calculate target relief height ($H$):
    $$H = \frac{L_{\text{shadow}} \cdot H_{\text{tow}}}{R_{\text{target}}}$$
  * Automatically classifies detected candidates into rigorous operational tiers:
    * `VERIFIED_HAZARD`: Promoted when acoustic shadow relief satisfies $H \ge 0.25\text{ m}$.
    * `PROBABLE_HAZARD`: High-confidence detections ($\ge 0.60$) where acoustic shadows are obscured or clipped.
    * `UNVERIFIED_CLUTTER`: Low-relief acoustic reverberations, seabed clutter, or artifacts lacking physical height ($H = 0\text{ m}$).
* **Topographic & Geomorphology Filtering:**
  * Aspect-ratio and wide-area bounding box gates ($> 550\text{ px}$) suppress periodic sand dunes and seabed ripple patterns that mimic structural ribs.
  * Nadir-boundary ringing filters suppress water column transition artifacts lacking acoustic shadows.
* **Telemetry & Georeferencing Pipeline:**
  * Parses binary eXtended Triton Format (`.xtf`) ping records using `pyxtf`.
  * Normalizes acoustic returns via Time-Varied Gain (TVG), bilateral filtering, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
  * Remaps slant range to ground range to remove nadir distortion.
  * Decodes navigation coordinates (`SensorXcoordinate`, `SensorYcoordinate`) with sanitization and coordinate clamping to eliminate Deck.gl/pydeck projection crashes.
* Reporting:
  * Generates survey inspection logs exportable to **CSV** and vector **GeoJSON** (`Point([Longitude, Latitude])`).

---

## Repository Structure

```text
NayanSagar_2.o/
├── app/
│   ├── core/
│   │   ├── physics.py             # Geometric shadow relief height & verification logic
│   │   ├── reporter.py            # IHO S-44 CSV/GeoJSON audit log generators
│   │   └── tiler.py               # Dynamic 640x640 sliding-window tiling & overlap handling
│   ├── telemetry/
│   │   ├── xtf_reader.py          # Raw .xtf parsing, TVG, CLAHE, & coordinate decoding
│   │   └── synthetic_stream.py    # Fallback and telemetry simulation fixtures
│   ├── static/                    # Dashboard styles, sample images, and offline test assets
│   ├── config.py                  # Operational defaults, thresholds, and sensor baselines
│   └── main.py                    # Multi-tab Streamlit console with guarded session lifecycle
├── ai_engine/
│   ├── infer.py                   # Dual-stage inference orchestration & tensor normalization
│   ├── postprocess.py             # Global coordinate clustering and duplicate suppression
│   └── weights/                   # YOLO & autoencoder checkpoints (best.pt git-ignored)
├── training_scripts/              # Dataset preparation, augmentation, and model training routines
├── data/                          # Local rights-managed sonar recordings & ground-truth assets
├── run_app.bat                    # One-click Windows execution launcher
├── requirements.txt               # Locked Python dependencies
└── README.md
