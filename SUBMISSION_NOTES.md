# Submission Notes: Case 4 Churn Detective

## What's Submitted

- `notebooks/churn_analysis.ipynb` which is the full end-to-end analysis covering EDA, modelling, SHAP interpretability, segmentation and retention recommendations
- `app.py` which is the four-page Streamlit dashboard with Overview, EDA Explorer, Predict and Segments pages
- `outputs/` containing the saved model, feature names, segment CSV and all figures
- `DECISIONS.md` covering assumptions, modelling choices and AI disclosure

---

## What Broke During Deployment

### Problem 1: HuggingFace Spaces rejected binary files via git

HF Spaces now uses Xet storage for binary files like `.pkl` and `.png`. A regular git push was rejected with:

```
Your push was rejected because it contains binary files.
Please use https://huggingface.co/docs/hub/xet to store binary files.
```

How I would fix it cleanly: configure `.gitattributes` with `filter=xet` for binary extensions and install the `hf_xet` hooks before pushing. During the submission window I worked around it using the `huggingface_hub` Python API with `api.upload_folder()` which handles binary uploads directly without going through git.

### Problem 2: HF Spaces showed "Welcome to Streamlit" instead of the app

The default HuggingFace Docker template points to `src/streamlit_app.py` and exposes port 8501. My app lives at `app.py` in the repo root and HF Spaces requires port 7860.

I replaced the default Dockerfile entirely with one that copies the right files and runs on the right port:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY app.py .
COPY data/ ./data/
COPY outputs/ ./outputs/

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
```

### Problem 3: HF token authentication was failing

The first push attempt failed with "not authorized" because the access token I had only had Read permission. I regenerated a new token with Write permissions in HF account settings and that fixed it.

### Problem 4: Everything kept collapsing so I uploaded manually

After all the git and API upload attempts kept breaking I just went into the HuggingFace Spaces Files tab and uploaded the files manually through the UI. Not the cleanest way to do it but it got the app live and that was the priority at that point.

---

## If the Live URL Is Down

The app runs correctly locally. To check:

```bash
uv run streamlit run app.py
```

All four pages load fine. Overview shows the KPI cards and SHAP beeswarm. EDA Explorer has the interactive charts. Predict takes slider inputs and returns a live churn probability with a SHAP waterfall. Segments shows the cluster scatter plot and the retention playbook.

---

## Known Limitations

- The model was trained on synthetic data so real-world recalibration would be needed before acting on the revenue projections.
- Segment separation on price is weak because all three clusters average around $71 per month. The differentiator is behaviour not spend level.
- There is no causal inference. High support calls flag churn risk but fixing the support team does not guarantee retention if the underlying network quality is the actual root cause.
