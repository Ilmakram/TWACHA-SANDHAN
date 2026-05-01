# Dermatological-Manifestation-SIH

This project is being developed by our team as part of the Smart BU internal hackathon being conducted at our university to shortlist projects for the Smart India Hackathon 2023. The project addresses problem statement number 1344: AI-based tool for preliminary diagnosis of Dermatological manifestations.

## Proposed Approach

Our current proposed approach and pitch will consist of a way to handle three key challenges addressed in the problem statement:

1. Lack of Screening technology due to poor infrastructure.
2. Lack of up-to-date diagnostic tools due to poor connectivity.
3. Preliminary screening being difficult in remote areas.

Our proposed approach, therefore involves two components:

- An AI-based smart device that will rely on on-board hardware and software for function, but can be remotely updated and monitored to ensure peak functionality. This solves the problems as follows:
- A backend monitoring service that can generate synthetic data, train and update models, monitor models and try to find shortcomings over long term.

This solves the three key challenges in the following way:

1. Lack of screening technology due to poor infrastructure is mitigated because preliminary screening through a single, small image processing device should prove helpful for places that do not have anything.
2. Poor connectivity is mitigated by the fact that the smart device does not need to be online at all times; Just enough for monitoring data to be uploaded and for any updates to be downloaded.
3. This can be due to lack of good image data or transportation being a hassle. Both are mitigaed to some extent by modern MLOps and Synethetic data technology.

## Deployment Notes

Do not commit local Python environments, datasets, or trained model binaries to Git. Recreate the backend environment with:

```bash
cd backend1
python -m venv tf_env
tf_env\Scripts\activate
pip install -r requirements.txt
```

The API currently expects the trained model and label map at:

- `backend1/output/best_model.h5`
- `backend1/output/label_map.json`

Those files are intentionally ignored because the model is too large for normal GitHub pushes. Store them in external storage, release assets, or Git LFS if your deployment platform supports it.

## Deploying the Backend

The backend is a FastAPI service in `backend1/app.py`. It can be deployed separately from the static frontend.

### Recommended backend host
- Render, Railway, Azure Web App, PythonAnywhere, or Fly.io

### Render-specific deployment
- A `render.yaml` file is included at the repository root.
- Render will deploy the backend service from `backend1` using `uvicorn`.
- Render must use Python 3.11 for TensorFlow compatibility.
- You can also create the service manually in Render and set:
  - Root directory: `backend1`
  - Build command: `python -m pip install --upgrade pip && pip install -r requirements.txt`
  - Start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- A `backend1/runtime.txt` file is included to force Python 3.11.

### Required files
- `backend1/requirements.txt`
- `backend1/app.py`
- `backend1/output/best_model.h5`
- `backend1/output/label_map.json`

### Start command

For Render/Railway, use:

```bash
cd backend1
uvicorn app:app --host 0.0.0.0 --port $PORT
```

### If you want a simple deploy helper
Create a `backend1/Procfile` with:

```text
web: uvicorn app:app --host 0.0.0.0 --port $PORT
```

## Connecting the Frontend

The frontend currently uses `assets/js/detect.js` to call the backend. It now supports overriding the backend host with `window.BACKEND_URL`.

In your deployed frontend HTML, add this before the script that loads `detect.js`:

```html
<script>
  window.BACKEND_URL = "https://your-backend-url.example.com";
</script>
```

Then deploy the frontend to Vercel or any static host.

## Notes

- Your backend currently also supports a simple `/chat` endpoint so chat on the frontend will not fail.
- Keep `backend1/output/` files available to the backend service during deployment.
