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
