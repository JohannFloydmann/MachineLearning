# SIS-3: ML System Extension

This project extends Practical Task 6 with a Streamlit frontend and MLflow experiment tracking/model registration. It uses the existing Logistic Regression pipeline trained on the Statlog German Credit dataset, served by FastAPI and containerized with Docker.

## Project files

- `train.py` — trains and evaluates the pipeline, saves `model.joblib` and `training_metrics.json`, and logs/registers the model in MLflow.
- `main.py` — FastAPI health and prediction endpoints.
- `streamlit_app.py` — user form that calls the FastAPI `/predict` endpoint. Category codes stay in the API payload; users see descriptive labels.
- `docker-compose.yml` — starts the API and Streamlit frontend.
- `requirements.txt` — API and frontend dependencies used by Docker.
- `requirements-mlflow.txt` — app and MLflow dependencies for local training/tracking.
- `german.data` — Statlog German Credit dataset.
- `model.joblib` — saved preprocessing and Logistic Regression pipeline used by the API container.

## 1. Train, track, and register the model

If you have not created the Conda environment yet, create it once. Then activate it and install the training dependencies:

```bash
conda create -n sis3-ml python=3.11 -y
conda activate sis3-ml
python -m pip install -r requirements-mlflow.txt
```

In one terminal, start the local MLflow tracking server from this directory:

```bash
conda activate sis3-ml
mlflow server --port 5000
```

In a second terminal, activate the same environment, change to this directory, and train:

```bash
conda activate sis3-ml
python train.py
```

The script creates or reuses the **German Credit Risk** experiment. It logs Logistic Regression parameters, accuracy/precision/recall/F1 metrics, the `model.joblib` file, and an MLflow model artifact. It registers the model as **GermanCreditRiskModel**; the first successful run creates version 1 and each later training run creates the next version.

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) to inspect the experiment and its registered model. The script defaults to `http://127.0.0.1:5000`; set `MLFLOW_TRACKING_URI` if the tracking server is elsewhere.

## 2. Run the API locally

With dependencies installed and `model.joblib` available, run:

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try `POST /predict`. The API accepts all 20 German Credit features and returns a class code and label, for example:

```json
{"prediction": 0, "risk_label": "Good Risk"}
```

## 3. Run the API and frontend with Docker Compose

From this directory, build and start both services:

```bash
docker compose up --build
```

Open [http://127.0.0.1:8501](http://127.0.0.1:8501) for the form and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the API. The MLflow UI is at [http://127.0.0.1:5000](http://127.0.0.1:5000) while the Conda MLflow server from Section 1 is running. Stop the containers with `Ctrl+C`, then run `docker compose down`.

The Docker image copies the saved `model.joblib`; it does not train during image build because training logs to the separately running MLflow server. After retraining, rebuild the image if you want the API container to use the updated model file.

## Notes

The fixed stratified 80/20 split uses random state 42 for reproducibility. The reported metrics are a classroom demonstration on a small historical dataset; they are not evidence that this model is suitable for real lending decisions. Unknown categorical values are accepted by the encoder, but predictions should only be interpreted for inputs similar to the training data.
