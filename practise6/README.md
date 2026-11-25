# Practical Task 6: Deploy an ML Model with FastAPI and Docker

This project trains a Logistic Regression model on the Statlog German Credit dataset and serves its predictions through a FastAPI application. The notebook `practise6.ipynb` explains the work in stages, following the practical-task style used in the earlier practices and SISs.

## Project files

- `train.py` — prepares the data, trains the model, reports test metrics, and saves the fitted pipeline.
- `main.py` — FastAPI app with a health endpoint and a prediction endpoint.
- `german.data` — dataset used by the training script.
- `model.joblib` — saved preprocessing and Logistic Regression pipeline, created by `train.py`.
- `training_metrics.json` — test-set metrics written by `train.py`.
- `Dockerfile` — instructions for building and running the API container.

## 1. Train the model

From this directory, create and activate a virtual environment, then install the dependencies:

```bash
python3.11 -m venv .venv       # Windows: py -3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Train the model and write `model.joblib`:

```bash
python train.py
```

The target is mapped so `0` means **Good Risk** and `1` means **Bad Risk**. The preprocessing and classifier are saved together in one pipeline so the API applies the same transformations used during training.

## 2. Run the API locally

Start Uvicorn from this directory:

```bash
uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to check the root response and [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the interactive API documentation. The `POST /predict` form includes a complete example request.

The API accepts all 20 German Credit input features and returns a class code and label, for example:

```json
{"prediction": 0, "risk_label": "Good Risk"}
```

## 3. Build and run with Docker

Build the image from this directory. The Dockerfile trains the model during the image build with the pinned dependencies, so the saved artifact matches the container's scikit-learn version:

```bash
docker build -t german-credit-api .
```

Run the container and publish its port:

```bash
docker run --rm -p 8000:8000 german-credit-api
```

Then check `http://127.0.0.1:8000`, submit the example request at `/docs`, and confirm the Swagger page opens at `http://127.0.0.1:8000/docs`.

## Notes

The fixed stratified 80/20 split uses random state 42 for reproducibility. Reported metrics are a classroom demonstration on a small historical dataset; they are not evidence that this model is suitable for real lending decisions. Unknown categorical values are accepted by the encoder, but the model's output should only be interpreted for inputs similar to the training data.
