# Road Health Index Dashboard

Self-contained FastAPI dashboard for the existing Road RSL Prediction models. It only reads `../data` and `../models`; it does not edit project files or artifacts.

## Run

From this folder, create/activate a virtual environment, install requirements, then start the server:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. API documentation is available at <http://127.0.0.1:8000/docs>.

## REST endpoints

- `GET /api/sections?search=0101` — section search
- `GET /api/section/{shrp_id}?state_code={state}` — historical records, defaults, and prediction
- `GET /api/network-summary` — condition distribution
- `POST /api/predict` — live what-if prediction
- `POST /api/report.csv` — download a prediction report

The UI makes a formatted PDF client-side. MySQL is not required for the local dashboard because it reads the existing Excel source data. Add a database adapter only when the project needs persistent user accounts, saved scenarios, or multi-user deployment.
