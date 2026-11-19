# Driver Risk Monitor

Real-time multi-driver risk assessment from vehicle telemetry, built for fleet managers at Innonics Sdn. Bhd.
Final-year project WIA3003 (Universiti Malaya), Afifah Zahra binti Zulkifli (22003045), supervisor Dr. Liew Wei Shiung.

Every driver gets one **Driver Safety Score (DSS)** from 0 to 100 that updates as incidents happen. Managers can change
how much each behaviour counts, review incidents, follow drivers over time, and export reports.

## What it does
| Page | Purpose |
|---|---|
| Live Monitor | Multi-vehicle map, live DSS per driver, incident feed (recorded telemetry replayed as a live stream) |
| Incidents & Review | Priority queue, decision and note per incident, supervisor alerts, unusual trips flagged by the model |
| Driver Performance | Ranking, weekly trend, what drives each driver's risk, lowest-scoring trips |
| Scoring Policy | Set the weight of each behaviour (speeding, harsh driving, idling, route deviation, unauthorized stop, fuel loss) and preview the effect on the ranking |
| History & Reports | Trip history with filters, route and telemetry per trip, score audit, CSV / HTML / PDF reports |
| System & Models | Data coverage and quality, risk-model versions with retraining, alert gateway status |

## How the score works
Telemetry is turned into **events** (a run of speeding is one event, not hundreds of rows): speeding, harsh driving,
idling, route deviation from a learned per-vehicle corridor, unauthorized stop, fuel loss while parked, and fuel loss
while off-route or at an unknown stop. Each event adds weighted risk points. Live score:
`DSS = 100 x exp(-R / R0)` with older incidents fading by a memory half-life. Trip score uses risk per 100 km driven.
Details and validation: `docs/`.

## Setup (Windows, from the project folder)
```
pip install streamlit plotly pandas numpy reportlab scikit-learn scipy requests python-dotenv
python -m src.ingest --data-dir "<folder with the Innonics JSON files>"
streamlit run src/dashboard.py
```
The data is private and is not in this repository. Optional alerts: copy `.env.example` to `.env` and set
`TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`; without them alerts run in logged simulation mode.

## Tests and validation
```
python -m pytest tests -q                       # 32 automated tests
python scripts/validate_fuel_injection.py       # fuel-loss detector recall on injected drops
streamlit run src/evaluation.py                 # project view: model validation and method (for the panel)
```
Results and test IDs: `docs/test_plan_and_results.md`. Requirement coverage: `docs/requirements_traceability.md`.
AP1 feedback log: `docs/ap1_feedback_response.md`. Dataset: `docs/data_description.md`.

## Project layout
```
src/
  dashboard.py        product app (6 pages)          evaluation.py   panel view (not for end users)
  risk_engine.py      detectors, events, DSS         dash_core.py    data layer, replay, audits, reports
  ml_layer.py         features, models, versions     performance.py  driver ranking and trends
  review_store.py     review decisions, alert log    alerts.py       priorities, messages, Telegram dispatch
  ingest.py, preprocessing.py                        JSON -> SQLite
database/             telematics_core.db, reviews.db, models/
tests/  scripts/  docs/
legacy (to archive):  app.py, views/, detectors.py, scoring.py   earlier prototype, not used by the dashboard
```

## Limitations
- The live view replays recorded data; Innonics supplied no live feed.
- No crash, claim or theft labels exist. Detectors are checked with synthetic tests and fuel-drop injection (recall 88-98%).
  The risk model learns the rule-based score and is validated on vehicles it has not seen (within one tier for 90% of trips, exact tier 38%).
- The fuel detector sees drops only around engine-off moments; across hourly parked reports the sensor is too noisy to separate theft from noise.
- Two vehicles supply most driving days. Drivers with fewer than five driving days are marked "Limited data".
