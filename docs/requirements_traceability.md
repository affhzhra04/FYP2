# Requirements traceability

Sources: **U** = university project brief, **C** = collaborator (Innonics) expectation, **A** = AP1 panel feedback, **N** = non-functional.
Tests are listed in `docs/test_plan_and_results.md`. "Walkthrough" = manual check with a screenshot.

| ID | Src | Requirement | Implementation | Verification | Status |
|---|---|---|---|---|---|
| FR-01 | U | Real-time map with multiple vehicles and their routes | Live Monitor: map, tier-coloured markers, 60-minute trails, incident dots, replay clock | IT-03, ST-01 | Met (replay of recorded data; no live feed supplied) |
| FR-02 | U | Assess driver risk behaviour in real time | Seven detectors turned into events; live incident feed | UT-04..UT-09, IT-03, VAL-01 | Met |
| FR-03 | U | One Driver Safety Score per driver, updating in real time | DSS = 100 x exp(-R/R0) with memory half-life; live leaderboard | UT-10, IT-03, ST-01 | Met |
| FR-04 | U | User can tune the DSS formula (route deviations, unauthorized stops, speed) | Scoring Policy: one slider per factor, strictness, thresholds, preview of ranking change | IT-02, ST-02 | Met |
| FR-05 | C | Prepared vehicle telemetry to assess driving risk | `ingest.py`, `risk_engine.prepare` (session split, deltas, distances) | UT-01..UT-03, data description | Met |
| FR-06 | C | Dashboard for monitoring | Live Monitor, Driver Performance | ST-01, ST-08 | Met |
| FR-07 | C | Session history | History & Reports: filters, table, route and telemetry per trip | ST-03, ST-04 | Met |
| FR-08 | C | Reporting | CSV (sessions, incidents, queue, driver report), HTML and PDF trip report | IT-05, ST-05 | Met (PDF download confirmed, ST-05) |
| FR-09 | A | Correlate fuel drop with off-route behaviour | Compound event `fuel_off_route` (fuel loss within 15 min of a deviation or unknown stop) | UT-07, VAL-01 | Met (rule-based) |
| FR-10 | A | Add machine learning | Isolation Forest unusual-trip flag with reasons; Random Forest risk model; saved versions and retraining | IT-08..IT-11, VAL-02, ST-10 | Met |
| FR-11 | extra | Incident review workflow | Priority queue, decision and note saved per incident, alert history | UT-11..UT-13, ST-06 | Met (save confirmed, ST-06) |
| FR-12 | extra | Supervisor alerts | Alert message and Telegram dispatch; simulated if no token | UT-14..UT-16, ST-07 | Met (live Telegram delivery confirmed, ST-07; simulation when no token is set) |
| FR-13 | extra | Driver performance view | Ranking, weekly trend, behaviour mix, limited-data flag | UT-17..UT-18, ST-08 | Met |
| NFR-01 | N | Responsiveness | Load: read 0.6 s + detection 1.3 s; live refresh about 30 ms per tick | PERF-01 | Met |
| NFR-02 | N | Explainability | Per-trip score audit (points per event, formula), per-trip reasons for unusual flags | IT-01, ST-04 | Met |
| NFR-03 | N | Reliability and error handling | Friendly errors for missing/empty database, bad inputs rejected, dispatch never raises | UT-02, UT-03, IT-06, UT-12, UT-16 | Met |
| NFR-04 | N | Maintainability | Engine, data layer, ML, review, alerts, performance in separate modules; 133 automated tests | Test run | Met |
| NFR-05 | N | Data integrity | Reviews stored in their own database so re-ingestion never deletes them | UT-11 | Met |
| NFR-06 | N | Usability (stakeholder) | Manager-oriented wording and workflow | UAT-01..UAT-03 | Pending Innonics feedback |
