# Test plan and results

Run: `python -m pytest tests -q` from the project root. Last full run (21 Nov 2026): **133 passed, 0 failed**. This includes 33 tests for the current engine, data layer,
ML layer, review and alerts (`test_risk_engine.py` 11, `test_dash_core.py` 8, `test_ml_layer.py` 6,
`test_review_alerts_performance.py` 8) and the earlier tests of the original modules. `tests/conftest.py` blanks the
Telegram token during tests, so tests never send real messages.
Tests use small synthetic fleets with known answers, so expected values are exact. The earlier tests of the legacy
Enterprise Fleet app (`app.py`, `views/`) are outside this scope.

## 1. Unit tests (one function, known input, known output)
| ID | Test | Input / expected | Result |
|---|---|---|---|
| UT-01 | `test_session_vehicle_split` | id `8_9_2` -> session `8_9_2`, vehicle `2` | Pass |
| UT-02 | `test_missing_columns_raises` | frame without required columns -> ValueError | Pass |
| UT-03 | `test_empty_raises` | empty frame -> ValueError | Pass |
| UT-04 | `test_speeding_rows_collapse_to_one_event` | many consecutive rows over limit -> one event | Pass |
| UT-05 | `test_harsh_requires_dense_pings` | large speed change over a long gap -> no harsh event | Pass |
| UT-06 | `test_idling_needs_five_minutes` | idle under 5 min -> none; over -> one | Pass |
| UT-07 | `test_fuel_theft_stationary_persistent_only` | drop while parked and persisting -> event; moving or non-persistent -> none | Pass |
| UT-08 | `test_fuel_theft_ignores_long_gap` | drop across a gap over 15 min -> none | Pass |
| UT-09 | `test_clean_vehicle_scores_100` | no events -> DSS 100 | Pass |
| UT-10 | `test_dss_bounds_monotonic_and_recovery` | DSS in (0,100], more risk -> lower, older events recover, weights matter | Pass |
| UT-11 | `test_review_save_update_and_reload` | second save updates, no duplicate; stable incident key | Pass |
| UT-12 | `test_review_rejects_bad_input` | unknown status or type -> ValueError | Pass |
| UT-13 | `test_alert_log_roundtrip` | logged alert is read back | Pass |
| UT-14 | `test_priority_rules` | fuel loss and speeding 15 km/h over -> High; idling -> Low | Pass |
| UT-15 | `test_alert_text_is_markdown_safe` | no `_` or `*` in the message | Pass |
| UT-16 | `test_dispatch_never_raises` | gateway failure returns a mode and status | Pass |
| UT-17 | `test_driver_table_ranks_and_flags_limited_data` | worst first, trips under 50 km excluded, "Limited data" flag | Pass |
| UT-18 | `test_trend_compares_last_two_months_with_data` | month-on-month change of 30 points | Pass |
| UT-19 | `test_features_one_row_per_session_and_finite` | one finite feature row per trip | Pass |
| UT-20 | `test_aggressive_sessions_have_higher_speed_features` | aggressive trip has higher p99 speed and fast share | Pass |
| UT-21 | `test_parking_is_not_an_unauthorized_stop` | stop of 2 h -> one event; stop of 8 h or a vehicle-day with no movement -> none | Pass |
| UT-22 | `tests/test_incident_analysis.py` (11 tests) | counts by type, hour, weekday and week; rate per 1,000 km (zero km safe); points match the score formula; review confirm rate; thresholds in rule text; no-incident case | Pass (re-run on laptop to confirm) |

## 2. Integration tests (modules working together)
| ID | Test | Checks | Result |
|---|---|---|---|
| IT-01 | `test_speeding_session_scores_lower_and_audit_matches` | engine -> session score -> audit total agree | Pass |
| IT-02 | `test_weight_zero_removes_penalty` | policy weight 0 removes that behaviour's penalty | Pass |
| IT-03 | `test_live_state_updates_with_time` | replay clock advance changes events and leaderboard | Pass |
| IT-04 | `test_bundle_sessions_and_days` | detectors -> sessions -> replay days | Pass |
| IT-05 | `test_reports_render` | HTML and PDF reports render | Pass |
| IT-06 | `test_unreadable_database_gives_friendly_error` | missing database -> clear message | Pass |
| IT-07 | `test_session_score_is_normalised_per_100km`, `test_audit_risk_equals_raw_over_exposure` | per-100 km scoring consistent across engine and audit | Pass |
| IT-08 | `test_run_ml_leave_one_vehicle_out_beats_baseline` | Random Forest beats median baseline on held-out vehicles | Pass |
| IT-09 | `test_isolation_forest_scores_percentiles` | anomaly percentile in 0-100 | Pass |
| IT-10 | `test_model_registry_train_save_load_and_rollback` | versions 1 and 2 saved, rollback works, saved model scores trips with reasons | Pass |
| IT-11 | `test_training_needs_enough_driving_trips` | too little data -> clear error | Pass |

## 3. Validation on real data
| ID | Check | Result |
|---|---|---|
| VAL-01 | Fuel-loss detector, 60 injected 8 L drops, 4 seeds (`scripts/validate_fuel_injection.py`) | Recall 88-98%, mean 94%; no meaningful extra alerts. Scope and limits in `docs/results/fuel_injection_summary.md` |
| VAL-02 | Leave-one-vehicle-out validation of the risk model on 206 driving trips (System & Models, v3 trained 20 Nov 2026 after the parking fix) | Within one tier 91% (baseline 50%); rank agreement 0.43; average score error 15.1 points. Earlier v1: 90%, 0.45, 14.9. Labels are rule-derived |
| VAL-03 | Isolation Forest agreement with rules (v1, before the parking fix) | Spearman 0.29; 20% overlap of top 10%; 5 trips flagged that the rules rate Excellent or Low. Re-run pending for v3 |
| VAL-04 | Stop-detector audit (`scripts/audit_suspicious_sessions.py`) | 57 of 692 sessions flagged before the fix (0 km days scoring High Risk); 45 after stops of 6 h or more and no-movement days were excluded. The remaining 45 are to be reviewed with Innonics |

## 4. System tests (browser walkthrough, `streamlit run src/dashboard.py`)
| ID | Steps | Expected | Result |
|---|---|---|---|
| ST-01 | Live Monitor, day 27 Aug 2024, press Play | Markers move, scores and incident feed update | Pass (screenshot) |
| ST-02 | Scoring Policy, change Speeding, Idling, Route weights | Blue bars differ from grey; ranks update | Pass (screenshot) |
| ST-03 | History & Reports, filter and open trip 1_29_9 | Table, route map, speed and fuel charts | Pass (screenshot) |
| ST-04 | Same trip, Score audit tab | Points per event, total, tier | Pass (screenshot) |
| ST-05 | Download HTML and PDF report | Files open with the same score | Pass (screenshot, session 1_29_9) |
| ST-06 | Incidents & Review: pick incident, set Confirmed with note, Save, reload page | Status and note persist; counters change | Pass (screenshot) |
| ST-07 | Supervisor alert | Message previewed; logged; sent by Telegram when configured, simulated if not | Pass: LIVE_API, DELIVERED on 20 Nov 2026 04:33 MYT (screenshots of the page and the phone) |
| ST-08 | Driver Performance | Trend, ranking, behaviour mix, limited-data flags | Pass (screenshot) |
| ST-09 | Delete the database, start the app | Friendly message, no crash | Pass (screenshot, 21 Nov 2026: "Database not found ... Run the ingestion step first", no traceback) |
| ST-10 | System & Models: train new version, switch version | Version table updates; Unusual trips shows the active version | Pass (screenshot: v1, v2, v3) |
| ST-11 | Incident Analysis: change vehicle, date range and incident-type filters on each tab | KPIs, charts and tables update; empty filter shows a message, no error | Pass (screenshots, 21 Nov 2026: filtered view of 137 incidents; empty selection shows "No incidents match these filters.") |

## 5. Performance (PERF-01, 692 trips)
Read database 0.6 s; run all detectors 1.3 s; session scoring 0.02 s; one live refresh about 30 ms (mean of five). First
model training about 30 s, afterwards the saved model scores trips in about a second.

## 6. User acceptance (UAT with Innonics)
Awaiting Innonics. Questions: (UAT-01) do the dashboard, history and reports meet expectations; (UAT-02)
what should change; (UAT-03) are the risk categories meaningful. Replies to be pasted here when received.
