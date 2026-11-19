# AP1 panel feedback and how it was addressed

| # | Panel comment | What was done | Where to see it | Status |
|---|---|---|---|---|
| 1 | All objectives are design/develop; add evaluation | Objectives reworded (below) so one objective is to evaluate. Evaluation evidence: unit/integration tests, fuel-injection validation, leave-one-vehicle-out model validation, stakeholder review | `docs/test_plan_and_results.md`, `docs/results/`, `src/evaluation.py` | Evidence done; stakeholder review pending reply |
| 2 | Flowchart unclear (first diamond) | Redraw with labelled decisions: ping valid? -> parked or moving? -> each detector's yes/no branch -> event -> score update | Slides (system flow) | To redraw |
| 3 | Describe the dataset (public or private?) | Dataset description with source, counts, coverage, timestamp zone, limits | `docs/data_description.md` | Done |
| 4 | Too rule-based; add ML | Isolation Forest (unusual trips, with reasons) and Random Forest risk model validated by leave-one-vehicle-out; saved model versions and retraining in the product | `src/ml_layer.py`, System & Models, Incidents & Review > Unusual trips | Done |
| 5 | Correlate fuel drop with off-route / moonlighting in one penalty index | Fuel loss within 15 minutes of a route deviation or an unauthorized stop becomes a compound event with a higher base penalty (3.0 vs 2.0) inside the single DSS | `detect_all` and `correlate_fuel_with_route` in `src/risk_engine.py` | Done as a rule-based compound event, not a statistical correlation |
| 6 | Scope down and add more value and impact | Removed the 50 km Kuala Lumpur geofence and ghost-hours alerts (it produced 59% of the old alerts and does not fit a fleet that works across the peninsula); focused on risk scoring plus a manager workflow: incident review, driver performance, reports, alerts | Dashboard pages | Done |

## Objectives (proposed wording)
1. To design and develop a pipeline that turns raw Innonics GPS and fuel telemetry into driver-behaviour events (speeding, harsh driving, idling, route deviation, unauthorized stops, fuel loss).
2. To develop a real-time Driver Safety Score that combines those events and lets users tune the importance of each factor.
3. To develop a monitoring dashboard with a live multi-vehicle map, incident review, driver performance, trip history and reports for fleet managers.
4. To evaluate the system through detector validation, held-out testing of the machine-learning layer, and stakeholder review.
