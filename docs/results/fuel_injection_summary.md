# Fuel-loss detector: validation by injection

The telemetry has no labelled fuel theft, so ground truth was created: 60 real parked, ignition-off pings (first report
after the engine stops, ping within 15 minutes of the previous one) were chosen at random and the fuel level was lowered
by 8 L from that ping to the end of the session. The unchanged detector was then run on the modified data.

Command: `python scripts/validate_fuel_injection.py --seed N` (per-drop results are in `fuel_injection_seed<N>.csv`).

| Seed | Injected | Detected | Recall |
|---|---|---|---|
| 1 | 60 | 57 | 95% |
| 2 | 60 | 56 | 93% |
| 3 | 60 | 59 | 98% |
| 7 | 60 | 53 | 88% |

**Mean recall 94% (range 88-98%).** The number of fuel events after injection exceeded the number of detected injections
by only -2 to +2 events, so the injection did not create meaningful extra alerts.

## Scope and limits (state these when presenting)
- The detector can only time-stamp a drop when a ping arrives within 15 minutes of the previous one. Parked vehicles
  otherwise report once an hour. Across those hourly gaps the fuel reading moves by several litres in both directions
  (12,325 parked pings with 900-7,200 s gaps: drops of 3 L or more occur 522 times, rises of 3 L or more 491 times),
  so drops there cannot be separated from sensor noise or refuelling. The recall above therefore applies to drops around
  engine-off moments, not to siphoning during a long unattended stop.
- Missed injections are mostly cases where the next report did not stay lower (persistence check) or came after the session ended.
- Synthetic ground truth only: this measures detection, not how often real theft happens.
