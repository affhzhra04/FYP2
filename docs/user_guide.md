# Driver Risk Monitor: User Guide

Real-time multi-driver risk assessment with spatiotemporal anomaly detection and economic integrity monitoring.
Prepared for Innonics Sdn. Bhd. fleet supervisors and managers. All times are Malaysia time.

---

## 1. What the system does

The system reads vehicle telematics (GPS position, speed, fuel level, ignition) and does four things:

1. **Detects incidents**: speeding, harsh driving, excessive idling, route deviation, unauthorized stops, fuel loss, and fuel loss during an off-route event.
2. **Scores every driver** from 0 to 100 (Driver Safety Score, DSS). Higher is safer.
3. **Queues incidents for review**, so a supervisor can record a decision (Confirmed, False alarm, Follow-up needed) with findings.
4. **Flags unusual trips** that the rules alone would not catch, using a machine-learning model, and **alerts supervisors** through Telegram.

Every score can be traced back to the incidents that produced it.

---

## 2. First-time setup

### 2.1 Requirements
- Python 3.10 or later
- The Innonics JSON telemetry files in one folder

### 2.2 Install
Open a terminal in the project folder:

```
pip install -r requirements.txt
```

If `requirements.txt` is missing a package, install it directly:

```
pip install streamlit plotly pandas numpy reportlab scikit-learn scipy requests python-dotenv
```

### 2.3 Load the data (once)
```
python -m src.ingest --data-dir "<folder with the JSON files>"
```

This reads every file, removes invalid or duplicate readings, and stores the cleaned data in `database/telematics_core.db`. It prints an ingestion report (files read, readings kept, readings removed and why).

Options: `--max-files N` (try a small sample), `--skip-db` (dry run, nothing saved), `--db-path <file>` (use another database).

**Important: run ingestion once.** Running it again appends the same readings a second time. The dashboard ignores exact duplicates, so scores are not affected, but the database grows. If you must re-ingest, back up or delete `database/telematics_core.db` first. Reviews are stored separately in `database/reviews.db` and are never touched by ingestion.

### 2.4 Start the dashboard
```
streamlit run src/dashboard.py
```
The dashboard opens in your browser (normally http://localhost:8501).

The first time the Incidents & Review page or the History & Reports page is opened, the risk model is trained automatically (about a minute). It is saved and reused afterwards.

### 2.5 Optional: Telegram alerts
See section 9.

### 2.6 Optional: project/evaluation view
```
streamlit run src/evaluation.py
```
Shows model validation and the method. Intended for the project panel, not daily operations.

---

## 3. Screen layout

The **left sidebar** is on every page and holds two things:

**Page menu** (top): Live Monitor, Incidents & Review, Incident Analysis, Driver Performance, Scoring Policy, History & Reports, System & Models.

**Scoring policy and detection settings** (below the menu). They apply to every page, so changing them re-scores the whole fleet.

| Control | What it does | Default |
|---|---|---|
| Behaviour weights (one slider per incident type, 0 to 5) | How many risk points each type of incident carries. Higher means more penalty. | Fuel loss 5, others per policy |
| Score half-life (hours) | Live Monitor only. Time for an incident's effect on a driver's score to halve. Shorter means scores recover faster. | 6 |
| Score calibration > Scoring strictness | Lower is stricter (same incidents cost more points); higher is more lenient. | 10 |
| Reset to defaults | Restores weights, half-life and strictness. | |
| Detection thresholds: Speed limit | Speed above which speeding is counted. | 90 km/h |
| Detection thresholds: Idling | Minutes stationary with engine on before idling is counted. | 5 min |
| Detection thresholds: Unauthorized stop | Minutes stopped away from known locations before it is counted. | 30 min |
| Detection thresholds: Fuel loss | Litres of unexplained fuel drop before it is counted. | 1.0 L |

Changing a detection threshold re-runs the detectors (a few seconds). Settings are not saved between browser sessions; use Reset to defaults to return to the standard policy.

---

## 4. Pages

### 4.1 Live Monitor
**Purpose:** see the fleet as it operates. Recorded telemetry is replayed in time order to simulate a live feed.

**How to use**
1. Choose the day to replay.
2. Press play, or drag the **Replay time** slider to any moment of that day.
3. Read the map: marker colour shows each vehicle's current risk tier; the grey line is the last 60 minutes of route; red markers are recent incidents.
4. Read the **Driver Safety Score (DSS)** list for each driver's current score and tier.
5. Read the **Incident feed** for incidents as they occur.
6. Pick a vehicle to focus on to see its last 3 hours of telemetry (speed and fuel).

**Use it for:** a quick look at who needs attention right now.

### 4.2 Incidents & Review
**Purpose:** work through detected incidents and record a decision.

**Tab: Incident queue**
1. Filter the list by priority, type, vehicle or status (Open by default shows what still needs a decision).
2. Select an incident under **Incident review**. The map shows the location; the details show vehicle, time and priority.
3. Record the decision: **Confirmed**, **False alarm** or **Follow-up needed**. Enter your name under *Reviewed by* and your findings under *Review notes (findings)*. Save.
4. For serious cases use **Supervisor alert > Send alert (Telegram)** to notify the supervisor. Each sent alert is logged underneath.
5. **Download queue (CSV)** exports the filtered list.

**Priority rules**
- **High:** any fuel loss; speeding 15 km/h or more over the limit; an unauthorized stop of 2 hours or more.
- **Medium:** other speeding, harsh driving, route deviation, shorter unauthorized stops.
- **Low:** idling.

**Tab: Unusual trips**
Trips whose overall pattern differs from the rest of the fleet (the top 10% by the anomaly model), each with a reason (for example unusual speed variation or unusual stop pattern). Select a trip to see the route and its rule-based score. An unusual trip is **a prompt to look**, not a finding of wrongdoing.

### 4.3 Incident Analysis
**Purpose:** understand patterns across all incidents, for planning and coaching rather than single cases.

Use the filters at the top (vehicles, date range, incident types). Four tabs:
- **Overview:** totals, share by type, and the key findings in plain language.
- **Time patterns:** incidents by hour of day (00 to 23), by weekday and hour, and weekly trend. Use it to schedule supervision at the busiest hours.
- **Vehicles:** compare vehicles per 1,000 km (fair across vehicles that drive different distances) or by risk points per 1,000 km. Shows each vehicle's dominant incident type.
- **Detection and review:** the rule that triggers each type, and how many reviewed incidents were confirmed versus false alarms. Types with many false alarms are candidates for threshold tuning.

**Download analysed incidents (CSV)** exports the filtered data.

### 4.4 Driver Performance
**Purpose:** compare drivers over time and find who needs coaching.

1. The table ranks drivers by DSS with their tier, distance and incidents.
2. **Drivers shown on chart** selects whose score trend is plotted.
3. **Driver (vehicle)** opens one driver's detail: risk contribution by behaviour (what is costing this driver points) and recent incidents.
4. **Download driver report (CSV)** exports the table.

### 4.5 Scoring Policy
**Purpose:** see the effect of a policy change before adopting it.

1. Change weights or strictness in the sidebar.
2. Pick a **Comparison day**. The page shows *Current weights* against the standard policy, with *Fleet average DSS* and each driver's change in score and rank.
3. The *Policy preview* shows how many drivers move tier.

Use it to answer "if we penalise idling less, who is affected?" without changing any records.

### 4.6 History & Reports
**Purpose:** look back and produce evidence.

1. Filter by **Vehicles**, **Date range** and **Risk tier**. One row is one vehicle-day. Short sessions (fewer than the minimum pings) are excluded by default; untick to include them.
2. **Download sessions (CSV)** and **Download incidents (CSV)** export the filtered data.
3. Select a **Session** for detail:
   - *Route and telemetry*: the route on the map with incidents marked, and speed and fuel charts.
   - *Score audit and report*: every incident with its points and how the score was calculated. **Download report (HTML)** or **(PDF)** gives a per-trip report for the driver or file.
4. Review decisions can also be recorded here, per session.

### 4.7 System & Models
**Purpose:** confirm the system is healthy and manage the risk model.

- **Data:** coverage (vehicles, dates, readings) and data quality.
- **Risk model:** the active model version, its validation, and a list of saved versions. **Switch active version** then **Use this version** to roll back or forward. Train a new version here when new data has been added.
- **Alert gateway:** shows whether Telegram is configured. **Send test message** sends a test to the configured chat.

---

## 5. Everyday workflows

### 5.1 Daily review (about 10 minutes)
1. **Live Monitor:** replay the day and note drivers in High or Critical tier.
2. **Incidents & Review > Incident queue:** filter Status = Open and Priority = High. Review each, record a decision and notes. Send a Telegram alert for confirmed serious cases.
3. Work through Medium priority as time allows.
4. **Unusual trips:** look at newly flagged trips and note any that need a conversation with the driver.

### 5.2 Investigating one driver
1. **Driver Performance:** open the driver; see which behaviour contributes most.
2. **History & Reports:** filter to that vehicle and period; open the worst sessions.
3. Check the route and telemetry; download the PDF report for the file.

### 5.3 Reviewing whether the thresholds suit the fleet
1. **Incident Analysis > Detection and review:** find types with a high false-alarm share.
2. Adjust the relevant threshold in the sidebar (for example a longer idling time).
3. **Scoring Policy:** check how scores shift before adopting the change.

### 5.4 Weekly coaching and supervision planning
1. **Incident Analysis > Vehicles:** note vehicles whose incidents are dominated by one type.
2. **Time patterns:** note the busiest hours and plan supervision around them.
3. Export the CSV for the weekly meeting.

### 5.5 Producing a report for a driver or an investigation
History & Reports > select Session > Score audit and report > Download report (PDF).

### 5.6 Retraining the risk model
After new data has been ingested: System & Models > Risk model > train a new version. Compare its validation with the active version, then **Use this version** if it is better.

---

## 6. How the score works (for explaining a result)

1. Each incident has **points** = (type weight / 3) x base penalty x severity.
2. A trip's **risk (R)** = total points per 100 km driven (minimum distance counted is 50 km, so a very short trip cannot look extreme).
3. **DSS = 100 x exp(-R / R0)**, where R0 is the *Scoring strictness* setting (default 10). No incidents gives 100.
4. In the Live Monitor, older incidents count less: their effect halves every *half-life* hours (default 6).

| DSS | Tier |
|---|---|
| 90 and above | Excellent |
| 75 to 89 | Low Risk |
| 50 to 74 | Moderate Risk |
| 25 to 49 | High Risk |
| below 25 | Critical |

**Incident types**

| Type | Meaning |
|---|---|
| Speeding | Speed above the limit for a sustained period. |
| Harsh driving | Sharp acceleration or braking. |
| Idling | Engine on, vehicle stationary beyond the threshold. |
| Route deviation | Vehicle leaves its normal route. |
| Unauthorized stop | Stopped away from known locations beyond the threshold. |
| Fuel loss | Fuel level drops by more than the threshold without driving that explains it. |
| Fuel loss off route | A fuel loss within 15 minutes of a route deviation or unauthorized stop on the same trip. Scored on top of the fuel loss itself. |

---

## 7. Unusual-trip model: what it can and cannot tell you

- It learns what normal trips look like across the fleet and flags the top 10% most different, with a reason.
- A second model estimates a trip's risk tier from driving patterns. It is checked by leaving out one vehicle at a time and predicting that vehicle's trips: about 91% of trips are placed within one tier of the rule-based result.
- Its reference answers come from the rule-based scores, so this shows **agreement with the rules**, not accuracy against crashes or confirmed theft.
- A flag means "worth a look". A person decides.

---

## 8. Data notes

- Source: Innonics GPS and fuel telematics, 9 August to 31 August 2024.
- 900 files were supplied, 692 contained data. 299,278 readings were read; 3,088 exact duplicate timestamps were removed, leaving 296,190 readings.
- Fuel data is event-based and noisy; fuel-loss incidents are indicators for review, not proof of theft.
- Review decisions are stored in `database/reviews.db`. To clear all reviews, close the dashboard and delete that file; it is recreated empty on next start.

---

## 9. Telegram alerts

### 9.1 Set up
1. In Telegram, message **@BotFather**, send `/newbot` and follow the prompts. Copy the **bot token**.
2. Add the bot to the chat or group that should receive alerts (or start a private chat with it and send any message).
3. Find the **chat ID** (for example with the bot's `getUpdates` page, or a chat-ID bot). Group IDs are negative numbers; a private chat ID is positive.
4. In the project folder, copy `.env.example` to `.env` and fill in:

```
TELEGRAM_BOT_TOKEN=<your token>
TELEGRAM_CHAT_ID=<chat id>
TELEGRAM_AUTO_NOTIFY=false
```
5. Restart the dashboard. In **System & Models > Alert gateway**, press **Send test message**. The message should arrive within seconds.

Keep `.env` private. It is excluded from version control; never share the token.

### 9.2 Sending an alert
Incidents & Review > select an incident > **Supervisor alert > Send alert (Telegram)**. The alert carries vehicle, incident type, time, location and priority. Delivery is recorded in the alert log shown under the button.

An alert reaches **everyone in the configured chat**. To alert several supervisors, add them to a Telegram group and use that group's ID.

---

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "The telematics_pings table is empty" or database missing | Ingestion has not been run. See 2.3. |
| `InconsistentVersionWarning` from scikit-learn in the terminal | The model was saved with a newer scikit-learn. It is harmless. To remove it: `pip install --upgrade scikit-learn`. |
| First visit to Incidents & Review is slow | The risk model is training once. It is saved afterwards. |
| Page is slow after changing a threshold | Detectors are re-running on the whole dataset. Wait a few seconds. |
| Telegram test fails | Check `.env` (token, chat ID), that the bot is a member of the chat, and the internet connection. Restart the dashboard after editing `.env`. |
| Numbers differ slightly from an older export | The scoring policy or thresholds in the sidebar differ from when it was made. Use Reset to defaults. |
| Reviews disappeared | `database/reviews.db` was deleted or moved. |
| Port already in use | Run `streamlit run src/dashboard.py --server.port 8502`. |

---

## 11. Glossary

- **DSS:** Driver Safety Score, 0 to 100, higher is safer.
- **Session / trip:** one vehicle's continuous activity on one day.
- **Incident:** one detected event (for example one speeding episode), with a start time, duration and severity.
- **Priority:** how urgently an incident needs review (High, Medium, Low).
- **Tier:** the band a DSS falls in (Excellent to Critical).
- **Half-life:** time for an incident's effect on the live score to halve.
- **Anomaly score:** how unusual a trip is relative to the fleet, as a percentile; 90 and above is flagged.
- **False alarm:** a reviewed incident judged not to be a real problem. A high share for one type suggests the threshold needs adjusting.
