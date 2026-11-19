# Dataset description

**Source and ownership.** Private telematics data supplied by Innonics Sdn. Bhd. (the project stakeholder) from
commercial vehicles in Malaysia. It is not public and is not redistributed in this repository.

**Content.** One JSON file per vehicle per day, named `month_day_vehicle` (for example `8_7_9` = vehicle 9 on 7 August).
Each record holds: datetime, mileage, heading, speed, longitude, latitude, acc, event_message, fuel1_volume,
fuel2_volume, temperature.

| Item | Value |
|---|---|
| Files | 900 JSON files, 692 containing data |
| Raw pings | 299,278 |
| Stored after ingestion cleaning | 296,190 (confirm the exact reason from the ingestion report before presenting) |
| Vehicles | 10 trackers (IDs 0 to 9) |
| Trips (vehicle-days) | 692, of which 689 have at least 20 pings |
| Period | January to August 2024 (data arrives in monthly bursts, not every day) |
| Timestamps | Malaysia time (MYT, UTC+8), confirmed by the project owner |
| Typical reporting interval | about 38 s while moving; parked, ignition-off vehicles report about once an hour |

**Coverage is uneven.** Vehicles 8 and 9 supply most of the driving days (72 and 64 days of 50 km or more); vehicles 2, 3
and 6 have one or two. The dashboard marks drivers with fewer than five driving days as "Limited data".

**No labels.** The data holds no crash, claim or theft labels. Rule outputs are therefore validated with synthetic tests
and fuel-drop injection (`docs/results/fuel_injection_summary.md`), and the machine-learning layer is trained on the
rule-based score and evaluated on vehicles it has not seen.

**Known data issues handled.** Fuel readings jump by tens of litres while moving (treated as sensor noise, not theft);
hourly heartbeat pings while parked make fuel changes unreliable over those gaps; the old "vehicle" count of 692 in the
FYP1 report was really vehicle-days.
