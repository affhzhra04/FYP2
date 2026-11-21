"""read_pings must return one row per (vehicle_id, timestamp) even if ingestion ran twice."""
import sqlite3

from src import dash_core as C


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE telematics_pings (id INTEGER PRIMARY KEY AUTOINCREMENT, vehicle_id TEXT, timestamp TEXT, "
                "latitude REAL, longitude REAL, speed REAL, fuel_level REAL, heading REAL, ignition INTEGER)")
    con.executemany("INSERT INTO telematics_pings(vehicle_id,timestamp,latitude,longitude,speed,fuel_level,heading,ignition) "
                    "VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit(); con.close()


def test_duplicate_rows_are_collapsed(tmp_path):
    a = ("8_9_1", "2024-08-09 09:00:00", 3.0, 101.0, 50.0, 200.0, 90.0, 1)
    b = ("8_9_1", "2024-08-09 09:00:30", 3.001, 101.001, 52.0, 199.9, 90.0, 1)
    c = ("8_9_2", "2024-08-09 09:00:00", 3.1, 101.1, 0.0, 150.0, 0.0, 0)      # same time, different session: kept
    db = tmp_path / "t.db"
    make_db(db, [a, b, a, c, b])                                              # a and b appear twice
    df = C.read_pings(db)
    assert len(df) == 3
    assert sorted(df["vehicle_id"]) == ["8_9_1", "8_9_1", "8_9_2"]
    assert list(df.index) == [0, 1, 2]
