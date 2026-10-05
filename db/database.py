# SQLite setup with corruption-safe handling

import os
import time
import shutil
import sqlite3


def get_db():
    """Open (or create) the SQLite database at `data/memory.db`.

    If the file exists but is not a valid SQLite database, move it to a
    backup (`memory.db.corrupt.<ts>`) and create a fresh database so the
    application can continue running.
    """
    data_dir = "data"
    db_path = os.path.join(data_dir, "memory.db")
    os.makedirs(data_dir, exist_ok=True)

    # Try opening and doing a quick integrity check. If it fails, back up
    # the existing file and create a fresh DB.
    try:
        conn = sqlite3.connect(db_path)
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA integrity_check;")
            row = cur.fetchone()
            if row is None or row[0] != "ok":
                raise sqlite3.DatabaseError("integrity_check failed")
        except sqlite3.DatabaseError:
            try:
                conn.close()
            except Exception:
                pass
            if os.path.exists(db_path):
                corrupt_name = f"{db_path}.corrupt.{int(time.time())}"
                shutil.move(db_path, corrupt_name)
            conn = sqlite3.connect(db_path)

        return conn
    except sqlite3.DatabaseError:
        # If connect itself failed (corrupt file or filesystem issue),
        # back up and recreate.
        if os.path.exists(db_path):
            corrupt_name = f"{db_path}.corrupt.{int(time.time())}"
            try:
                shutil.move(db_path, corrupt_name)
            except Exception:
                pass
        return sqlite3.connect(db_path)