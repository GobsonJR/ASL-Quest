import sqlite3

db = sqlite3.connect("data/asl_quest.db")
cur = db.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
tables = [t[0] for t in cur.fetchall()]
print("Tables:", tables)
if "native_signs" in tables:
    cur.execute("SELECT COUNT(*) FROM native_signs")
    print("native_signs count:", cur.fetchone()[0])
    cur.execute("SELECT id, gloss, display_name, active FROM native_signs LIMIT 5")
    for r in cur.fetchall():
        print(" ", r)
else:
    print("native_signs NOT in DB - need to run migration 0002")

if "alembic_version" in tables:
    cur.execute("SELECT version_num FROM alembic_version")
    print("Alembic version:", cur.fetchone())
db.close()
