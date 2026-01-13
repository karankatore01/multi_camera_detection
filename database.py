import sqlite3
import pickle

conn = sqlite3.connect("persons.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS persons (
    person_id INTEGER PRIMARY KEY AUTOINCREMENT,
    embeddings BLOB
)
""")
conn.commit()

def get_all_persons():
    cursor.execute("SELECT person_id, embeddings FROM persons")
    rows = cursor.fetchall()
    return [(pid, pickle.loads(emb)) for pid, emb in rows]

def add_or_update_person(pid, embedding, max_embeddings=10):
    if pid is None:
        cursor.execute(
            "INSERT INTO persons (embeddings) VALUES (?)",
            (pickle.dumps([embedding]),)
        )
        conn.commit()
        return cursor.lastrowid
    else:
        cursor.execute(
            "SELECT embeddings FROM persons WHERE person_id=?",
            (pid,)
        )
        embeddings = pickle.loads(cursor.fetchone()[0])
        embeddings.append(embedding)
        embeddings = embeddings[-max_embeddings:]

        cursor.execute(
            "UPDATE persons SET embeddings=? WHERE person_id=?",
            (pickle.dumps(embeddings), pid)
        )
        conn.commit()
        return pid
