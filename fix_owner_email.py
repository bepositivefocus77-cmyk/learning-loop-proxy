import sqlite3

conn = sqlite3.connect("learning_loop.db")
conn.execute("UPDATE users SET email = ? WHERE email = ?", ("owner@example.com", "owner@local"))
conn.commit()
print("Updated. Rows affected:", conn.total_changes)
conn.close()