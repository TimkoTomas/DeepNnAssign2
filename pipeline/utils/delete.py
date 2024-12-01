import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('vectors.db')
c = conn.cursor()

# Drop the reviews table if it exists
c.execute('''DROP TABLE IF EXISTS reviews''')

# Drop the vectors table if it exists
c.execute('''DROP TABLE IF EXISTS vectors''')

# Commit the changes and close the connection
conn.commit()
conn.close()

print("Tables 'reviews' and 'vectors' have been deleted.")