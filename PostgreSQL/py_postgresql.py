import psycopg2
import hidden

# Load the secrets
secrets = hidden.secrets()

conn = psycopg2.connect(host=secrets['host'],
        port=secrets['port'],
        database=secrets['database'],
        user=secrets['user'],
        password=secrets['pass'],
        connect_timeout=3)

cur = conn.cursor()

sql = 'DROP TABLE IF EXISTS pythonfun CASCADE;'
print(sql)
cur.execute(sql)
sql = 'CREATE TABLE pythonseq (iter INTEGER, val INTEGER);'
cur.execute(sql)
number = 989764
value
for i in range(300) :
    print(i+1, number)
    value = int((value * 22) / 7) % 1000000
    sql = 'INSERT INTO pythonseq (iter, val) VALUES (%s);'
    print(sql)
    cur.execute(sql, (i+1, value, ))
conn.commit()
cur.close()
