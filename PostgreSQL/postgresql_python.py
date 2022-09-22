import psycopg2
import pandas as pd
Database=''
Endpoint = ''
port = ''
user=''
password=""
csv_file = ""

engine = psycopg2.connect(
    database=Database,
    user=user,
    password=password,
    host=Endpoint,
    port=port,
    options="-c search_path=schame_n" # schame
)
cursor=engine.cursor()
cursor.execute('SHOW search_path;')
print(cursor.fetchall())


#creating table
sql = 'DROP TABLE IF EXISTS table_name CASCADE;'
print(sql)
cur.execute(sql)
sql = 'CREATE TABLE table_name (col1 INTEGER, col2 INTEGER);'
cur.execute(sql)
conn.commit()


# insert values

with open(csv_file, 'r') as row:
    next(row)# Skip the header row.
    cursor.copy_from(row, 'table1', sep=',')



cursor.commit()
cursor.close()
