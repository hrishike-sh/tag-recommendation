import duckdb

con = duckdb.connect()
path = "data/lake/1k/529b8f83aada-3f0122fb/track_metadata.parquet"
desc = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
print("Schema:", desc)
sample = con.execute(f"SELECT * FROM read_parquet('{path}') LIMIT 3").fetchall()
print("Sample:", sample)
