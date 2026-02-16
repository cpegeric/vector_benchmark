We need a benchmark test for vector index.  The need to know the timing of create index, search index in parallel (QPS), bulk delete, insert and update.
The exising tool is @indextest.py which will generate the data on the fly in multiple block with each block 1000 rows.  The generated dataset is 
always reproducible because of using same rand seed.  All command to create index, search, insert, delete are in SQL.

We want to extend the framework to do the followings:

1. generate the dataset in csv format for offline loading
2. to support for async ans sync index creation (looking at the SQL)
3. run recall
4. bulk delete, update, insert
5. for search, we need to support normal index search and pre-filtering index search

The Table consists of columns 
1. id - in64
2. vector - []float32/[]float64
3. i32v - int32 for pre-filtering
4. f32v - float32 for pre-filtering
5. str - string for pre-filtering

all columns have its own rand but share the same rand seed, i.e. []rand

python scripts or library need to implement:

1. gen.py - to support generate both offline csv file and data in block as library to other components to call when so operations in stream
2. create.py - to create index for both async ans sync index with offline csv and stream data (in blocks)
3. recall.py - run recall to get the timing in QPS and stats in parallel search
4. insert.py - to simulate inserts after index created
5. delete.py - to simluate delete after index created
5. update.py - to simluale update after index created

configuration in json:

{
  "host: "localhost",
  "database" :"mydb",
  "table": "mytbl",
  "index"  : "myidx",
  "index_type" : "hnsw/ivfflat",
  "distance": "vector_l2_ops",
  "dimension": : 1024,
  "dataset": 10000
}


commands:

# offline csv generation
e.g. gen.py -f cfg.json -o output.csv

# to create index
e.g. create.py -f cfg.json [-s] [-a]
-s is synchronous mode
-a is asynchonous mode

for synchonous mode,  do it in sequence
1. create table
2. insert data
3. create index

for asynchronous mode,
1. create table
2. create index
3. insert data

create gen.py and create.py and use @indextest.py as reference
