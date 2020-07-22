import time
import multiprocessing

start_time = time.time()

def count(name):
    for i in range(1, 100):
        print(name, ":", i)

k = 4
num_list = [f"p{i}" for i in range(k)]

pool = multiprocessing.Pool(processes=int(k//2))
pool.map(count, num_list)
pool.close()
pool.join()

print(f"------ End {time.time() - start_time} ------")
