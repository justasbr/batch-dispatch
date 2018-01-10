import numpy as np


def batch_time(batch_size):
    return round(batch_size ** 0.91, 2)


def calculate_results(items_per_sec, batch_size, batch_time_mean, time_std=0):
    NUM_OF_ITEMS = 10000

    thruput = 1000.0 / ((batch_time_mean * 1.0) / batch_size)  # ms for one img

    sec_between_items = 1.0 / items_per_sec
    # print(sec_between_items)
    inputs = np.arange(0, sec_between_items * NUM_OF_ITEMS, sec_between_items)  # include 1
    # print(len(inputs), inputs)
    outputs = np.arange(batch_time_mean, batch_time_mean * (NUM_OF_ITEMS * 1.2 / batch_size), batch_time_mean)
    outputs = np.repeat(outputs, batch_size)
    outputs = outputs / 1000
    outputs = outputs[:NUM_OF_ITEMS]
    lat = 1000 * (outputs - inputs)

    lat[lat < batch_time_mean] = batch_time_mean

    assert len(lat[lat < 0]) == 0
    assert len(lat[lat < batch_time_mean]) == 0

    # latencies[latencies < 0] = 0
    # print(latencies)
    print("items=" + str(len(lat)), end="\t")
    print("b_size=" + str(batch_size), end="\t")
    print("b_time=" + str(batch_time_mean), end="\t")
    print("thru_sec=\t" + str(int(thruput)), end="\t")
    print("latMS=\t" + str(np.mean(lat, dtype=np.int)) + " \t+/-\t " + str(round(np.std(lat, dtype=np.int))))


# items_per_sec = 2000
# batch_size = 4
# batch_time_mean = 7  # ms
# time_std = 2  # ms (later)

for b in range(1, 100, 1):
    calculate_results(items_per_sec=1200, batch_size=b, batch_time_mean=batch_time(b))
# calculate_results(items_per_sec=1000, batch_size=4, batch_time_mean=7, time_std=2)
# calculate_results(items_per_sec=1000, batch_size=16, batch_time_mean=27, time_std=2)
# calculate_results(items_per_sec=1000, batch_size=64, batch_time_mean=107, time_std=2)
# calculate_results(items_per_sec=1000, batch_size=256, batch_time_mean=427, time_std=2)
