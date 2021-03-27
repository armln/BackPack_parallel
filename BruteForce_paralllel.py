import pandas as pd
from itertools import combinations
import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np

tests = ["5", "8", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26"]
# tests = ["5", "8"]


def backpack_brute_force(n, capacity, weight_cost, combinations):
    best_cost = None
    best_combination = []
    # generating combinations : C by 1 from n, C by 2 from n, ...
    # combinations(range(4), 3) --> 012 013 023 123
    for combination in combinations:
        weights = sum([weight[0] for weight in combination])
        costs = sum([cost[1] for cost in combination])
        if (best_cost is None or best_cost < costs) and weights <= capacity:
            best_cost = costs
            best_combination = [0] * n
            for c in combination:
                best_combination[weight_cost.index(c)] = 1

    return best_cost, best_combination


def task(args):
    pid, num_threads, tuple_array, capacity, n, combs = args
    t = len(combs)
    task_size = t // num_threads
    start = task_size * pid
    count = task_size if (pid != (num_threads - 1)) else task_size + t % num_threads
    cost, comb = backpack_brute_force(n, capacity, tuple_array, combs[start:start+count])

    return cost, comb



def main(num_threads):
    for file in tests[:10]:
        test = pd.read_csv("./tests/test_" + file + ".csv", header=None, delimiter=";").values
        result = pd.read_csv("./tests/bpresult_" + file + ".csv", header=None, delimiter=";").values.ravel()
        n = test.shape[0]
        capacity = sum(test[:, 0]) / 2
        tuple_array = [tuple(x for x in row) for row in test]
        combs = []
        for i in range(n):
            combs.extend(list(combinations(tuple_array, i)))

        with mp.Pool(num_threads) as pool:
            answer = pool.map(task, [(i, num_threads, tuple_array, capacity, n, combs) for i in range(num_threads)])

            answer = sorted(answer, key=lambda x: x[0], reverse=True)
            print(f'File: {file}, result: {all(result == answer[0][1])}')
        # print(tuple_array, capacity)
        # best_cost, best_combination = backpack_brute_force(n, capacity, tuple_array)
        # print(assert_allclose(best_combination, result))
        # print(best_combination, file)
        # backpack_brute_force(n, capacity, tuple_array)



if __name__ == '__main__':
    threads = [t for t in range(1, 5)]
    times = []

    for t in threads:
        print('processes cnt:', t)
        start = time.time()
        main(t)
        end = time.time()
        delta = end - start
        print("Time: ", delta)
        times.append(delta)
        print(times)

    times = np.array(times) / times[0]
    # plt.figure(figsize=(16,9))
    plt.plot(threads, times, 'o-', label='n-th thread time vs 1-thread time')
    plt.legend()
    plt.show()
