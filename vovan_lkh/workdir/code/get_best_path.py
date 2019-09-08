import json
import numpy as np

DATA_PATH = '../data/0.json'

json_file = json.load(open(DATA_PATH, 'r'))


json_file['orders'] = json_file['orders'][:50]
dim = 2 * len(json_file['orders'])

cnt = 1
inverse_id_map = {}
inverse_order_map = {}
for order in json_file['orders']:
    cnt += 1
    inverse_id_map[order['pickup_point_id']] = cnt
    inverse_order_map[cnt] = order
    cnt += 1
    inverse_id_map[order['dropoff_point_id']] = cnt
    inverse_order_map[cnt] = order

sol_file = open('../tmp.txt', 'r').readlines()
aa = [int(a[:-1]) for a in sol_file[:-2]]

solutions = []

cnt = 0
for a in aa:
    if a > 101 or a == 1:
        solutions.append([])
        cnt += 1
        continue
    solutions[cnt - 1].append(a - 2)

sol_revards = np.zeros(len(solutions))
for i, sol in enumerate(solutions):
    if len(sol) == 0:
        continue

    for order_id in sol:
        sol_revards[i] += json_file['orders'][order_id // 2]['payment']

sol_revards = sol_revards / 2

best_sol_id = np.argsort(sol_revards)[-2]
best_sol = solutions[best_sol_id]

print(best_sol)

pm = 0

result_json = []
for point_id in best_sol:
    order = point_id // 2
    if point_id % 2 == 0:
        action = 'pickup'
        point_id = json_file['orders'][order]['pickup_point_id']
        order_id = json_file['orders'][order]['order_id']
    else:
        action = 'dropoff'
        point_id = json_file['orders'][order]['dropoff_point_id']
        order_id = json_file['orders'][order]['order_id']

    result_json.append({"courier_id": 1,
    	                "action": action,
    	                "order_id": order_id,
    	                "point_id": point_id})

    pm += (json_file['orders'][order]['payment'])

    print(action, order_id, point_id)

json.dump(result_json, open('simple_sol.json', 'w'))

print(json_file['orders'][45])

print(result_json)
print(pm / 2)
