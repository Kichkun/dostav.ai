import json
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = '../data/contest_input_filtered.json'

json_file = json.load(open(DATA_PATH, 'r'))

couriers = json_file['couriers']

orders = json_file['orders']
orders_pickup_times_0 = np.array([(order['pickup_from'], order['pickup_to']) for order in orders.values()])
orders_pickup_times = np.array([((10 * (order['pickup_from'] // 10)), (10 * (order['pickup_to'] // 10))) for order in orders.values()])
orders_drop_times = np.array([((10 * (order['dropoff_from'] // 10)), (10 * (order['dropoff_to'] // 10))) for order in orders.values()])
orders_from = np.array([(order['pickup_location_x'], order['pickup_location_y']) for order in orders.values()])
orders_to = np.array([(order['dropoff_location_x'], order['dropoff_location_y']) for order in orders.values()])
payments = np.array([(order['payment']) for order in orders.values()])

delivery_dists = np.abs(orders_to - orders_from).sum(axis=1)

def manhattan(vector, point):
    return np.abs((vector - point)).sum(axis=1)

def courier_moves(x, y, R, max_orders, used_ids, cur_step=0, max_step=40):
    if cur_step > max_step:
        return used_ids
    
    courier_pr = (x, y)
    cur_time = 360
    
    dists = manhattan(orders_from, courier_pr)
    print(dists)

    to_time_max = np.minimum(cur_time + dists + 10, orders_pickup_times[:, 1])
    available_in_time_from = orders_pickup_times[:, 0] < to_time_max
    available_time_to = orders_pickup_times[:, 1] < (cur_time + dists - 3)

    available = (available_in_time_from * available_time_to)

    in_radius = dists < R

    near_and_available = available * in_radius

    price_per_meter = payments / (dists + delivery_dists + 1)
    sorted_ids = np.argsort(price_per_meter)


    orders_to_use = np.where(near_and_available)[0]
    price_selected = price_per_meter[orders_to_use]
    sorted_prices = np.argsort(price_selected)
    
    sorted_prices = np.argsort(price_per_meter[orders_to_use])
    orders_to_use = orders_to_use[sorted_prices[::-1]][:max_orders]
    
    additional_ids = []
    for cur_order in orders_to_use:
        if cur_order in used_ids:
            continue
        else:
            additional_ids.append(cur_order)

            cur_x = orders_to[cur_order, 0]
            cur_y = orders_to[cur_order, 1]
            
            used_ids = courier_moves(cur_x, cur_y, R, max_orders, additional_ids + used_ids, cur_step + 1)

    return used_ids

for courier in couriers:
	print(courier)
	cor_x = courier['location_x']
	cor_y = courier['location_y']

	print(cor_x, cor_y)

	candidates = []
	courier_moves(cor_x, cor_y, 180, 10, candidates)
	candidates = list(set(candidates))
	print(candidates)

#	print(np.max(candidates))

#	orders_used = np.array(orders)[candidates]
#	break
