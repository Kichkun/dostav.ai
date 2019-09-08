import json
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm_notebook

def load_data(file):
    """Загрузка входных данных из файла"""
    with open(file, 'r') as f:
        input_data = json.load(f)
    couriers = {}
    orders = {}
    points = {}
    for depotData in input_data['depots']:
        points[depotData['point_id']] = {
            'location': [depotData['location_x'], depotData['location_y']],
            'timewindow': [0, 1439],
        }
    for courierData in input_data['couriers']:
        couriers[courierData['courier_id']] = {
            'location': [courierData['location_x'], courierData['location_y']],
            'time': 360,
        }
    for orderData in input_data['orders']:
        points[orderData['pickup_point_id']] = {
            'location': [orderData['pickup_location_x'], orderData['pickup_location_y']],
            'timewindow': [orderData['pickup_from'], orderData['pickup_to']],
            'order_time': {orderData['order_id']: orderData['pickup_from']}
        }
        points[orderData['dropoff_point_id']] = {
            'location': [orderData['dropoff_location_x'], orderData['dropoff_location_y']],
            'timewindow': [orderData['dropoff_from'], orderData['dropoff_to']],
        }
        orders[orderData['order_id']] = orderData
    return couriers, orders, points

def get_travel_duration_minutes(location1, location2):
    """Время перемещения курьера от точки location1 до точки location2 вминутах"""
    distance = abs(location1[0] - location2[0]) + abs(location1[1] - location2[1])
    return 10 + distance

def get_current_courier_reward(time, payment):
    work_payment = time * 2
    profit = orders_payment - work_payment
    return profit

def delete_element(array, idx):
    result = []
    for i in range(len(array)):
        if not(i == idx):
            result.append(array[i])
    return result

def main():
    payment = 0
    data = load_data("contest_input.json")
    reward = 0
    stop_flag = 0
    couriers_start_position = [data[0][i+1]['location'] for i in range(len(data[0]))]
    points_pickup = np.array([[data[1][i+10001]['pickup_location_x'],
                               data[1][i+10001]['pickup_location_y']] for i in range(len(data[1]))])
    time_pickup = np.array([[data[1][10001+i]['pickup_from'],
                             data[1][10001+i]['pickup_to']] for i in range(len(data[1]))])
    time_dropoff = np.array([[data[1][10001+i]['dropoff_from'],
                              data[1][10001+i]['dropoff_to']] for i in range(len(data[1]))])    
    points_dropoff = np.array([[data[1][i+10001]['dropoff_location_x'],
                                data[1][i+10001]['dropoff_location_y']] for i in range(len(data[1]))])
    payments = np.array([data[1][10001+i]['payment'] for i in range(len(data[1]))])
    pickup_id = np.array([data[1][10001+i]['pickup_point_id'] for i in range(len(data[1]))])
    dropoff_id = np.array([data[1][10001+i]['dropoff_point_id'] for i in range(len(data[1]))])
    cnt = 0
    cnt_courier = 0
    routes = []
    for courier in tqdm_notebook(couriers_start_position):
        if stop_flag == 1:
            break 
        print('courier {} begins route at {}'.format(cnt_courier, courier))
        cnt_courier += 1
        tmp_x = [courier[0]]
        tmp_y = [courier[1]]
        print('processing courier {}..\n'.format(cnt))
        cnt += 1
        courier_position = courier
        current_time = 360
        distances = []
        routes = []
        # select avaliable pickup points
        while current_time < 1400:
            distances = []
            avaliable_pickups = []
            print("current time {}. Passing order {}\n".format(current_time, cnt))
            cnt += 1
            tmp_route = []
            for j in range(len(points_pickup)):
                # calculate route_time for each point from current courier position to dropoff
                route_time = get_travel_duration_minutes(points_pickup[j],points_dropoff[j]) + get_travel_duration_minutes(points_pickup[j],
                                                                                                 courier_position)
                time_to_pickup = get_travel_duration_minutes(courier_position, points_pickup[j])
                if current_time + time_to_pickup  in time_pickup[j] and current_time + route_time in time_dropoff[j]:
                    avaliable_pickups.append(points_pickup[j])       

            # calculate Manhattan distance to closest avaliable points
            min_dist = np.inf
            if len(avaliable_pickups) == 0:
                break
                print('сука')
            for i in range(len(avaliable_pickups)):
                dist = get_travel_duration_minutes(courier, avaliable_pickups[i])
                if not(dist == 0):
                    tmp = get_travel_duration_minutes(courier, avaliable_pickups[i])
                    if tmp < min_dist:
                        min_dist = tmp
                        current_order = i

                tmp_route.append( {"courier_id": str(cnt_courier),
                      "action":"pickup", "order_id":str(10001+j),
                      "point_id":str(pickup_id[current_order])})
                tmp_route.append( {"courier_id": str(cnt_courier),
                                  "action":"dropoff", "order_id":str(10001+j),
                                  "point_id":str(dropoff_id[current_order])})
            
            # delete visited location
            print('the most far point: {}'.format(min_dist))
            courier_position = points_dropoff[j]
            print('send courier on {}'.format(courier_position))
            tmp_x.append(courier_position[0])
            tmp_y.append(courier_position[1])

            current_time += route_time 
            payment = payments[current_order] 
            print('Payment: {}, salary: {}'.format(payments[current_order], route_time))

            #deleting visited elements
            points_dropoff = delete_element(points_dropoff,current_order)
            time_pickup = delete_element(time_pickup,current_order)
            points_pickup = delete_element(points_pickup, current_order)
            time_dropoff = delete_element(time_dropoff, current_order)
            payments = delete_element(payments, current_order)

            print("{} points left".format(len(points_pickup)))
        reward += payment   
        print(reward)
        print("-"*80)
        routes.append(tmp_route)
    print('total reward: {}'.format(reward - (current_time-360)*2))
    routes = routes[0]
    routes_dict = dict({})
    for i in range(len(routes)):
        routes_dict.update({str(i):routes[i]})
    data=json.dumps(routes_dict)
    with open('output.txt', 'w') as outfile:
        json.dump(data, outfile)
if __name__ == '__main__':
    main()
