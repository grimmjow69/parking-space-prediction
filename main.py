from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import random
from ultralytics import YOLO

from parking_spot import ParkingSpot
from parking_spot_result import ParkingSpotResult
import base64

app = Flask(__name__)

parking_areas = {
    'zaliv_a': {
        'url': 'http://user:eltodo@158.193.166.34/snap.jpg',
        'spots': [
            ParkingSpot("FRONT_D_1", [[673, 297], [722, 297], [722, 341], [673, 341]]),
            ParkingSpot("FRONT_D_2", [[717, 299], [769, 299], [769, 338], [717, 338]]),
            ParkingSpot("FRONT_D_3", [[767, 294], [822, 294], [822, 339], [767, 339]]),
            ParkingSpot("FRONT_D_4", [[818, 298], [867, 298], [867, 339], [818, 339]]),
            ParkingSpot("FRONT_D_5", [[864, 292], [927, 292], [927, 337], [864, 337]]),
            ParkingSpot("FRONT_D_6", [[919, 289], [973, 289], [973, 340], [919, 340]]),
            ParkingSpot("FRONT_D_7", [[969, 285], [1029, 285], [1029, 336], [969, 336]]),
            ParkingSpot("FRONT_D_8", [[1032, 282], [1081, 282], [1081, 334], [1032, 334]]),
            ParkingSpot("FRONT_D_9", [[1083, 285], [1130, 285], [1130, 338], [1083, 338]]),
            ParkingSpot("FRONT_D_10", [[1131, 282], [1187, 282], [1187, 336], [1131, 336]]),
            ParkingSpot("FRONT_D_11", [[1190, 282], [1247, 282], [1247, 343], [1190, 343]]),
            # ParkingSpot("RAD6_1", [[1843, 741], [1911, 741], [1911, 816], [1843, 816]]),
            # ParkingSpot("RAD6_2", [[1803, 654], [1911, 654], [1911, 741], [1803, 741]]),
            # ParkingSpot("RAD6_3", [[1768, 593], [1911, 593], [1911, 683], [1768, 683]]),
            # ParkingSpot("RAD6_4", [[1721, 551], [1911, 551], [1911, 646], [1721, 646]]),
            # ParkingSpot("RAD6_5", [[1710, 519], [1911, 519], [1911, 598], [1710, 598]]),
            # ParkingSpot("RAD6_6", [[1674, 505], [1910, 505], [1910, 578], [1674, 578]]),
            # ParkingSpot("RAD6_7", [[1664, 479], [1901, 479], [1901, 553], [1664, 553]]),
            # ParkingSpot("RAD6_8", [[1633, 455], [1862, 455], [1862, 526], [1633, 526]]),
            # ParkingSpot("RAD6_9", [[1606, 434], [1811, 434], [1811, 510], [1606, 510]]),
            # ParkingSpot("RAD6_10", [[1596, 431], [1792, 431], [1792, 488], [1596, 488]]),
            # ParkingSpot("RAD6_11", [[1583, 421], [1796, 421], [1796, 473], [1583, 473]]),
            # ParkingSpot("RAD6_12", [[1569, 395], [1771, 395], [1771, 450], [1569, 450]]),
            # ParkingSpot("RAD6_13", [[1564, 384], [1788, 384], [1788, 435], [1564, 435]]),
            # ParkingSpot("RAD6_14", [[1545, 371], [1744, 371], [1744, 417], [1545, 417]]),
            # ParkingSpot("RAD6_15", [[1542, 356], [1710, 356], [1710, 408], [1542, 408]]),
            # ParkingSpot("RAD6_16", [[1519, 335], [1678, 335], [1678, 385], [1519, 385]]),
            # ParkingSpot("RAD6_17", [[1509, 321], [1639, 321], [1639, 364], [1509, 364]]),
            # ParkingSpot("RAD6_18", [[1488, 311], [1625, 311], [1625, 357], [1488, 357]]),
            # ParkingSpot("RAD6_19", [[1478, 303], [1614, 303], [1614, 349], [1478, 349]])
        ]
    }
}


def evaluate_occupied_spots_in_area(parking_spots, predictions):
    occupied_spots = []
    for spot in parking_spots:
        spot_polygon = spot.get_polygon()

        occupied = False

        for pred in predictions:
            point = (int(pred['x']), int(pred['y']))

            if cv2.pointPolygonTest(spot_polygon, point, False) >= 0:
                occupied = True
                break

        occupied_spots.append(ParkingSpotResult(spot.name, occupied))

    return occupied_spots


def get_model_predications_result(image):
    model = YOLO('best.pt')
    results = model(image, conf=0.3, iou=0.7)

    predictions = []

    for result in results:
        boxes = result.boxes
        for box in boxes.xyxy:
            x_center = int((box[0] + box[2]) / 2)
            y_center = int((box[1] + box[3]) / 2)
            prediction = {'x': x_center, 'y': y_center}
            predictions.append(prediction)

    return predictions


@app.route('/get_parking_occupancy', methods=['GET'])
def get_parking_occupancy():
    all_occupancy_results = {}

    for area, details in parking_areas.items():
        url = details['url']
        spots = details['spots']

        response = requests.get(url)

        if response.status_code != 200:
            continue

        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is not None:
            roboflow_predictions = get_model_predications_result(image)
            spot_occupancy_result = evaluate_occupied_spots_in_area(spots, roboflow_predictions)
            result_list = [obj.__dict__ for obj in spot_occupancy_result]
            all_occupancy_results[area] = result_list

    return all_occupancy_results


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
