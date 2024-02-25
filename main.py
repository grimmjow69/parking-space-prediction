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
        'url': '',
        'images': [
            'images/zaliv_a_snow.png',
            'images/zaliv_a_day.png',
            'images/zaliv_a_light_snow.png',
        ],
        'spots': [
            ParkingSpot("spot_1", [[732, 339], [895, 339], [895, 409], [732, 409]]),
            ParkingSpot("spot_2", [[767, 258], [897, 258], [897, 307], [767, 307]]),
            ParkingSpot("spot_3", [[1029, 251], [1168, 251], [1168, 299], [1029, 299]]),
            ParkingSpot("spot_4", [[1040, 290], [1188, 290], [1188, 337], [1040, 337]]),
            ParkingSpot("spot_5", [[1047, 331], [1205, 331], [1205, 387], [1047, 387]]),
            ParkingSpot("spot_6", [[1071, 373], [1203, 373], [1203, 417], [1071, 417]]),
            ParkingSpot("spot_7", [[1082, 409], [1230, 409], [1230, 461], [1082, 461]]),
            ParkingSpot("spot_8", [[1077, 449], [1246, 449], [1246, 505], [1077, 505]]),
            ParkingSpot("spot_9", [[1084, 488], [1255, 488], [1255, 558], [1084, 558]]),
            ParkingSpot("spot_10", [[1101, 539], [1277, 539], [1277, 634], [1101, 634]]),
            ParkingSpot("spot_11", [[1100, 627], [1296, 627], [1296, 713], [1100, 713]]),
            ParkingSpot("spot_12", [[1113, 702], [1310, 702], [1310, 786], [1113, 786]]),
            ParkingSpot("spot_13", [[1114, 772], [1330, 772], [1330, 877], [1114, 877]]),
            ParkingSpot("spot_14", [[1136, 874], [1368, 874], [1368, 1023], [1136, 1023]]),
            ParkingSpot("spot_15", [[578, 972], [827, 972], [827, 1048], [578, 1048]]),
            ParkingSpot("spot_16", [[582, 859], [825, 859], [825, 976], [582, 976]]),
            ParkingSpot("spot_17", [[608, 765], [822, 765], [822, 868], [608, 868]]),
            ParkingSpot("spot_18", [[628, 686], [834, 686], [834, 775], [628, 775]]),
            ParkingSpot("spot_19", [[650, 620], [847, 620], [847, 696], [650, 696]]),
            ParkingSpot("spot_20", [[651, 544], [859, 544], [859, 627], [651, 627]]),
            ParkingSpot("spot_21", [[670, 494], [860, 494], [860, 581], [670, 581]]),
            ParkingSpot("spot_22", [[682, 448], [861, 448], [861, 526], [682, 526]]),
            ParkingSpot("spot_23", [[698, 405], [856, 405], [856, 464], [698, 464]]),
            ParkingSpot("spot_24", [[720, 375], [872, 375], [872, 428], [720, 428]])
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
    model = YOLO('best_model.pt')
    classes = [0]
    results = model(image, conf=0.1, classes=classes, iou=0.3, imgsz=640)

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
        # url = details['url']
        spots = details['spots']
        images = details['images']
        image = random.choice(images)

        # response = requests.get(url)

        # if response.status_code != 200:
        #     continue
        #
        # image_data = response.json().get('image_base64')
        # if not image_data:
        #     continue

        # image_bytes = base64.b64decode(image_data)
        # image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        # image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is not None:
            roboflow_predictions = get_model_predications_result(image)
            spot_occupancy_result = evaluate_occupied_spots_in_area(spots, roboflow_predictions)
            result_list = [obj.__dict__ for obj in spot_occupancy_result]
            all_occupancy_results[area] = result_list

    return all_occupancy_results


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
