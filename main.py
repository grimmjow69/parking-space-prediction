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
            ParkingSpot("FRONT_D_1", [[1191, 283], [1256, 283], [1256, 341], [1191, 341]]),
            ParkingSpot("FRONT_D_2", [[1139, 286], [1196, 286], [1196, 335], [1139, 335]]),
            ParkingSpot("FRONT_D_3", [[1080, 284], [1137, 284], [1137, 335], [1080, 335]]),
            ParkingSpot("FRONT_D_4", [[1022, 282], [1085, 282], [1085, 334], [1022, 334]]),
            ParkingSpot("FRONT_D_5", [[969, 285], [1031, 285], [1031, 333], [969, 333]]),
            ParkingSpot("FRONT_D_6", [[916, 283], [978, 283], [978, 335], [916, 335]]),
            ParkingSpot("FRONT_D_7", [[866, 290], [926, 290], [926, 333], [866, 333]]),
            ParkingSpot("FRONT_D_8", [[827, 287], [877, 287], [877, 330], [827, 330]]),
            ParkingSpot("FRONT_D_9", [[772, 290], [827, 290], [827, 334], [772, 334]]),
            ParkingSpot("FRONT_D_10", [[723, 293], [776, 293], [776, 333], [723, 333]]),
            ParkingSpot("FRONT_D_11", [[666, 284], [720, 284], [720, 338], [666, 338]]),

            ParkingSpot("FRONT_A_19", [[1091, 781], [1536, 781], [1536, 987], [1091, 987]]),
            ParkingSpot("FRONT_A_20", [[1094, 703], [1507, 703], [1507, 888], [1094, 888]]),
            ParkingSpot("FRONT_A_21", [[1129, 659], [1481, 659], [1481, 795], [1129, 795]]),
            ParkingSpot("FRONT_A_22", [[1131, 595], [1444, 595], [1444, 704], [1131, 704]]),
            ParkingSpot("FRONT_A_23", [[1132, 571], [1437, 571], [1437, 670], [1132, 670]]),
            ParkingSpot("FRONT_A_24", [[1149, 538], [1418, 538], [1418, 624], [1149, 624]]),
            ParkingSpot("FRONT_A_25", [[1154, 515], [1424, 515], [1424, 607], [1154, 607]]),
            ParkingSpot("FRONT_A_26", [[1164, 490], [1405, 490], [1405, 567], [1164, 567]]),
            ParkingSpot("FRONT_A_27", [[1172, 466], [1397, 466], [1397, 544], [1172, 544]]),
            ParkingSpot("FRONT_A_28", [[1173, 452], [1386, 452], [1386, 513], [1173, 513]]),
            ParkingSpot("FRONT_A_29", [[1200, 433], [1396, 433], [1396, 498], [1200, 498]]),
            ParkingSpot("FRONT_A_30", [[1202, 416], [1387, 416], [1387, 468], [1202, 468]]),
            ParkingSpot("FRONT_A_31", [[1211, 405], [1388, 405], [1388, 446], [1211, 446]]),
            ParkingSpot("FRONT_A_32", [[1207, 389], [1403, 389], [1403, 436], [1207, 436]]),
            ParkingSpot("FRONT_A_33", [[1218, 376], [1388, 376], [1388, 427], [1218, 427]]),
            ParkingSpot("FRONT_A_34", [[1229, 367], [1386, 367], [1386, 403], [1229, 403]]),
            ParkingSpot("FRONT_A_35", [[1231, 361], [1358, 361], [1358, 396], [1231, 396]]),
            ParkingSpot("FRONT_A_36", [[1236, 346], [1361, 346], [1361, 385], [1236, 385]]),

            ParkingSpot("FRONT_B_1", [[572, 796], [984, 796], [984, 941], [572, 941]]),
            ParkingSpot("FRONT_B_2", [[653, 691], [1024, 691], [1024, 832], [653, 832]]),
            ParkingSpot("FRONT_B_3", [[740, 616], [1073, 616], [1073, 745], [740, 745]]),
            ParkingSpot("FRONT_B_4", [[789, 574], [1087, 574], [1087, 688], [789, 688]]),
            ParkingSpot("FRONT_B_5", [[829, 548], [1093, 548], [1093, 631], [829, 631]]),
            ParkingSpot("FRONT_B_6", [[890, 516], [1130, 516], [1130, 603], [890, 603]]),
            ParkingSpot("FRONT_B_7", [[925, 489], [1145, 489], [1145, 573], [925, 573]]),
            ParkingSpot("FRONT_B_8", [[947, 469], [1145, 469], [1145, 540], [947, 540]]),
            ParkingSpot("FRONT_B_9", [[973, 449], [1158, 449], [1158, 513], [973, 513]]),
            ParkingSpot("FRONT_B_10", [[981, 432], [1167, 432], [1167, 493], [981, 493]]),
            ParkingSpot("FRONT_B_11", [[1008, 419], [1182, 419], [1182, 485], [1008, 485]]),
            ParkingSpot("FRONT_B_12", [[1012, 402], [1176, 402], [1176, 462], [1012, 462]]),
            ParkingSpot("FRONT_B_13", [[1030, 387], [1182, 387], [1182, 436], [1030, 436]]),
            ParkingSpot("FRONT_B_14", [[1053, 379], [1185, 379], [1185, 434], [1053, 434]]),
            ParkingSpot("FRONT_B_15", [[1061, 363], [1193, 363], [1193, 411], [1061, 411]]),
            ParkingSpot("FRONT_B_16", [[1079, 353], [1204, 353], [1204, 399], [1079, 399]]),
            ParkingSpot("FRONT_B_17", [[1088, 339], [1211, 339], [1211, 383], [1088, 383]]),
            ParkingSpot("FRONT_B_18", [[1089, 327], [1223, 327], [1223, 367], [1089, 367]]),

            ParkingSpot("FRONT_B_19", [[1, 723], [240, 723], [240, 871], [1, 871]]),
            ParkingSpot("FRONT_B_20", [[24, 656], [346, 656], [346, 799], [24, 799]]),
            ParkingSpot("FRONT_B_21", [[134, 613], [433, 613], [433, 729], [134, 729]]),
            ParkingSpot("FRONT_B_22", [[221, 576], [496, 576], [496, 680], [221, 680]]),
            ParkingSpot("FRONT_B_23", [[315, 540], [568, 540], [568, 643], [315, 643]]),
            ParkingSpot("FRONT_B_24", [[392, 509], [613, 509], [613, 603], [392, 603]]),
            ParkingSpot("FRONT_B_25", [[440, 482], [665, 482], [665, 570], [440, 570]]),
            ParkingSpot("FRONT_B_26", [[490, 472], [701, 472], [701, 544], [490, 544]]),
            ParkingSpot("FRONT_B_27", [[549, 451], [746, 451], [746, 518], [549, 518]]),
            ParkingSpot("FRONT_B_28", [[584, 432], [779, 432], [779, 496], [584, 496]]),
            ParkingSpot("FRONT_B_29", [[632, 413], [799, 413], [799, 481], [632, 481]]),
            ParkingSpot("FRONT_B_30", [[665, 396], [825, 396], [825, 456], [665, 456]]),
            ParkingSpot("FRONT_B_31", [[698, 383], [852, 383], [852, 442], [698, 442]]),
            ParkingSpot("FRONT_B_32", [[714, 368], [870, 368], [870, 427], [714, 427]]),
            ParkingSpot("FRONT_B_33", [[733, 361], [885, 361], [885, 411], [733, 411]]),
            ParkingSpot("FRONT_B_34", [[757, 347], [907, 347], [907, 402], [757, 402]]),
            ParkingSpot("FRONT_B_35", [[783, 337], [907, 337], [907, 388], [783, 388]]),
            ParkingSpot("FRONT_B_36", [[810, 328], [928, 328], [928, 379], [810, 379]]),
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
    model = YOLO('best2.pt')
    results = model(image, conf=0.3, iou=0.7)

    predictions = []

    for result in results:
        result.show();
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
        # image = 'images/...'

        if image is not None:
            roboflow_predictions = get_model_predications_result(image)
            spot_occupancy_result = evaluate_occupied_spots_in_area(spots, roboflow_predictions)
            result_list = [obj.__dict__ for obj in spot_occupancy_result]
            all_occupancy_results[area] = result_list

    return all_occupancy_results


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
