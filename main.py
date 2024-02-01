import json

import cv2
from roboflow import Roboflow
from parking_spot import ParkingSpot
from parking_spot_result import ParkingSpotResult


def read_parking_spot_image(file_path):
    image = cv2.imread(file_path)
    return image


def annotate_and_identify_occupied_spots(image, parking_spots, predictions):
    occupied_spots = []
    for spot in parking_spots:
        spot_polygon = spot.get_polygon()

        occupied = False

        # Initialize spot as green (no car)
        color = (0, 255, 0)

        for pred in predictions:
            if pred['class_id'] == 0:  # Assuming class_id 0 is a car
                point = (int(pred['x']), int(pred['y']))

                # Check if the point is inside the polygon
                if cv2.pointPolygonTest(spot_polygon, point, False) >= 0:
                    color = (0, 0, 255)  # Car detected, change color of spot to red
                    occupied = True
                    break  # No need to check other predictions for this spot

        # Drawing of parking spot
        cv2.polylines(image, [spot_polygon], isClosed=True, color=color, thickness=1)
        occupied_spots.append(ParkingSpotResult(spot.name, occupied))

    return occupied_spots


def show_annotated_parking_image(image):
    cv2.imshow("Parking Spots", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_parking_spot_image(api_key, project_name, image_path, parking_spots):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    model = project.version(1).model

    # Predict using the pretrained model from roboflow
    response = model.predict(image_path, confidence=1, overlap=80)
    predictions = response.json()['predictions']

    image = read_parking_spot_image(image_path)

    if image is not None:
        spot_occupancy_result = annotate_and_identify_occupied_spots(image, parking_spots, predictions)
        show_annotated_parking_image(image)
        return spot_occupancy_result
    else:
        print("Error: The image could not be loaded.")


# Example of usage
api_key = "..."
project_name = "car_detection-l1xo4"
image_path = "images/zalivautickadenzasnezene2.png"

# Example of few manually marked spots
parking_spots = [
    ParkingSpot("Spot1", [[732, 339], [895, 339], [895, 409], [732, 409]]),
    ParkingSpot("Spot2", [[767, 258], [897, 258], [897, 307], [767, 307]]),
    ParkingSpot("Spot3", [[1029, 251], [1168, 251], [1168, 299], [1029, 299]]),
    ParkingSpot("Spot4", [[1040, 290], [1188, 290], [1188, 337], [1040, 337]]),
    ParkingSpot("Spot5", [[1047, 331], [1205, 331], [1205, 387], [1047, 387]]),
    ParkingSpot("Spot6", [[1071, 373], [1203, 373], [1203, 417], [1071, 417]]),
    ParkingSpot("Spot7", [[1082, 409], [1230, 409], [1230, 461], [1082, 461]]),
    ParkingSpot("Spot8", [[1077, 449], [1246, 449], [1246, 505], [1077, 505]]),
    ParkingSpot("Spot9", [[1084, 488], [1255, 488], [1255, 558], [1084, 558]]),
    ParkingSpot("Spot10", [[1101, 539], [1277, 539], [1277, 634], [1101, 634]]),
    ParkingSpot("Spot11", [[1100, 627], [1296, 627], [1296, 713], [1100, 713]]),
    ParkingSpot("Spot12", [[1113, 702], [1310, 702], [1310, 786], [1113, 786]]),
    ParkingSpot("Spot13", [[1114, 772], [1330, 772], [1330, 877], [1114, 877]]),
    ParkingSpot("Spot14", [[1136, 874], [1368, 874], [1368, 1023], [1136, 1023]]),
    ParkingSpot("Spot15", [[578, 972], [827, 972], [827, 1048], [578, 1048]]),
    ParkingSpot("Spot16", [[582, 859], [825, 859], [825, 976], [582, 976]]),
    ParkingSpot("Spot17", [[608, 765], [822, 765], [822, 868], [608, 868]]),
    ParkingSpot("Spot18", [[628, 686], [834, 686], [834, 775], [628, 775]]),
    ParkingSpot("Spot19", [[650, 620], [847, 620], [847, 696], [650, 696]]),
    ParkingSpot("Spot20", [[651, 544], [859, 544], [859, 627], [651, 627]]),
    ParkingSpot("Spot21", [[670, 494], [860, 494], [860, 581], [670, 581]]),
    ParkingSpot("Spot22", [[682, 448], [861, 448], [861, 526], [682, 526]]),
    ParkingSpot("Spot23", [[698, 405], [856, 405], [856, 464], [698, 464]]),
    ParkingSpot("Spot24", [[720, 375], [872, 375], [872, 428], [720, 428]])
]

occupancy_result = process_parking_spot_image(api_key, project_name, image_path, parking_spots)
json_result = json.dumps([obj.__dict__ for obj in occupancy_result])

print(json_result)
