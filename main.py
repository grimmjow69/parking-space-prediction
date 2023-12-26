import cv2
from roboflow import Roboflow
import numpy as np


def load_image(file_path):
    image = cv2.imread(file_path)
    return image


def draw_spots(image, parking_spots, predictions):
    occupied_spots = []
    for spot in parking_spots:
        spot_points = np.array(spot, np.int32)
        spot_points = spot_points.reshape((-1, 1, 2))

        # Initialize as not occupied
        occupied = False

        # Initialize spot as green (no car)
        color = (0, 255, 0)

        # Check each prediction to see if it's inside the current spot
        for pred in predictions:
            if pred['class_id'] == 0:  # Assuming class_id 0 is a car
                point = (int(pred['x']), int(pred['y']))
                # Check if the point is inside the polygon

                if cv2.pointPolygonTest(spot_points, point, False) >= 0:
                    color = (0, 0, 255)  # Car detected, change color of spot to red
                    occupied = True
                    break  # No need to check other predictions for this spot

        # Drawing of parking spot
        cv2.polylines(image, [spot_points], isClosed=True, color=color, thickness=1)
        occupied_spots.append(occupied)

    return occupied_spots


def display_image(image):
    cv2.imshow("Parking Spots", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def model_prediction(api_key, project_name, image_path, parking_spots):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace().project(project_name)
    model = project.version(1).model

    # Predict using the model
    response = model.predict(image_path, confidence=10, overlap=40)
    predictions = response.json()['predictions']

    image = load_image(image_path)

    if image is not None:
        draw_spots(image, parking_spots, predictions)
        display_image(image)
    else:
        print("Error: The image could not be loaded.")


# Example of usage
api_key = "..."
project_name = "car_detection-l1xo4"
image_path = "parking.jpg"

# Example of few manually marked spots
parking_spots = [
    [(451, 518), (559, 526), (559, 488), (459, 483)],
    [(460, 483), (559, 488), (559, 463), (468, 460)],
    [(470, 276), (491, 276), (493, 262), (474, 261)],
    [(490, 276), (511, 276), (513, 260), (494, 260)],

    [(512, 277), (534, 277), (534, 260), (513, 260)],
    [(534, 277), (555, 277), (555, 260), (534, 260)],
    [(780, 288), (802, 287), (785, 270), (769, 270)],
    [(711, 285), (735, 285), (723, 268), (702, 268)],
]

model_prediction(api_key, project_name, image_path, parking_spots)
