from ultralytics import YOLO


if __name__ == "__main__":

    # Load the model with the pre-trained weights
    model = YOLO("yolov8n-pose.pt")

    # Estimate the pose of the user in the webcam
    model.predict(source=1, show=True)
