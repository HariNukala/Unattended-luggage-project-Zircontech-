import cv2
from ultralytics import YOLO, solutions
import torch
import numpy as np
from collections import defaultdict
import gradio as gr
import tempfile
import os

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

# Load MiDaS model for depth estimation
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

# Load YOLO model
model = YOLO('yolov8x.pt')
names = model.model.names
model.to(device)

pixels_per_meter = 300
unattended_threshold = 2.0  # meters

dist_obj = solutions.DistanceCalculation(names=names, view_img=False, pixels_per_meter=pixels_per_meter)

# Set model parameters
model.overrides['conf'] = 0.5  # NMS confidence threshold
model.overrides['iou'] = 0.5  # NMS IoU threshold
model.overrides['agnostic_nms'] = True  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# Store scores for each person-luggage pair using tracker ID
ownership_scores = defaultdict(lambda: defaultdict(int))


def calculate_distance(depth_map, point1, point2):
    dist_2d_m, dist_2d_mm = dist_obj.calculate_distance(point1, point2)
    z1 = depth_map[int(point1[1]), int(point1[0])] / pixels_per_meter
    z2 = depth_map[int(point2[1]), int(point2[0])] / pixels_per_meter
    depth_diff = np.abs(z1 - z2)
    distance = np.sqrt(dist_2d_m ** 2 + depth_diff ** 2)
    return distance


def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    owners = {}  # Store assigned owners for luggage using tracker ID
    abandoned_luggages = set()  # Store abandoned luggage using tracker ID

    frame_count = 0
    output_frames = []  # Store the processed frames to return as video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % 10 != 0:
            continue

        # Process frame with YOLO
        results = model.track(frame, persist=True, classes=[0, 28, 24, 26], show=False)
        frame_ = results[0].plot()

        # MiDaS depth estimation
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = midas_transforms(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            depth_map = prediction.cpu().numpy()

        # Extract objects and calculate distances
        persons = []
        luggages = []
        num_boxes = len(results[0].boxes)
        for i in range(num_boxes):
            box = results[0].boxes[i]
            centroid = get_centroid(box)
            track_id = box.id
            if box.cls == 0:
                persons.append((track_id, centroid))
            elif box.cls in [24, 28, 26]:
                luggages.append((track_id, centroid))

        for person_id, person_centroid in persons:
            for luggage_id, luggage_centroid in luggages:
                distance_m = calculate_distance(depth_map, person_centroid, luggage_centroid)
                if distance_m <= unattended_threshold and luggage_id not in abandoned_luggages:
                    ownership_scores[luggage_id][person_id] += 1

        # Check for abandoned luggage
        for luggage_id, luggage_centroid in luggages:
            person_in_range = any(
                calculate_distance(depth_map, person_centroid, luggage_centroid) <= unattended_threshold
                for person_id, person_centroid in persons
            )
            if not person_in_range and luggage_id not in abandoned_luggages:
                abandoned_luggages.add(luggage_id)

        # Visualization
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            cv2.rectangle(frame_, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            centroid = get_centroid(box)
            cv2.circle(frame_, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

            # Check if the object is a luggage and is abandoned
        if box.cls in [24, 28, 26] and box.id in abandoned_luggages:
            # Add text "Unattended Luggage" near the luggage
            cv2.putText(frame_, "Unattended Luggage", (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)  # Red color text
        output_frames.append(frame_)

    cap.release()
    return output_frames


def get_centroid(box):
    return dist_obj.calculate_centroid(box.xyxy[0].cpu().numpy().astype(int))


def video_interface(video_path):
    processed_frames = process_video(video_path)
    if not processed_frames:
        return None

    # Save processed frames as a video
    height, width, _ = processed_frames[0].shape
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(temp_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for frame in processed_frames:
        out.write(frame)

    out.release()

    # Provide both video playback and download
    if os.path.getsize(temp_file.name) > 50 * 1024 * 1024:  # If video is larger than 50MB, provide download
        return {"output": temp_file.name, "message": "The video is large. Click the link to download."}

    return temp_file.name


# Create a Gradio interface
def gradio_interface(video_path):
    result = video_interface(video_path)
    if isinstance(result, dict):
        return result['output'], result['message']
    return result, None


interface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Video(format="mp4"),  # No need for `source`
    outputs=["video", "text"],
    title="Abandoned Object Detection"
)

if __name__ == "__main__":
    interface.queue(max_size=20).launch(
        server_name="127.0.0.1",  # Change this to "127.0.0.1" if you want local access only
        server_port=7860,  # Specify a port to run the server (default is 7860)
        debug=True,  # Enable debugging mode
        share=True  # Set `share=True` to create a public shareable link for testing (if required)
    )
