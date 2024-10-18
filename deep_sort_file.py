from deep_sort_realtime.deepsort_tracker import DeepSort
from deep_sort_realtime.deepsort_tracker import Detection
from deep_sort_realtime.deepsort_tracker import Tracker
from deep_sort_realtime.deepsort_tracker import nn_matching
import cv2
from ultralytics import YOLO, solutions
import torch
import numpy as np
from collections import defaultdict


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

# Initialize DeepSORT
max_cosine_distance = 0.3
nn_budget = None
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
tracker = Tracker(metric)

# Set model parameters
model.overrides['conf'] = 0.5  # NMS confidence threshold
model.overrides['iou'] = 0.5  # NMS IoU threshold
model.overrides['agnostic_nms'] = True  # NMS class-agnostic
model.overrides['max_det'] = 1000  # maximum number of detections per image

# Store scores for each person-luggage pair using tracker ID
ownership_scores = defaultdict(lambda: defaultdict(int))


def calculate_distance(depth_map, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    dist_2d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / pixels_per_meter
    z1 = depth_map[int(y1), int(x1)] / pixels_per_meter
    z2 = depth_map[int(y2), int(x2)] / pixels_per_meter
    depth_diff = np.abs(z1 - z2)
    distance = np.sqrt(dist_2d ** 2 + depth_diff ** 2)
    return distance


def get_centroid(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def process_video(video_source):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    owners = {}  # Store assigned owners for luggage using tracker ID
    abandoned_luggages = set()  # Store abandoned luggage using tracker ID

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        if frame_count % 10 != 0:
            continue

        results = model(frame, classes=[0, 28, 24, 26], show=False)

        # Convert YOLO detections to DeepSORT format
        detections = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = box.conf.cpu().numpy()[0]
            cls = box.cls.cpu().numpy()[0]
            detection = Detection(xyxy, conf, cls)
            detections.append(detection)

        # Update DeepSORT tracker
        tracker.predict()
        tracker.update(detections)

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

        persons = []
        luggages = []

        # Process tracked objects
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            #class_id = track.ge
            class_id = int(detections[len(detections) - 1].feature.max())
            track_id = track.track_id

            if class_id == 0:
                persons.append((track_id, get_centroid(bbox)))
            elif class_id in [24, 28, 26]:
                luggages.append((track_id, get_centroid(bbox)))

        for person_id, person_centroid in persons:
            for luggage_id, luggage_centroid in luggages:
                distance_m = calculate_distance(depth_map, person_centroid, luggage_centroid)
                if distance_m <= unattended_threshold and luggage_id not in abandoned_luggages:
                    ownership_scores[luggage_id][person_id] += 1

        for luggage_id, luggage_centroid in luggages:
            person_in_range = any(
                calculate_distance(depth_map, person_centroid, luggage_centroid) <= unattended_threshold
                for person_id, person_centroid in persons
            )
            print(f"Luggage ID: {luggage_id}, Person in Range: {person_in_range}")

            if not person_in_range and luggage_id not in abandoned_luggages:
                print(f"Luggage with ID {luggage_id} is unattended!")
                abandoned_luggages.add(luggage_id)

        for luggage_id, scores in ownership_scores.items():
            if luggage_id not in owners:
                owner, max_score = max(scores.items(), key=lambda x: x[1], default=(None, 0))
                if owner is not None:
                    owners[luggage_id] = owner

        for luggage_id, owner_id in list(owners.items()):
            owner_present = any(
                calculate_distance(depth_map, person_centroid, luggage_centroid) <= unattended_threshold
                for person_id, person_centroid in persons if person_id == owner_id
            )
            if not owner_present:
                print(f"Luggage with ID {luggage_id} is abandoned!")
                # Find the bounding box of the abandoned luggage and annotate the frame
                for track in tracker.tracks:
                    if track.track_id == luggage_id:
                        bbox = track.to_tlbr()
                        xyxy = bbox.astype(int)

                        rect_width = xyxy[2] - xyxy[0]
                        center_bottom = (xyxy[0] + rect_width // 2, xyxy[3])

                        text = "Unattended"
                        font_scale = 0.57
                        thickness = 1

                        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                        text_position = (
                            center_bottom[0] - text_size[0] // 2,
                            center_bottom[1] + text_size[1] + 5
                        )

                        cv2.rectangle(
                            frame,
                            (text_position[0] - 8, text_position[1] - text_size[1] - 5),
                            (text_position[0] + text_size[0] + 8, text_position[1] + 5),
                            (0, 0, 255),
                            -1
                        )

                        cv2.putText(
                            frame,
                            text,
                            text_position,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA
                        )

                        break
                abandoned_luggages.add(luggage_id)
                del owners[luggage_id]

        # Visualization
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr().astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            centroid = get_centroid(bbox)
            cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 5, (0, 255, 0), -1)

        cv2.imshow('Suspicious Objects', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "sample3.mp4"
    process_video(video_path)
