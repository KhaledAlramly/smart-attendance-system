import cv2
import torch
import csv
import os
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


DATASET_DIR = "./models/model"


mtcnn = MTCNN(image_size=160, margin=14, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


THRESHOLD = 1.0


print("⏳ Loading dataset and creating embeddings...")
student_embeddings = {}

for student_name in os.listdir(DATASET_DIR):
    student_folder = os.path.join(DATASET_DIR, student_name)
    if not os.path.isdir(student_folder):
        continue

    embeddings_list = []
    for img_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_name)
        try:
            img = Image.open(img_path).convert('RGB')
            faces = mtcnn(img)
            if faces is not None:
                if faces.ndim == 3:
                    faces = faces.unsqueeze(0)  # [1,C,H,W]
                if faces.shape[1] == 1:
                    faces = faces.repeat(1, 3, 1, 1)
                embeddings_list.append(resnet(faces).detach())
        except:
            print(f"❌ {student_name} - {img_name} No face detected")
            continue

    if embeddings_list:
        student_embeddings[student_name] = embeddings_list
        print(f"✅ {student_name}: {len(embeddings_list)} valid embeddings")
    else:
        print(f"⚠️ No valid face detected for {student_name}")

print(f"✅ Loaded {len(student_embeddings)} students.")


def new_attendance_file():
    today = datetime.now().strftime("%Y-%m-%d")
    filename = f"attendance_{today}.csv"
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Time"])
    return filename

attendance_file = new_attendance_file()
marked_students = {}
unknown_count = 0


correct_recognitions = 0
total_detections = 0


with open(attendance_file, 'r') as f:
    reader = csv.reader(f)
    next(reader, None)
    for row in reader:
        if row:
            name, time = row
            marked_students[name] = time


def print_dashboard():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"=== LIVE ATTENDANCE DASHBOARD ({datetime.now().strftime('%Y-%m-%d')}) ===\n")
    print("{:<20} {:<20}".format("Name", "Time"))
    print("-" * 40)
    for name, time in marked_students.items():
        print("{:<20} {:<20}".format(name, time))
    print("\nTotal Students Present:", len(marked_students))
    print("Unknown Faces Detected:", unknown_count)


    if total_detections > 0:
        accuracy = (correct_recognitions / total_detections) * 100
        print(f"Recognition Accuracy: {accuracy:.2f}%")
    else:
        print("Recognition Accuracy: N/A")

    print("\nPress 'q' on the webcam window to quit.")

print("Starting Face Recognition. Press 'q' to quit.")
print_dashboard()


cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR:Camera Can't open")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        faces = mtcnn(rgb)
        if faces is not None:
            if not isinstance(faces, list):
                faces = [faces]

            for face, box in zip(faces, boxes):
                if face is None:
                    continue


                if face.ndim == 3:
                    face = face.unsqueeze(0)
                if face.shape[1] == 1:
                    face = face.repeat(1, 3, 1, 1)
                elif face.shape[1] == 4:
                    face = face[:, :3, :, :]


                with torch.no_grad():
                    embedding = resnet(face)
                    name = "Unknown"
                    min_dist = float('inf')
                    for student_name, embeddings_list in student_embeddings.items():
                        for emb in embeddings_list:
                            dist = (embedding - emb).norm().item()
                            if dist < THRESHOLD and dist < min_dist:
                                min_dist = dist
                                name = student_name


                total_detections += 1
                if name != "Unknown":
                    correct_recognitions += 1


                x1, y1, x2, y2 = [int(b) for b in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


                if name == "Unknown":
                    unknown_count += 1
                elif name not in marked_students:
                    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    marked_students[name] = now_time
                    with open(attendance_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([name, now_time])

            print_dashboard()

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()