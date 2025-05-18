pip install gradio ultralytics tensorflow opencv-python numpy
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

try:
    yolo_model = YOLO("yolov8n.pt")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load YOLO model: {e}")

try:
    keras_model = tf.keras.models.load_model("cnn.keras")
    input_height, input_width = keras_model.input_shape[1:3]
except Exception as e:
    raise RuntimeError(f"❌ Failed to load Keras model: {e}")

target_labels = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

price_table = {
    "Ca hu kho": 22000,
    "Canh cai": 9000,
    "Canh chua": 10000,
    "Com": 5000,
    "Dau hu sot ca": 16000,
    "Ga chien": 25000,
    "Rau muong xao": 8000,
    "Thit kho": 17000,
    "Thit kho trung": 18000,
    "Trung chien": 12000
}

def detect_and_classify(image_bgr):
    seen_classes = set()
    h, w, _ = image_bgr.shape
    results = yolo_model(image_bgr)
    detections = results[0].boxes.data.cpu().numpy()

    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        if score < 0.3:
            continue

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        crop = image_bgr[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (input_width, input_height))
        input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
        predictions = keras_model.predict(input_data, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        predicted_label = target_labels[predicted_index]
        seen_classes.add(predicted_label)

    return seen_classes

def run_camera_feed():
    cap = cv2.VideoCapture(0)  

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera feed not available.")
            break

        detected_labels = detect_and_classify(frame)

        if detected_labels:
            y = 40
            total_price = sum(price_table[label] for label in detected_labels)
            for label in sorted(detected_labels):
                text = f"{label}: {price_table[label]:,}đ"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                y += 30

            cv2.putText(frame, f"Tổng cộng: {total_price:,}đ", (10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Không phát hiện được món ăn", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow("🍱 Nhận diện món ăn", frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_camera_feed()

    btn.click(fn=detect_and_classify, inputs=image_input, outputs=[food_output, total_output])
    gr.Markdown("<center>© 2025 - Đồ án AI - UEH</center>")

if __name__ == "__main__":
    demo.launch()
