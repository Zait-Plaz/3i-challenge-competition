pip install gradio ultralytics tensorflow opencv-python numpy
import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO

yolo_model = YOLO("yolov8n.pt")

keras_model = tf.keras.models.load_model("cnn.keras")
input_height, input_width = keras_model.input_shape[1:3]

target_labels = [
    "Ca hu kho", "Canh cai", "Canh chua", "Com trang", "Dau hu sot ca",
    "Ga chien", "Rau muong xao", "Thit kho", "Thit kho trung", "Trung chien"
]

price_table = {
    "Ca hu kho": 25000,
    "Canh cai": 10000,
    "Canh chua": 30000,
    "Com trang": 5000,
    "Dau hu sot ca": 25000,
    "Ga chien": 20000,
    "Rau muong xao": 8000,
    "Thit kho": 25000,
    "Thit kho trung": 25000,
    "Trung chien": 6000
}

def detect_and_classify(image):
    try:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = yolo_model(image_bgr)
        detections = results[0].boxes.data.cpu().numpy()

        seen_classes = set()

        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            if score < 0.3:
                continue

            crop = image_bgr[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue

            resized = cv2.resize(crop, (input_width, input_height))
            input_data = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)
            predictions = keras_model.predict(input_data, verbose=0)
            predicted_index = int(np.argmax(predictions[0]))
            predicted_label = target_labels[predicted_index]

            seen_classes.add(predicted_label)

        if not seen_classes:
            return "⚠️ Không phát hiện được món ăn", "0đ"

        result_text = "\\n".join(
            [f"🍽️ {food}: {price_table[food]:,}đ" for food in sorted(seen_classes)]
        )
        total_price = sum([price_table[food] for food in seen_classes])
        return result_text, f"💰 Tổng cộng: {total_price:,}đ"

    except Exception as e:
        return f"❌ Lỗi: {str(e)}", "0đ"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🍱 Nhận Diện Món Ăn & Tính Tiền - Web App (.keras)")
    with gr.Row():
        image_input = gr.Image(type="numpy", label="📷 Ảnh khay cơm")
    with gr.Row():
        food_output = gr.Textbox(label="📋 Món ăn và giá", lines=10)
        total_output = gr.Textbox(label="💵 Tổng hóa đơn")
    btn = gr.Button("🚀 Nhận diện & Tính tiền")
    btn.click(fn=detect_and_classify, inputs=image_input, outputs=[food_output, total_output])
    gr.Markdown("<center>© 2025 - Đồ án AI - UEH</center>")

if __name__ == "__main__":
    demo.launch()
