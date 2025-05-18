import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
import os
import uuid
from datetime import datetime
from ultralytics import YOLO

os.makedirs("output", exist_ok=True)
yolo_model = YOLO("Interface/yolov8n.pt")
keras_model = tf.keras.models.load_model("Interface/cnn.keras")
input_height, input_width = keras_model.input_shape[1:3]

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

def detect_and_classify(image):
    max_size = 1024
    if max(image.shape[:2]) > max_size:
        ratio = max_size / max(image.shape[:2])
        image = cv2.resize(image, (int(image.shape[1]*ratio), int(image.shape[0]*ratio)))

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image_bgr.shape
    results = yolo_model(image_bgr)
    detections = results[0].boxes.data.cpu().numpy()
    seen_classes = set()

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

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image_bgr, predicted_label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if not seen_classes:
        return "⚠️ Không phát hiện được món ăn!", "0đ", image, None

    result_text = "\n".join(
        [f"🍽️ {label}: {price_table[label]:,}đ" for label in sorted(seen_classes)]
    )
    total_price = sum([price_table[label] for label in seen_classes])
    total_text = f"💰 Tổng cộng: {total_price:,}đ"

    save_path = f"output/monan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg"
    cv2.imwrite(save_path, image_bgr)

    return result_text, total_text, cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), "Interface/QR.jpg"

with gr.Blocks(title="Nhận Diện Món Ăn - UEH", theme=gr.themes.Soft(primary_hue="blue")) as demo:
    with gr.Row():
        gr.Image(
            value="Interface/logo-light.png",
            show_label=False,
            show_download_button=False,
            height=60
        )
    gr.HTML("<h1 style='text-align:center;'> ỨNG DỤNG NHẬN DIỆN MÓN ĂN </h1>")
    with gr.Group():
        with gr.Row():
            image_input = gr.Image(type="numpy", label=" Ảnh đầu vào ", scale=1, visible=True)
            image_output = gr.Image(label=" Kết quả ", scale=1, visible=False)

        btn = gr.Button(" Bắt đầu ")
        status_label = gr.Markdown("", visible=False)

        with gr.Row():
            food_output = gr.Textbox(label="📋 Món ăn và giá", lines=10, visible=False)
            total_output = gr.Textbox(label="💵 Tổng hóa đơn", max_lines=1, visible=False)

        qr_output = gr.Image(label=" Mã QR thanh toán ", visible=False, height=200)
        done_btn = gr.Button("✅ Tôi đã thanh toán")

    def handle_input(img):
        if img is None:
            return "❌ Vui lòng tải ảnh!", "0đ", None, None, gr.update(value="Vui lòng tải ảnh!", visible=True)
        result_text, total_text, processed_image, qr_path = detect_and_classify(img)
        return (
            gr.update(value=result_text, visible=True),
            gr.update(value=total_text, visible=True),
            gr.update(value=processed_image, visible=True),
            gr.update(value=qr_path, visible=True),
            gr.update(value="", visible=False)
        )

    def reset_ui():
        return (
            gr.update(value=None, visible=True),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(value="", visible=False),
            gr.update(visible=False),
            gr.update(value="", visible=False)
        )

    def show_loading():
        return gr.update(value="⏳ Đang xử lý ảnh, vui lòng chờ trong giây lát...", visible=True)

    btn.click(fn=show_loading, inputs=[], outputs=[status_label])\
       .then(fn=handle_input, inputs=image_input,
             outputs=[food_output, total_output, image_output, qr_output, status_label])

    done_btn.click(
        fn=reset_ui,
        inputs=[],
        outputs=[image_input, image_output, food_output, total_output, qr_output, status_label]
    )

    gr.HTML(
        "<p style='text-align:center; font-size:12px; color:gray'>Ảnh sau xử lý sẽ được lưu vào thư mục <code>output/</code>.</p>"
    )
    gr.HTML("<center style='color:gray'>© 2025 - Đồ án Trí tuệ Nhân tạo - UEH</center>")

if __name__ == "__main__":
    demo.launch(inbrowser=True)
