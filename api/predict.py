# api/predict.py

from flask import Flask, request, jsonify
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import io # สำหรับจัดการ binary data ของรูปภาพ

app = Flask(__name__)

# --- กำหนดค่าคงที่สำหรับโมเดลของคุณ ---
# ชื่อไฟล์โมเดลที่บันทึกไว้ในโฟลเดอร์ 'models/'
# ตรวจสอบให้แน่ใจว่านามสกุล .h5 หรือ .keras ตรงกับไฟล์จริงของคุณ
MODEL_FILENAME = "durian.h5"

# ชื่อคลาสของโรคใบเรียน (ต้องเรียงลำดับเดียวกับที่โมเดลถูก Train มา)
CLASS_NAMES = ['Anthracnose', 'Algal Spot', 'Healthy', 'Leaf Blight', 'Powdery Mildew']

# ขนาดภาพที่โมเดลคาดหวัง (MobileNetV2 ใช้ 224x224)
IMAGE_SIZE = (224, 224)

# ตัวแปร Global สำหรับเก็บโมเดล เพื่อให้โหลดเพียงครั้งเดียว
model = None

# --- ฟังก์ชันสำหรับโหลดโมเดล AI ---
def load_ai_model():
    """
    โหลดโมเดล TensorFlow/Keras จากไฟล์.
    จะโหลดเพียงครั้งเดียวเมื่อฟังก์ชันเริ่มทำงาน (Cold Start).
    """
    global model
    if model is None:
        model_path = os.path.join(os.getcwd(), 'models', MODEL_FILENAME)
        print(f"Attempting to load model from: {model_path}")
        try:
            # compile=False เพื่อป้องกันปัญหาที่อาจเกิดขึ้นกับการโหลด Optimizer
            # เนื่องจากเราเพียงแค่ต้องการทำนายผล
            model = tf.keras.models.load_model(model_path, compile=False)
            print(f"AI Model '{MODEL_FILENAME}' loaded successfully.")

            # ตรวจสอบว่า TensorFlow กำลังใช้ GPU หรือ CPU
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                print("TensorFlow is using GPU (though Vercel Free Tier won't provide it).")
                # สำหรับ Vercel Free Tier จะไม่ใช้ GPU ดังนั้นบรรทัดนี้จะไม่ได้แสดง
                # แต่โค้ดนี้ยังคงอยู่เพื่อความสมบูรณ์ในการตรวจสอบ environment
            else:
                print("TensorFlow is using CPU (expected on Vercel Free Tier).")
        except Exception as e:
            print(f"Error loading model '{MODEL_FILENAME}': {e}")
            model = None # ตั้งค่าเป็น None เพื่อบ่งบอกว่าโหลดไม่สำเร็จ
    return model

# โหลดโมเดลทันทีที่ Script ถูกโหลด (เมื่อ Serverless Function "Warm Up" ครั้งแรก)
load_ai_model()


# --- Endpoint สำหรับการทำนายผล ---
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    รับไฟล์รูปภาพผ่าน POST request, ประมวลผลด้วยโมเดล AI
    และส่งผลลัพธ์การจำแนกโรคใบเรียนกลับไป.
    """
    # ตรวจสอบว่าโมเดลโหลดสำเร็จหรือไม่
    if model is None:
        return jsonify({"error": "AI Model not loaded or failed to load. Check server logs."}), 500

    # ตรวจสอบว่ามีไฟล์รูปภาพส่งมาหรือไม่ใน request
    if 'file' not in request.files:
        return jsonify({"error": "No 'file' part in the request. Please upload an image file."}), 400

    file = request.files['file']
    # ตรวจสอบว่าชื่อไฟล์ไม่ว่างเปล่า
    if file.filename == '':
        return jsonify({"error": "No selected file. Please choose an image to upload."}), 400

    # ประมวลผลไฟล์รูปภาพ
    if file:
        try:
            # 1. อ่านข้อมูลรูปภาพจากไฟล์ที่อัปโหลด (เป็น binary)
            img_bytes = file.read()
            # 2. เปิดรูปภาพด้วย Pillow (PIL) จากข้อมูล binary
            img = Image.open(io.BytesIO(img_bytes))

            # 3. ปรับขนาดรูปภาพให้ตรงกับที่โมเดลคาดหวัง (224x224)
            img = img.resize(IMAGE_SIZE)

            # 4. แปลงรูปภาพ PIL เป็น NumPy array
            img_array = np.array(img)

            # 5. ตรวจสอบและปรับจำนวน Channel (ถ้าเป็น RGBA ให้เป็น RGB เท่านั้น)
            if img_array.shape[-1] == 4: # ตรวจสอบว่ามี 4 channels (RGBA)
                img_array = img_array[..., :3] # เลือกแค่ 3 channels แรก (RGB)

            # 6. Normalize ค่า pixel ให้อยู่ในช่วง 0-1 (ตามที่คุณทำใน ImageDataGenerator)
            img_array = img_array / 255.0

            # 7. เพิ่ม dimension สำหรับ Batch (โมเดลคาดหวัง input เป็น Batch)
            # จาก (224, 224, 3) เป็น (1, 224, 224, 3) สำหรับ 1 รูปภาพ
            img_array = np.expand_dims(img_array, axis=0)

            # 8. ทำนายผลด้วยโมเดล AI
            predictions = model.predict(img_array)

            # 9. แปลงผลลัพธ์จากโมเดล (softmax output)
            predicted_class_index = np.argmax(predictions[0]) # หา index ของคลาสที่มีความน่าจะเป็นสูงสุด
            predicted_class_name = CLASS_NAMES[predicted_class_index] # แปลง index เป็นชื่อคลาส
            confidence = float(np.max(predictions[0])) # ดึงค่าความเชื่อมั่นสูงสุด

            # ส่งผลลัพธ์กลับเป็น JSON
            return jsonify({
                "predicted_class": predicted_class_name,
                "confidence": confidence,
                "all_probabilities": predictions[0].tolist() # ส่งค่า probability ของทุก class กลับไปด้วย
            }), 200

        except Exception as e:
            # ดักจับข้อผิดพลาดและส่งกลับเป็น JSON
            print(f"Error during prediction: {e}")
            return jsonify({"error": f"Failed to process image or predict: {str(e)}"}), 500

# --- Endpoint สำหรับ Health Check หรือ Default Page ---
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def health_check(path):
    """
    Endpoint สำหรับตรวจสอบว่า API ทำงานอยู่หรือไม่.
    """
    return jsonify({
        "message": "Durian Leaf Disease Backend API is running!",
        "version": "1.0",
        "model_loaded": model is not None,
        "endpoint_info": "Use POST /api/predict with 'file' (image) for prediction."
    }), 200

# --- สำหรับรันบน Local (จะไม่ถูกรันบน Vercel) ---
if __name__ == '__main__':
    # รัน Flask app บน http://127.0.0.1:5000/
    # ในโหมด Debugging ซึ่งจะรีโหลดโค้ดอัตโนมัติเมื่อมีการเปลี่ยนแปลง
    app.run(debug=True, port=5000)