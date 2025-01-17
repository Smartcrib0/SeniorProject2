import cv2
import threading
import numpy as np
import sounddevice as sd
import wavio
import librosa
from tensorflow.keras.models import load_model
from ultralytics import YOLO
import os
import pygame

pygame.mixer.init()

# تحميل النموذج المدرب الأول لتصنيف الصوت (تحديد إذا كان هناك بكاء)
model_path_audio = 'Senior_Project/audio_classifier_mfcc_improved.h5'
model_audio = load_model(model_path_audio)
CATEGORIES = ['Crying', 'Laugh', 'Noise', 'Silence']

# تحميل النموذج المدرب الثاني لتحديد سبب البكاء
model_path_cry_reason = 'Senior_Project/sound_detection_modelCNN.h5'
model_cry_reason = load_model(model_path_cry_reason)
classes = ['belly_pain', 'burping', 'cold-hot', 'discomfort', "dontKnow", 'hungry', 'lonely', 'scared', 'tired']

# مسار نموذج YOLO
model_yolo = YOLO("yolo11n.pt")

def play_sound(file_path):
    try:
        pygame.mixer.music.load(file_path)  # تحميل الملف الصوتي
        pygame.mixer.music.play(loops=-1, fade_ms=0)  # تشغيل الصوت بشكل متكرر طوال فترة البكاء
        print(f"تشغيل الصوت: {file_path}")
    except Exception as e:
        print(f"حدث خطأ: {e}")

def stop_sound():
    pygame.mixer.music.stop()  # إيقاف الموسيقى

# دالة لتسجيل الصوت بشكل غير متزامن
def record_audio_async(duration=6, filename="detected_audio.wav", callback=None):
    def _record():
        sample_rate = 44100
        channels = 2
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
        sd.wait()
        wavio.write(filename, audio, sample_rate, sampwidth=2)
        if callback:
            callback(filename)
    threading.Thread(target=_record, daemon=True).start()

# دالة لمعالجة الصوت واستخراج MFCC
def process_audio_mfcc(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc = np.mean(mfcc.T, axis=0)
    return np.expand_dims(mfcc, axis=-1)

# دالة لتنبؤ إذا كان هناك بكاء باستخدام النموذج المدرب الأول
def predict_cry(file_path):
    mfcc = process_audio_mfcc(file_path)
    mfcc = np.expand_dims(mfcc, axis=0)
    prediction = model_audio.predict(mfcc)
    return CATEGORIES[np.argmax(prediction)]

# دالة لمعالجة الصوت واستخراج mel-spectrogram
def process_audio_mel(file_path, duration_seconds=6, target_sr=22050):
    y, sr = librosa.load(file_path, sr=target_sr)
    y = librosa.util.fix_length(y, size=target_sr * duration_seconds)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db

# دالة لتنبؤ سبب بكاء الطفل باستخدام النموذج المدرب الثاني
def predict_cry_reason(file_path):
    mel_spec = process_audio_mel(file_path)
    mel_spec = np.expand_dims(mel_spec, axis=-1)
    mel_spec = np.expand_dims(mel_spec, axis=0)
    mel_spec = mel_spec / np.max(mel_spec)
    prediction = model_cry_reason.predict(mel_spec)
    return classes[np.argmax(prediction)]

# فئة لإدارة معالجة الصوت في الخلفية
class AudioProcessor():
    cry_prediction = ""
    cry_reason = ""
    is_processing = False

    def process_audio_file(self, filename):
        try:
            y, sr = librosa.load(filename, sr=None)
            energy = np.sum(y**2) / len(y)  # الطاقة
            rms = librosa.feature.rms(y=y)[0].mean()  # الجهارة
            zcr = librosa.feature.zero_crossing_rate(y)[0].mean()  # معدل عبور الصفر

            # التحقق من السكون بناءً على الطاقة والجهارة
            if energy < 1e-5 and rms < 0.01:
                self.cry_prediction = "Silence"
                self.cry_reason = "No Reason"
            elif zcr < 0.02 and rms < 0.015:
                self.cry_prediction = "Silence"
                self.cry_reason = "No Reason"
            else:
                cry_prediction = predict_cry(filename)
                self.cry_prediction = cry_prediction
                if cry_prediction == 'Crying':
                    cry_reason = predict_cry_reason(filename)
                    self.cry_reason = cry_reason
                else:
                    self.cry_prediction = "Other Sound"
                    self.cry_reason = ""
        except Exception as e:
            print(f"Error in audio processing: {e}")
        finally:
            if os.path.exists(filename):
                os.remove(filename)
            self.is_processing = False

    def start_audio_check(self):
        if not self.is_processing:
            self.is_processing = True
            record_audio_async(duration=6, callback=self.process_audio_file)

# فتح الكاميرا
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

audio_processor = AudioProcessor()

# بدء عملية الفيديو والكشف عن الأجسام
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read a frame from the camera.")
        break

    # أداء الكشف عن الأجسام باستخدام YOLO
    results = model_yolo(frame)
    detected = False
    for box in results[0].boxes:
        cls = box.cls
        conf = box.conf
        if int(cls[0]) == 67 and conf[0] > 0.70:  # إذا تم اكتشاف الهاتف المحمول
            detected = True
            break

    # إذا تم الكشف عن هاتف محمول
    if detected:
        audio_processor.start_audio_check()
        frame_prediction_cry = audio_processor.cry_prediction
        frame_prediction_reason = audio_processor.cry_reason
        if frame_prediction_cry == "Silence":
            cv2.putText(frame, "Cry Detection: Silence", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            stop_sound()  # إيقاف الصوت إذا لم يكن هناك بكاء
        elif frame_prediction_cry == "Crying":
            cv2.putText(frame, "Cry Detection: Crying", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Cry Reason: {frame_prediction_reason}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            if not pygame.mixer.music.get_busy():  # إذا لم تكن الموسيقى قيد التشغيل بالفعل
                play_sound("music1.mp3")  # تشغيل الموسيقى عند بكاء الطفل
        else:
            cv2.putText(frame, "Cry Detection: Other Sound", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            stop_sound()  # إيقاف الصوت إذا كانت الأصوات الأخرى

    else:
        cv2.putText(frame, "Status: The child is not present", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        stop_sound()  # إيقاف الصوت إذا لم يتم اكتشاف الطفل

    # عرض إطار الفيديو
    cv2.imshow("YOLO Object Detection", frame)

    # الخروج من الفيديو عند الضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير موارد الكاميرا وإغلاق جميع نوافذ OpenCV
cap.release()
cv2.destroyAllWindows()
