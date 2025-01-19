import cv2
from ultralytics import YOLO

# تحميل الموديل
model = YOLO("yolo11n.pt")

# رابط الفيديو من الراسبيري باي
video_url = "http://192.168.173.58:5000/video_feed"

# فتح الفيديو
cap = cv2.VideoCapture(video_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("لا يمكن قراءة الفيديو")
        break

    # تطبيق YOLOv11 على كل إطار
    results = model(frame)

    # رسم النتائج على الإطار
    annotated_frame = results[0].plot()

    # عرض الفيديو
    cv2.imshow("yolo11n", annotated_frame)

    # الخروج بالضغط على 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تحرير الموارد
cap.release()
cv2.destroyAllWindows()
