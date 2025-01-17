from ultralytics import YOLO
import cv2

class DeepSort():
    def __init__(self):
        self.tracks = []

    def update(self,detections):
        fit_tracks = []
        for box in detections:
            match = False
            for i,track in enumerate(self.tracks):
                box_center = [(box[0] + box[2])/2,(box[1] + box[3])/2]
                track_center = [(track[0] + track[2])/2,(track[1] + track[3])/2]
                distance = ((box_center[0] - track_center[0]) ** 2 + (box_center[1] - track_center[1]) **2) **0.5
                if distance < 50 :
                    match = True
                    self.tracks[i] = box
                    fit_tracks.append(box)
                    break
            if not match:
                self.tracks.append(box)
        return fit_tracks


if __name__ == "__main__":
    model = YOLO('yolov5s.pt')
    cap = cv2.VideoCapture('test5.mp4')
    deep_sort = DeepSort()
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break
        result = model(frame)
        detections = []
        print(type(result[0]))
        for boxes in result[0].boxes:
            x1,y1,x2,y2 = boxes.xyxy[0].tolist()
            conf = boxes.conf.item()
            if conf > 0.5:
                detections.append([x1,y1,x2-x1,y2-y1])

        tracks = deep_sort.update(detections)
        for obj in tracks:
            x1,y1,w,h = obj
            cv2.rectangle(frame,(int(x1),int(y1)),(int(w+x1),int(h+y1)),[0,255,0],2)

        cv2.imshow('Traffic Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
