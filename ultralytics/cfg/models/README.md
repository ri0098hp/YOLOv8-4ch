# Models

## Metrics

| **Models**                | **Speed/Image** | **FPS** | **AP@All** | **AP@Hot** | **AP@Inter** | **AP@Cold** | **diff** |
| ------------------------- | --------------- | ------- | ---------- | ---------- | ------------ | ----------- | -------- |
| YOLOv8s                   | 2.8 ms          | 356     | 83.8       | 79.6       | 89.0         | 83.0        | 9.4      |
| YOLOv8s (COCO pretrained) | 2.8 ms          | 355     | 84.1       | 79.4       | 90.5         | 82.7        | 11.1     |
| YOLOv8m                   | 6.2 ms          | 162     | 84.7       | 81.4       | 90.0         | 83.4        | 8.6      |
| YOLOv8s-st1               | 3.0 ms          | 331     | 84.1       | 79.7       | 89.9         | 83.0        | 10.2     |
| YOLOv8s-st2               | 3.2 ms          | 308     | 85.1       | 81.4       | 90.1         | 84.0        | 8.7      |
| YOLOv8s-st2-add           | 3.2 ms          | 312     | 85.6       | 82.3       | 90.3         | 84.4        | 8.0      |
| YOLOv8s-st2-convk1        | 3.2 ms          | 307     | 85.4       | 81.9       | 90.3         | 84.3        | 8.4      |
| YOLOv8s-st2-convk3        | 3.4 ms          | 295     | 85.8       | 82.6       | 90.6         | 84.4        | 8.0      |
| YOLOv8s-st3               | 4.2 ms          | 240     | 85.6       | 83.2       | 90.3         | 83.8        | 7.1      |
| YOLOv8s-2stream           | 4.1 ms          | 241     | **86.5**   | 84.3       | 90.6         | 85.1        | **6.3**  |

- Dataset: All-Season Dataset (Ours)
- Device: NVIDIA RTX 3090
- Format: Troch (pt)
- Batch-Size: 1

## Architecture Diagram

### YOLOv8s 4ch

![yolov8s.drawio.svg](diagram/yolov8s.drawio.svg)

### YOLOv8s-st2

![yolov8s-st.drawio.svg](diagram/yolov8s-st.drawio.svg)

### YOLOv8s-st2-conv

![yolov8s-st-conv.drawio.svg](diagram/yolov8s-st-conv.drawio.svg)

### YOLOv8s-2stream

![yolov8s-2stream.drawio.svg](diagram/yolov8s-2stream.drawio.svg)
