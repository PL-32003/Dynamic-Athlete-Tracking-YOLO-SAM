import cv2
import numpy as np
import torch
from ultralytics import YOLO, SAM

# 硬體自動偵測 (GPU 或 CPU)
# 透過 torch 檢查當前環境是否支援 CUDA
if torch.cuda.is_available():
    device = 'cuda'
    print("🚀 成功偵測到可用的 GPU (CUDA)，將啟用顯示卡加速運算！")
else:
    device = 'cpu'
    print("🐢 未偵測到支援 GPU 的環境，將自動退回使用 CPU 運算 (處理速度會較慢)。")

# 載入模型
tracker_model = YOLO('yolo11n.pt').to(device) 
sam_model = SAM('sam2_b.pt').to(device)

video_path = "run.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"錯誤：無法開啟影片 {video_path}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter('detect_locked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)

# 【新增】：建立一個變數來儲存我們想要鎖定的目標 ID
target_track_id = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("影片處理完畢。")
        break

    # 執行 YOLO 追蹤 (降低信心度門檻以應付動態模糊)
    tracking_results = tracker_model.track(frame, persist=True, classes=[0], conf=0.2, verbose=False)
    
    mask_overlay = np.zeros_like(frame)

    if tracking_results[0].boxes.id is not None and len(tracking_results[0].boxes) > 0:
        
        boxes_xyxy = tracking_results[0].boxes.xyxy.cpu().numpy()
        track_ids = tracking_results[0].boxes.id.int().cpu().numpy()
        
        # --- 階段 A：鎖定目標 (根據最高信心度) ---
        # 如果還沒鎖定目標，就在第一幀找出「信心度最高」的人並鎖定他的 ID
        if target_track_id is None:
            # 提取畫面上所有人的信心度分數
            confidences = tracking_results[0].boxes.conf.cpu().numpy()
            
            # 找出信心度最高的那個人的索引值 (Index)
            best_idx = np.argmax(confidences)
            
            # 把這個人的 ID 記錄下來，作為接下來整部影片的唯一追蹤目標
            target_track_id = track_ids[best_idx]
            
            # 印出我們鎖定的 ID 和他當時的信心度，方便你除錯確認
            highest_conf = confidences[best_idx]
            print(f"🎯 已根據「最高信心度 ({highest_conf:.2f})」鎖定目標，追蹤 ID: {target_track_id}")

        # --- 階段 B：持續追蹤鎖定的 ID ---
        # 檢查我們鎖定的 ID 是否存在於當前這幀的畫面中
        if target_track_id in track_ids:
            # 找出目標 ID 在陣列中的索引位置
            # np.where 會回傳一個 tuple，我們取 [0][0] 來得到實際的 index
            target_idx = np.where(track_ids == target_track_id)[0][0]
            
            # 提取專屬該 ID 的 Bounding Box
            box = boxes_xyxy[target_idx]
            x1, y1, x2, y2 = box
            
            # 向外擴充邊界框 (Padding) 10%，確保手腳不被切斷
            pad_x = (x2 - x1) * 0.10
            pad_y = (y2 - y1) * 0.10
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(width, x2 + pad_x)
            y2 = min(height, y2 + pad_y)
            
            best_box = np.array([[x1, y1, x2, y2]]) 
            
            # 將擴充後的框傳給 SAM 進行分割
            sam_results = sam_model(frame, bboxes=best_box, verbose=False)

            # 繪製遮罩
            if sam_results[0].masks is not None:
                mask = sam_results[0].masks.data[0].cpu().numpy() 
                color = colors[target_track_id % 1000].tolist()
                rgb_mask = np.zeros_like(frame)
                rgb_mask[mask > 0.5] = color
                mask_overlay = cv2.addWeighted(mask_overlay, 1.0, rgb_mask, 0.6, 0)
        else:
            # 如果這一幀 YOLO 不小心把跑者跟丟了，可以在終端機提示
            print(f"⚠️ 警告：在影格 {int(cap.get(cv2.CAP_PROP_POS_FRAMES))} 遺失目標 ID {target_track_id}")

    # 將 Mask 疊加到原始幀上並寫入
    annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.7, 0)
    cv2.imshow("ID Locked SAM Tracking", annotated_frame)
    out.write(annotated_frame) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()