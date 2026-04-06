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

# 讓使用者決定要追蹤幾個最高信心度的目標
while True:
    try:
        num_targets = int(input("請輸入要鎖定的最高信心度目標數量 (例如輸入 1, 2 或 3): "))
        if num_targets > 0:
            break
        else:
            print("請輸入大於 0 的整數喔！")
    except ValueError:
        print("輸入格式錯誤，請輸入數字。")

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

out = cv2.VideoWriter('detect_top_n_locked.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
colors = np.random.randint(0, 255, (1000, 3), dtype=np.uint8)

# 【修改】：將單一變數改成一個列表，用來儲存多個目標的 ID
target_track_ids = [] 

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("影片處理完畢。")
        break

    # 執行 YOLO 追蹤
    tracking_results = tracker_model.track(frame, persist=True, classes=[0], conf=0.2, verbose=False)
    mask_overlay = np.zeros_like(frame)

    if tracking_results[0].boxes.id is not None and len(tracking_results[0].boxes) > 0:
        boxes_xyxy = tracking_results[0].boxes.xyxy.cpu().numpy()
        track_ids = tracking_results[0].boxes.id.int().cpu().numpy()
        confidences = tracking_results[0].boxes.conf.cpu().numpy()
        
        # --- 階段 A：根據最高信心度鎖定 Top-N 個目標 ---
        if len(target_track_ids) == 0:
            # 確保使用者要求的數量，不會超過畫面中實際偵測到的人數
            actual_num = min(num_targets, len(confidences))
            
            # 使用 np.argsort 排序。加個負號 (-confidences) 代表「由大到小」排序
            # [:actual_num] 則是取出排名前 N 名的索引值
            top_indices = np.argsort(-confidences)[:actual_num]
            
            # 將這幾個最高分的 ID 存入我們的追蹤清單中
            target_track_ids = track_ids[top_indices].tolist()
            
            print(f"🎯 已鎖定信心度前 {actual_num} 高的目標！追蹤 ID 為: {target_track_ids}")

        # --- 階段 B：持續追蹤這群被鎖定的 ID ---
        # 準備兩個空列表，用來收集「這幀畫面中依然存在的目標」的邊界框與 ID
        valid_boxes = []
        valid_ids = []
        
        # 檢查畫面上偵測到的每一個人，看看他的 ID 有沒有在我們的鎖定清單中
        for i, tid in enumerate(track_ids):
            if tid in target_track_ids:
                box = boxes_xyxy[i]
                x1, y1, x2, y2 = box
                
                # 加入 10% Padding 避免手腳被切掉
                pad_x = (x2 - x1) * 0.10
                pad_y = (y2 - y1) * 0.10
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(width, x2 + pad_x)
                y2 = min(height, y2 + pad_y)
                
                valid_boxes.append([x1, y1, x2, y2])
                valid_ids.append(tid)
        
        # 如果畫面中還有我們鎖定的目標 (可能有人跑出鏡頭外了)
        if len(valid_boxes) > 0:
            # 將清單轉換為 numpy 陣列格式 [N, 4]
            input_boxes = np.array(valid_boxes) 
            
            # 【效率優化】：一次把所有目標的邊界框丟給 SAM 處理
            sam_results = sam_model(frame, bboxes=input_boxes, verbose=False)

            if sam_results[0].masks is not None:
                # SAM 會回傳多個遮罩 (對應我們傳進去的多個框)
                masks = sam_results[0].masks.data.cpu().numpy() 
                
                # 逐一為每個目標上色
                for j, mask in enumerate(masks):
                    tid = valid_ids[j] # 取得對應的 ID 來決定顏色
                    color = colors[tid % 1000].tolist()
                    rgb_mask = np.zeros_like(frame)
                    rgb_mask[mask > 0.5] = color
                    mask_overlay = cv2.addWeighted(mask_overlay, 1.0, rgb_mask, 0.6, 0)

    # 顯示與寫入影片
    annotated_frame = cv2.addWeighted(frame, 1.0, mask_overlay, 0.7, 0)
    cv2.imshow("Top-N ID Locked SAM Tracking", annotated_frame)
    out.write(annotated_frame) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()