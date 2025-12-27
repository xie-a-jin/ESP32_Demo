import cv2
import mediapipe as mp
import torch
import serial
import time

# ==========================================
# 1. 串口初始化 (与 ESP32 握手)
# ==========================================
try:
    # 这里的 COM9 是你刚才烧录成功的端口
    # 必须确保 VS Code 的串口监视器已关闭，否则会报错 Access Denied
    ser = serial.Serial('COM9', 115200, timeout=1)
    print("✅ 成功连接到 ESP32 (COM9)")
    time.sleep(2)  # 等待硬件重启稳定
except Exception as e:
    print(f"❌ 无法连接串口: {e}")
    ser = None


# ==========================================
# 2. 定义并加载你的 PyTorch 模型
# ==========================================
class TinyHandModel(torch.nn.Module):
    def __init__(self):
        super(TinyHandModel, self).__init__()
        self.fc1 = torch.nn.Linear(63, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 3)  # 0:石头, 1:剪刀, 2:布
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


device = torch.device("cpu")
model = TinyHandModel().to(device)

try:
    # 载入你之前训练好的权重文件
    model.load_state_dict(torch.load("hand_model.pth", map_location=device))
    model.eval()
    print("✅ AI 模型加载成功！")
except Exception as e:
    print(f"❌ 模型文件加载失败，请检查 hand_model.pth 是否在当前文件夹下: {e}")
    exit()

# ==========================================
# 3. 初始化 MediaPipe 手部检测
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# 4. 开启摄像头并开始实时推理
# ==========================================
cap = cv2.VideoCapture(0)
print("🚀 系统已启动！请对着摄像头做手势...")
print("提示：握拳(石头) -> 亮灯 | 张开手(布) -> 灭灯 | 按 'q' 键退出")

last_cmd = ""  # 用于防止重复发送相同指令，减轻串口压力

while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    # 镜像处理并转换颜色
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 手部关键点识别
    results = hands.process(img_rgb)

    current_action = "Waiting for hands..."

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 在画面上画出骨架
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 提取 21 个关键点的 (x, y, z) 坐标作为模型输入
            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])

            # 模型推理
            input_tensor = torch.FloatTensor(coords).view(1, -1).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prediction = torch.argmax(output, dim=1).item()

            # --- 核心逻辑：手势控制硬件 ---
            if prediction == 0:  # 石头
                current_action = "ROCK -> LED ON"
                if ser and last_cmd != "0":
                    ser.write(b'0')  # 向串口发送字节 0
                    last_cmd = "0"
            elif prediction == 2:  # 布
                current_action = "PAPER -> LED OFF"
                if ser and last_cmd != "2":
                    ser.write(b'2')  # 向串口发送字节 2
                    last_cmd = "2"
            else:
                current_action = "SCISSORS -> No Action"

    # 将识别结果实时显示在图像窗口上
    cv2.putText(img, current_action, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("TinyML Hand Control", img)

    # 按 Q 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
if ser:
    ser.close()
cv2.destroyAllWindows()