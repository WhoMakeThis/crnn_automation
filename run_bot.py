import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms

from crnn.model import CRNN
from crnn.dataset import CHARS

# 서버 주소
BASE_URL = "http://211.188.49.36:5000"

# OCR 모델 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
IDX2CHAR[0] = ""  # CTC blank

# ✅ 이미지 전처리 - CRNN 학습 기준에 맞춤
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 150)),  # ← 수정됨
    transforms.ToTensor()
])

# 디코딩 함수
def decode_prediction(preds):
    preds = preds.permute(1, 0, 2)
    preds = torch.argmax(preds, dim=2)[0].detach().cpu().tolist()
    decoded = []
    prev = -1
    for p in preds:
        if p != prev and p != 0:
            decoded.append(IDX2CHAR.get(p, ""))
        prev = p
    return ''.join(decoded)

# ✅ 모델 로드 - height=50 기준
model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)
model.load_state_dict(torch.load("crnn/best_crnn_model.pth", map_location=DEVICE))
model.eval()

# 1. CAPTCHA 이미지 및 정답 요청
session = requests.Session()
res = session.get(BASE_URL + "/api/captcha/generate")
if res.status_code != 200:
    print("❌ CAPTCHA 생성 실패")
    exit()

data = res.json()
img_url = BASE_URL + data["captcha_url"]
correct_text = data["text"]

print(f"\n🖼 [CAPTCHA 이미지 URL] {img_url}")
print(f"🎯 [서버 제공 정답] {correct_text}")

# 2. 이미지 다운로드 및 예측
img_bytes = session.get(img_url).content
img = Image.open(BytesIO(img_bytes)).convert("L")
img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img)
    pred_text = decode_prediction(pred)

print(f"🤖 [모델 예측 결과] {pred_text}")

# 3. 결과 전송
verify_res = session.post(BASE_URL + "/api/captcha/verify", json={
    "user_input": pred_text,
    "correct_text": correct_text  # 서버 비교용 (실제 서비스에선 보내지 않아야 함)
})

# 4. 응답 처리
print()
if verify_res.status_code == 200:
    result = verify_res.json()
    if result.get("success"):
        print("✅ 인증 성공:", result.get("message"))
    else:
        print("❌ 인증 실패:", result.get("message"))
else:
    print("❌ 서버 응답 실패:", verify_res.status_code)
