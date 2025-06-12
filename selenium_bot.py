import time
import requests
from io import BytesIO
from PIL import Image
import torch
import torchvision.transforms as transforms
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

from crnn.model import CRNN
from crnn.dataset import CHARS

# 디바이스 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 문자 인덱스 매핑
IDX2CHAR = {idx + 1: char for idx, char in enumerate(CHARS)}
IDX2CHAR[0] = ""

# ✅ 이미지 전처리: CRNN 학습 기준에 맞춤
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((50, 150)),
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

# ✅ 모델 로드 (height=50 기준)
model = CRNN(50, 1, len(CHARS) + 1, 256).to(DEVICE)
model.load_state_dict(torch.load("crnn/best_crnn_model.pth", map_location=DEVICE))
model.eval()

# 웹드라이버 실행
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("http://211.188.49.36:5000")
wait = WebDriverWait(driver, 20)

# CAPTCHA 이미지 URL 추출
captcha_img = wait.until(EC.presence_of_element_located((By.ID, "captcha-img")))
img_url = ""
for _ in range(10):
    img_url = captcha_img.get_attribute("src")
    if img_url and img_url.strip():
        break
    time.sleep(0.5)

if not img_url or not img_url.strip():
    print("❌ CAPTCHA 이미지 URL 못 가져옴")
    driver.quit()
    exit()

if img_url.startswith("/"):
    img_url = "http://211.188.49.36:5000" + img_url

print(f"[CAPTCHA 이미지 URL] {img_url}")

# 이미지 다운로드 및 예측
img_bytes = requests.get(img_url).content
img = Image.open(BytesIO(img_bytes)).convert("L")
img = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    pred = model(img)
    pred_text = decode_prediction(pred)

print(f"[예측 결과] {pred_text}")

# CAPTCHA 자동 제출
try:
    # 입력창 채우기
    input_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']")))
    input_box.clear()
    input_box.send_keys(pred_text)

    # 제출 버튼 클릭
    submit_btn = wait.until(EC.element_to_be_clickable((By.ID, "submit-btn")))
    driver.execute_script("arguments[0].scrollIntoView(true);", submit_btn)
    submit_btn.click()

    print("✅ 인증 버튼 클릭 완료")
    time.sleep(3)

except Exception as e:
    driver.save_screenshot("error_screenshot.png")
    print(f"❌ 에러: {e}")
    print("🖼 스크린샷 저장: error_screenshot.png")

finally:
    driver.quit()
