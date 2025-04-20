# CRNN CAPTCHA 자동입력 봇

이 프로젝트는 학습된 CRNN 모델을 사용하여 웹에서 생성된 CAPTCHA 이미지를 인식하고 자동으로 입력하는 봇입니다.

## 구조

- `crnn/`: CRNN 모델 및 문자셋 정의
- `run_bot.py`: 자동화 메인 코드(얜 웹 직접접 안띄우고 서버에서만)
- `selenium_bot.py`: selenium으로 웹 직접 띄워서 해보려고 했는데 답 제출버튼이 안눌림 (일단 모델이 답을 못맞춰서 보류류)
- `requirements.txt`: 의존성 리스트
- `best_crnn_model.pth` : crnn 모델 학습 중에 로스 떨어지면 자동으로 전 에포크 모델로 저장하게 했음
- 그 외 error_screenshot.png, reponse_debug.html 이건 봇 만들면서 디버깅용으로 생성되는것들들

## 실행 방법

```bash
pip install -r requirements.txt
python run_bot.py