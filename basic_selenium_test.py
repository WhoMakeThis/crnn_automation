from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

# 브라우저 실행 파일 경로 (너 PC 기준으로 수정해줘)
chrome_binary_path = "C:/Users/user/Downloads/chrome-win64/chrome-win64/chrome.exe"

# 크롬 드라이버 경로 (너가 받은 버전에 맞게 수정)
chrome_driver_path = "C:/Users/user/Downloads/chromedriver-win64/chromedriver-win64/chromedriver.exe"

# 옵션 설정
options = Options()
options.binary_location = chrome_binary_path

# 드라이버 서비스
service = Service(executable_path=chrome_driver_path)

# 브라우저 실행
browser = webdriver.Chrome(service=service, options=options)

# 테스트용 주소
start_url = "http://211.188.49.36:5000/"
browser.get(start_url)
browser.maximize_window()

# 요소 찾고 클릭 (XPath 위치는 네가 보고 싶은 대상 요소로 수정)
element = browser.find_elements(By.XPATH, '/html/body/div/div[1]/button[1]')[0]
element.click()

time.sleep(3)
browser.quit()
