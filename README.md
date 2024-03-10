# 한글 인식 웹서비스
한글 1글자를 손글씨로 입력받아 어떤 글자인지 맞추는 웹사이트
### 기능
* 그림판(완료)
  - 글자를 그릴 수 있는 그림판
  - 전체 지우기 버튼
  - 판별 실행 버튼
  - 화면 변경시마다 캔버스 사이즈 자동 변경
* 웹서버(완료)
  - 이미지를 서버로 전송
  - 텍스트 판별을 위한 파이썬 코드 실행
  - 파이썬 코드 실행 결과를 클라이언트로 응답
* 추론 API(진행중)
  - Fast API
* 한글 판별기(완료)
  - 손글씨 이미지 데이터로 한글 판별기 제작
  - 전송된 이미지 사이즈를 학습 이미지와 같게 축소
  - 축소된 이미지를 RGB 코드로 변환
  - RGB코드로 변환된 손글씨 이미지를 한글 판별기에 입력하고 판별
### 서비스 URL
http://suitupid.cafe24.com:3000
