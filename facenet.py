import cv2

# 가중치 파일 경로
cascade_filename = 'haarcascade_frontalface_alt.xml'

# 모델 불러오기
cascade = cv2.CascadeClassifier(cascade_filename)

# 영상 파일 
cam = cv2.VideoCapture('sample.mp4') #sample.mp4. 불러오기

# 영상 검출기
def videoDetector(cam,cascade):
  
    while True:
        
        # 캡처 이미지 불러오기
        ret,img = cam.read()
        
        # 그레이 스케일 변환
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # cascade 얼굴 탐지 알고리즘 
        results = cascade.detectMultiScale(gray,            # 입력 이미지
                                           scaleFactor= 1.1,# 이미지 피라미드 스케일 factor
                                           minNeighbors=3,  # 인접 객체 최소 거리 픽셀
                                           minSize=(20,20)  # 탐지 객체 최소 크기
                                           )                                                                          
        for box in results:
            x, y, w, h = box
            cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2) #녹색 검출박스 생성
           
         # 영상 출력        
        cv2.imshow('multimedia',img)
        
        if cv2.waitKey(1) > 0: #아무키나 누르면 종료
  
            break
# 영상 탐지기 실행
videoDetector(cam,cascade)