<face_detection 2학기 텀프로젝트>

감정Db를 가지고 700장 training 감정준류 진행
얼굴에 바운딩 박스 주고 detection 줘서 학습시키는 것.
3가지 validation 학습.

1. 영상 전체로 감정 분류
2. 영상에다가 얼굴 검출 normalize 얼굴에 대해서만 검출
3. 영상에서 걈정별 얼굴을 검출하는 것.
이 세개를 검출하는 것을 목표로 한다.

케스케이드(검출부터 분룬 순차적) VS 바로 검출하기