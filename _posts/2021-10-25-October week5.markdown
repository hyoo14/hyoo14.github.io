---
layout: post
title:  "October week5"
date:   2021-10-25 16:30:10 +0900
categories: study
---




{% highlight ruby %}

10월 마지막주:
-KIFRS 기준서 다운로드(OR KGAAP)
-BERT pretrain(https://beomi.github.io/2020/02/26/Train-BERT-from-scratch-on-colab-TPU-Tensorflow-ver/)
-->fine tuning?(화요일 추가)
-ATTENTION 논문 복습? ->BERT논문 복습

{% endhighlight %}

2021년 10월 25일 월요일

버트모델 테스트
-GCP 버킷만들기
-AttributeError: module 'tensorflow._api.v2.train' has no attribute 'Optimizer' 오류발생
   :텐서플로우버전이 2로 되어서 그런듯 --이슈과 확실히 있음.. 다른 튜토리얼 추가로 참고해야하나..?
   :코랩에서 %tensorflow_version 1.15 로 버전 선택 가능 ->이슈 해결 (https://colab.research.google.com/notebooks/tensorflow_version.ipynb#scrollTo=-XbfkU7BeziQ)
-Not connected to TPU runtime
   :런타임 tpu로 안 하니 당연히 뜸 -> 런타임 tpu로 변경. 잘됨
-AuthorizationError: Error fetching credentials
   :런타임 리셋? -> 잘됨
-코랩에서 파일 읽기
   :구글 드라이브
   from google.colab import drive
   drive.mount('/gdrive')
   (https://colab.research.google.com/github/StillWork/ds-lab/blob/master/tip_colab_%ED%8C%8C%EC%9D%BC%EC%A0%80%EC%9E%A5%EB%B0%8F%EC%97%85%EB%A1%9C%EB%93%9C_colab.ipynb#scrollTo=zpL1IQforS1I)
-파이썬 파일 출력
   :참.. 기억력이 안 좋다 나는
    f = open("새파일.txt", 'w')
    f.close()
   (https://wikidocs.net/26)
-매우 적은 데이터로 프리-트레이닝 테스트 진행

-코렙 끊김 방지
   :개발자모드 콘솔에서 아래 실행
   function ConnectButton(){
    console.log("Connect pushed"); 
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click() 
   }
   setInterval(ConnectButton,60000);

BERT논문 오랜만에 읽는 중.

2021년 10월 26일 화요일
버트모델 테스트 이어서..
-체크포인트 읽어와서 다시 트레인
   :걍 처음부터 다 돌리면 알아서 리로드해서 트레이닝 시작함. (단 전처리도 다시하는.. ->전처리도 저장해놓으면 편할텐데)
   
논문 계속 읽는중

코드 문제 한문제 품

2021년 10월 27일 수요일
버트모델 테스트
-1,000,000번 돌려서 한사이클 테스트 완료 예정->완료
-파인 튜닝 해봐야지..(내일해야겠ㄷ..)

논문 다 읽음. 정리 해봐야지 -> 정리 함.

문제 풀이 시도하였으나 30퍼..ㅠ



