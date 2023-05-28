---
layout: post
title:  "[2017]Focal Loss for Dense Object Detection"
date:   2023-05-22 15:14:33 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
* CE변환한 Focal Loss로 imbalanced 데이터 분포에서 성능 향상  
** 기존에는 object detection에서 R-CNN과 같은 2 stage 접근  
** 1 stage가 빠르고 간단하지만 성능이 떨어짐  
** FL로 잘 분류하는 것들의 weight를 줄여서 성능 향상  
** SOTA 달성  




{% endhighlight %}  

<br/>


[Paper with my notes](https://drive.google.com/drive/folders/15hVZLozTfkgVA8qvDbbRkPaQdG68T5Lo?usp=sharing)  


[Lecture link](https://www.youtube.com/watch?v=44tlnmmt3h0)  

<br/>

# 단어정리  
* push the envelop: 한계를 넘어서다    
* lesion: 손상, 비정상  
* lesion study: 모델의 특정 부분이 비활성화될 때 성능에 어떤 영향을 미치는지 연구하는 것?  
* off-the-shelf model: 사전 학습된 모델?  









<br/>

# 1 Introduction  
* 기존 SOTA는 2 stage  
** 1. gen sparse set  
** 2. classify with CNN  
** 1 stage도 괜찮은 결과 보이는 경우 있음  
** 1 stage 사용 제안  
** 클래스 불균형 있을 때 two stage들은 candidate 수 줄임  
** 반면 1 stage는 candidate 늘림  
** imbalance 해소용 loss 제안  
** CE 스케일링한 것 제안  
** Ratinanet 구현 및 실험으로 COCO dataset에서 성능 증명  
<br/>

# 2 Related Work  
### Classic Object Detectiors:  
* 전통적 object deotector  
** CNN 슬라이드 윈도우 방식  
*** HOG, DPM 등이 있음  
*** 2 stage detector로 귀결됨  


### Two-stage Detectors:  
* 2 STAGE detectors  
** 1. sparse set 생성  
** 2. classify classes  


### One-stage Detectors:  
* 1 stage detector  
** OverFeat, SSD, YOLO가 예시  
** speed 높고 acc 낮음  
** Ratina net은 이전 dense detector와 비슷  
*** 앵커  
*** feature pyramid  


### Class Imbalance:  
* Imbalanced class 경우들 있음(본 논문이 커버하는 케이스)  
** 1) 학습에 방해  
** 2) 일반화 방해  
** sol: sample  
*** focal loss가 해결  


### Robust Estimation:  
* 견고한 추정  
** 아웃라이어 제거  
** focal loss가 근데 더 나은 방법임  
<br/>

# 3. Focal Loss  
* 이진 분류 CE  
** CE(p,y) = -log(p) if y=1, -log(1-p) otherwise.  
** pt = p if y=1, 1-p otherwise.  
** 작은 loss values가 rare class에서 압도  

## 3.1. Balanced Cross Entropy  
* 가중치 준 CE  
** 가중치 alpha t는 pt 확률로 추론  


## 3.2. Focal Loss Definition  
* Focal Loss 정의  
** 기존 가중 ce 문제:alpha 가중치는 easy/hard example 구분 못 함    
** (1-pt)^gamma 추가( modulating factor )  
** 1. 오분류 pt 작으면 modulating factor = 1 & loss 그대로  
** 하지만 잘 분류시 loss 줄어듦  
** 2. gamma는 부드럽게 조절, gamma=0이면 FL=CE, gamma 오르면 modulating factor증가(best:2)  
** 즉, 쉬운거 CE 줄이고 어려운거 CE 늘림  


### 3.3. Class Imbalance and Model Initialization  
* 클래스 불균형 & 모델 초기화  
** BC(Binary Classification) 모델로 y=-1 or 1로 초기화  
** 그래서 사전 value p 사용  
** 이게 성능 올림  


### 3.4. Class Imbalnce and Two-stage Detectors  
* 불균형 클래스 & 2단계 탐지기  
** (1) 2단계 cascade  
*** object 위치를 1 or 2로 줄임  
** (2) biased minibatch sampling   
*** 비율 조정  
** 본 제안 focal loss를 one-stage로 해결  
<br/>

# 4. RetinaNet Detector  
* 레티나넷 인식기  
** 싱글, 통일된 네트워크, 백본 네트워크로 구성, two task specific subnetwrok로 구성  
** 백본 CNN featuremap  
** 첫 subnet 백본 분류  
** 두번째 subnet bound box regression  


### Feature Pyramid Network Backbone:  
* 피처파라미드넷 백본(FPN)  
** CNN 증강, 피라미드 모양  
** top: ResNet  


*** Anchors:  
* 앵커  
** 앵커박스 사용, 사이즈, ratio 다양 {1:2, 1:1, 2:1}, {2^0, 2^1/3, 2^2/3}  
*** A=9개 앵커, per level  
*** range 32-813 픽셀  
** 각 앵커 k 할당 원핫백터(분류 target들)  
** RPN서 rule 할당 IoU 쓰레쉬홀드 (- [0, 0.4)  


### Classification Subnet:  
* 분류서브넷  
** obj 등장 확률 계산, A는 앵커개수, K는 obj 개수  
** FCN 추가된 FPN 형태  
** box regression과 share 안 함  


### Box Regression Subnet:  
** 작은 FCN 추가  
** common 구조는 공유  
** 다른 param 사용  


## 4.1. Inference and Training  
### Inference:  
** 속도 높이기 위해 box predict만 decode  


### Focal Loss:  
** FL 사용 gamma=2가 best  


### Initialization:  
* 초기화  
** 백본: ResNet50FPN, 101FPN  
** ImagNet 1K로 사전 학습  
** bias b=0, weight alpha=0.01, initialization to b=-log((1-pi)/pi), where pi=0.1  


### Optimization:  
* 최적화  
** SGD 사용 , 8GPU, 16image per mini batch  
** 90K iter train, l.r=0.01  
** weight decay 0.0001, momemtum0.9, loss=sum of FL, smooth L1 loss to box reg  
*** train time: 10~35hours  
<br/>

# 5. Experiments  
**bounding box detect in COCO로 실험  
** 일반론 따르고 COCO trainval35K split    
** random 35K subset of image  


## 5.1. Training Dense Detection  
* dense detection 학습  
** 50,101 ResNet  
** 600 픽셀 이미지  


### Network Initialization:  
* 네트워크 초기화  
** CE만 사용할 경우  빠르지만 발산  
** pi=0이 detect 초기화서 효과적  


### Balanced Cross Entropy:  
* 균형 CE test, alpha0.75줌  


### Focal Loss:  
* FL, gamma=2, yield 2.9AP    
** alpha는 [0.25, 0.75]  


### Analysis of the Focal Loss:  
* 분석   
** Cumulative Distribution Function(CFD)이용  
** gamma 증가시 weight 증가  
** concentrate on hard negative exaple  
** FL 이 easy negative effect discount  


### Online Hard Example Mining (OHEM):   
* two stage 제안 방법  
** SSD로 구현  


### Hinge Loss:  
* 효과 없음  


## 5.2. Model Architecture Design  
### Anchor Density:  
* 아키텍처  
** 앵커 one일때 two일 때 달리함(stage)  
** sweep으로 최적 찾음  
** 6~9가 최적  


### Speed versus Accuracy:  
* 속도 vs 정확도  
** 백본 크면 acc 증가하지만 속도 감소   


## 5.3. Comparison to State of the Art  
* SOTA와 비교  
** SOTA 달성  
<br/>

# 6. Conclusion  
* class 불균형 인식  
** one stage 걸림돌  
** two stage 만연한 상태: Faster R-CNN  
* 해결 위해 FL 도입(CE에 weight 조절 learning)  
** 간단, 효과적 SOTA 달성  