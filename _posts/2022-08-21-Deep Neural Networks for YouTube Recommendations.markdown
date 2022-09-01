---
layout: post
title:  "Deep Neural Networks for YouTube Recommendations"
date:   2022-08-21 15:51:19 +0900
categories: study
---


# additional something?  
*탐색(candidate generation)과 랭킹 따로 떼어놓은 이유는 랭킹에서는 고려할 다른 요소들이 많이 합쳐져있기 때문(최근 업로드 등..)  
*surrogate problem? offline metric에 오버피팅되어 실제로는 성능이 안 좋은 상태.. 그래서 A/B test 통해 실질 결과로 모델 선택함    
*clickbait? 클릭 미끼.. ctr의 문제점 언급할 때 쓰이는 듯?  
*candidate generation-영상을 시청할 확률학습, u-user embedding, v-video emgedding 있을 때 특정 video볼 확률, softmax  
*네거티브 샘플 - 전부 학습 안 하고 더 효과적으로  


*throughput-시간당 처리량(tps-transcation per second, rps-request per second, 시간당 처리 http수)    
*latency-요청부터 응답까지 걸리는 시간(qps-quries per second, 초당 처리할 수 있는 쿼리수)  
*candidate generation - recall 최적화  
**depth 늘리면 map 올라감  
*ranking - mse 줄이는 최적화  
**depth, width 늘리면 auc(area upder rocCurve) 좋아지는 쪽으로 감(클수록 1에 가까울수록 좋음)   
*softmax로 볼 확률 예측하게하므로 nearest neigbor 찾는 계산은 안 해도 되서 효율적   


*candidate generation서 미래에 뭐볼지 예측이 계속 볼지 예측보다 효과적(a/b test에서)  
*ranking서 기대시청시간 예측  

*cumulative distribution function->이걸로 정규화??  
*임베딩 어떻게 최신화할까? 모델 최신화??  
*네거티브 샘플링 자세히ㅡ샘플링해서 네거티브 파지티브 갯수 맞추는?  
*watchtime click이어서 가중 로지스틱 회귀 씀  
*피처엔지니어링-시청타임라인,좋아요,검색 등, 기기,문맥등  
*베스트피처-유저인터렉선 윗 씨밀러 비디오, 채널과 주제에대한 시청횟수  
*candi gen-nonlinear generalization of factorization technique->classification, 시청확률 학습하는 소프트맥스인셈  
*u,v, geo,gender,age 컨켓 후 소프트맥스 학습  
*이형 신호-연속형과 카테고리형 피처 모두 사용가능  
*로스는 크로스엔트로피 로스-미니마이즈  
*크로스엔트로피란?  
*서빙에서는 어프록싱 니어리스트네이버-해쉬함수 써서 좀 더 빠르게  
*rmse, auc 좀 더 자세히?  
*유튭 이외 데이터도 포함, 사용자학습데이터수 고정ㅡ사용자 가중치 고정 위해  
*무작위로 아이템 선정 예측보다 특정시점기준 예측이 더 효과적(시간적 정보를 담고 미래 예측이 ab테스트서 더 효과적, 레이블기준 시간적으로 전에 것들 갖고 예측하는 것이 효과적이었다)  
*피처 다 쓰고 뎁스 늘리면 성능 향상  
*MAP 계산?  
*랭킹에서는 영상과 사용자 사이 피처를 많이 사용해야함, 사용자의 이전 행동을 고려하여 아이템과 인터렉션 모델링해야함  
*랭킹피처공정-카테고리는 임베딩화, 컨티뉴어스는 노멀라이즈  
*사용자가 선택한 아이템과 그렇지 않은 아이템의 기대시청시간 예측이 목표  
*정리)  
**candidate generation-멀티클래스 클래시피케이션, 소프트맥스  
**랭킹-익스펙티드 워치타임  
**캔디잰, 랭킹 네트웍구조 비슷  
**age 피처 통해 시간 프레쉬니스 반영  
**딥cf로 더 많은 피처 반영  
**피처의 임베딩, 노멀라이즈 잘 활용  
**여러층 신경망 넌리니어하게 효과적으로 피처 모델링  

*candidate generation-cf사용, in:user history/out:expected related videos, factorization 기술에서 온 nonlinear generation임  
*cf? mf?  
*feature power(x, root x, x square ...)은 크게 중요하지는 않음(강의에 따르면)  
*feature engineering에서 user-video interaction이 중요했음  


*futuer   
**그래서 19년에는 멀티 클래스 랭킹 시스템  
**다음에 뭐 볼지 예측  
**ㅡ멀티태스크 러닝, 엔드투엔드 랭킹 시스템  
**멀티태스크 러닝이란? user engagement(click and watch time), user satisfaction(like and rating) 이렇게 테스크 두개 나눠서 학습시킴  
**와이드앤딥 모델 쓰고 position bias줄임(mitigate)  



{% highlight ruby %}
짧은 요약 :  

유튜브 추천 시스템에 DNN을 도입하여 성능 향상  

후보자 생성, 랭킹 2가지 모델로 구성됨  

둘 다 dnn 씀  




{% endhighlight %}


링크(https://drive.google.com/drive/folders/1c5lyOb0Vc-RMfISK4a0sogeBJW0PglCo?usp=sharing)


# 단어정리  
*dichotomy: 이분법, conjugation: 활용, coarse: 거친, demographics: 인구통계, calibrate: 측정하다,   
non-stationary: 비정상(불규칙), efficacy: 능률, outsized: 엄청난(대형의), propagate: 전파하다,   
cohort: 집단, counter-intuitively: 반집단적으로, segregated: 분리된, ordinal: 서수(순서),   
cardinality: 특정 데이터 집합의 유니크(Unique)한 값의 개수, churn: 마구 휘젓다, truncated: 끝을 자른,  
notoriously: 악명 높은,  held-out: 보류(중단), coarse: 조잡한, asymetric: 비대칭  


# 1. Intro  
유튜브에서 추천 어려움  
1) 스케일: 너무 큼(작은 것이 잘 동작)  
2) 프레쉬니스: 새로 나온 거 잘 추천해주기 어려운  
3) 노이즈: 기존 유저 정보 sparse하고 metadata 부실(잘 정리된 온톨로지가 존재 x)  
그치만 구글 브레인의 텐서플로우를 통해 유연하고 대용량 분산 좋은 모델링 가능  
*MF에서 DNN추천 적용  
*DNN은 CF, 크로스도메인, 음악 추천등에 쓰이고 있음  
*본 논문에서 간단한 오버뷰, 후보생성모델, 실험결과, 랭킹모델, 결론과 교훈 설명할 것  

# 2. System Overview  
시스템 전반  

*후보 생성 모델과 랭킹 모델로 이루어짐  
**둘 다 DNN모델  
*후보 생성 네트워크의 input은 유저 유튜브 히스토리임  
**탐색은 큰 코퍼스부터  
***CF용으로 넓은 퍼스널리티 제공됨  
*유저간 유사성  
**비디오, 서치쿼리, 인구통계보고 측정  
*랭킹 네트워크  
**비디오, 유저관련 목적함수 사용  
*이 두 시스템이 대용량 코퍼스에서 추천 가능케 해줌  
*프리시전, 리콜, 랭킹로스 등의 offline metric썼지만, A/B test했고 live로 실험함(ctr변화와 재생시간 봄)  
**offline metric과 A/B test 연관성이 항상 있지는 않았음  


# 3. Candidate Generation  
후보 생성   


*추천 전 - MF로 접근, 랭킹로스로 학습  
*factorization 흉내 네트워크(input: 유저 이전 시청) - > 비선형 factorization  


# 3.1 Recommendation as Classification  
분류로서 추천  

*멀티클래스 분류로 봄  


# Efficient Extreme Multiclass  
*train 위해 네거티브 샘플 씀  
**계층 소프트맥스 안씀.. 왜냐하면 정확도 떨어지기 때문인데 떨어진 이유는 트리 건널때 상관없는 것들 포함되기 때문  
*서빙타임서 N클래스 계산해야함  
**latency로 10ms 걸림  
**기존엔 hash와 분류기 씀  
**소프트맥스 서빙타임 없어서 scroing time이 nearest neighbor 서치로 줄어듬(dot product)  


# 3.2 Model Architecture  
모델 설계  
*CBOW에 영향 받음 - 임베딩 만들어 씀(고정 어휘)  
*이 임베딩(유저기록들)->FFNN에서 분류하다록  
**임베딩 평균을 내서 넣는 것(성능 괜춘)  
*GD, Back-Propagation 씀  
*피처들 F.C로 연결, ReLU씀  


# 3.3 Heterogeneous Signals  
이형 신호  
*DNN->MF  
**여러 카테고리 피처들 쉽게 합쳐짐(쿼리, 인구통계, 지정학적, 성별, 로그인정보 등 정규화해서)  


# "Example Age' Feature  
예제 나이 피처  

*리프레쉬니스 중요(시청자가 선호해서)  
*근데 ML은 과거데이터에 의존적(히스토리 학습하니)  
*비디오인기분포는 불규칙적인 multinomial분포임  
*추천에서 평균시청 likelihood 주마다 반영해줌  
*잘못된 것 정정위해 feature 0으로 만들어줌(window end마다)  


# 3.4 Label and Context Selection  
라벨과 컨텍스트 선택  


*추천은 대이인문제와 결과를 특정 문맥으로 바꾸는 것을 해결해야함  
**정확한 평가 예측은 효과적 영화추천으로 볼 수도 있음  
*대리인 문제 A/B 테스트 때 중요했는데 offline 땐 추정에 어려움이 있음  
*학습예제-유튜브 시점  
*새 컨첸츠 추천이 어렵  
*만약 유저가 새로운거 보면 CF로 바로 바녕ㅇ  
*고정된 트레이닝 개수 생성  
**small 집단 방지  
*의도적이지 않게 정보 제외해줘야 있어야 함  
**대리인문제 등 오버피팅 방지  
*주기적 이전 데이터 놓아주기?  
*다음에 뭘 볼지 예측하는 것에 focus를 두게끔  


# 3.5 Experiments with Features and Depth  
피처와 뎁스 실험  

*1M video, 1M 서치토큰, 256 floats 최근시청 50개, 최근 서치50개  
*softmax output-multinomial  
*유튜브 유저로 학습. tower 패턴 구조 네트워크(그냥 쌓은 거)  


# 4. RANKING  
랭킹  
candidates 수백으로 줄여서 좀 더 정확하게 찾아줌  
*DNN사용  
*A/B test로 성능 올림  
*CTR보단 watch time이 더 좋은 척도였음  


# 4.1 Feature Representation  
피처 표현  
*사용피처  
**전통적 카테고리외에 연속/ordinal(서수)  
***아이템인지 user context인지 구분해줌  


# Feature Engineering  
피처공학  
*수백 피처들 카테고리&연속으로 나눔  
*피처들 modi해줌  
*user, video data ->유용피처로  
*근데 일시적 유저행동은 어케해야하는지는?  
**밑에 답  

*유저의 이전 상호작용 중요시  
**item 스스로, 유사 아이템, 다른 이의 advertisement 매치  

*피처와 이전비디오 평가 중요 "churn(마구 휘젓다)" 추천  


# Embedding Categorical Features  
임베딩 카테고리 피처  
*sparse한 카테고리 -> dense하게 임베딩 사용  
*공유임베딩 종재 -> 일반화 향상위해 중요하고, 트레이닝 때 스피드향상 위해서 중요, 메모리 줄이기 위해서도 중요  


# Normalizing Continuous Features  
정규화-연속피처  
*NN 스케일링에 민감  
*정규화 중요-수렴 결과값위해  
*제곱 or 루트화 ->offline test서 성능 향상  


# 4.2 Modeling Expected Watch Time  
모델링-기대 시청 시간  
*기대 시청 시간->웨이티드 로지스틱 리그레션 사용  
*로지스틱 리그레션 with 크로스엔트로피  
*positive->청취시간에 가중 줌  


# 4.3 Experiments with Hidden Layers  
히든레이어 실험  
*히든 레이어 달리 하여 실험  
*width 증가시 depth 증가처럼 성능 올라감  
*CPU타임은 오래걸려서 트레이드오프  
*wide 줄일 수록 loss 0.2%씩 증가  
*긍/부정 가중 안 줄 경우 4.1% loss 증가  
**성능이 떨어짐  


# 5. CONCLUSIONS  
결론  
*DNN추천 -> candidate generation & ranking  
*Deep CF가 MF 압도  
*대리인문제->미래분류 성능 연관  
*신호버리기 중요(직전 정보)->오버피팅방지  
*bias 없애고 시간 정보 추가시 A/B test의 offline prediction성능 향상  
*DL Rank가 (감상 시간 예측)이 이전 모델들 압도  
*임베딩사용  
*정규화함  
*비선형 layer사용함  
*로지스틱회귀 가중 써서 감상시간 예측  
*랭크만 하는 것 보다 성능 향상됨  



