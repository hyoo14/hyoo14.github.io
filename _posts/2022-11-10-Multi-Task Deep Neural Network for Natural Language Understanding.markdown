---
layout: post
title:  "Multi-Task Deep Neural Network for Natural Language Understanding"
date:   2022-11-10 15:00:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

NLU용 MT-DNN 제안  
-> cross-task data 사용 + regulation을 통해 더 general한 모델 만들어서 new task 용이하게
-pre train 버트 사용  
-10 NLU task(SNLI, SciTail, GLUE 중 8개 task) test, GLUE 82.7%로 sota보다 2.2% 향상  
-실험 결과 적은 도메인 라벨로 도메인 적응 잘 함, SNLI&SciTail로 실험  

{% endhighlight %}


[링크](https://drive.google.com/drive/folders/1MOO0lPvXHxD0f4Puq0QTtkvnTZPAf8Yz)


# 단어정리  
*o: ㅇ  


# 1 Introduction  
* 단어나 문장으로 이루어진 벡터스페이스 표현을 학습하는 것은 NLU의 핵심  
**multi-task learning, pre-training이 대표적  
**본 논문에서는 둘 다 쓰는 MT-DNN 제안  
*MTL(multi-task learning)은 사람에서 영감 받음(예를 들어 스키 잘 타는 사람이 스케이트 잘 배운다)  
*2측면서 장점이 있음  
**DNN은 데이터가 많이 필요한 반면 MTL은 데이터 덜 사용  
**regulate 효과 줌->오버피팅 방지  
*PLM은 대용량 언레이블 비지도 학습  
**멀티 레이어 양방향 트렌스포머 버트 성공적  
**엔트테스크 파인튜닝에서 sota 보임  
*MTL과 PLM은 상호보완적  
**텍스트 대표성 향상(적용성 )  
**NLU task들에 적용  
**original MT-DNN에서 버트로 확장한 것을 제안  
**아래는 버트처럼 shared task layer  
**맨 위는 task specific(문장분류, 문장쌍 짝짓기, 유사도, 관련 랭킹)   
**버트와 달리 MTL사용하여 파인튜닝  
*MT-DNN sota 달성  
**GLUE NLU서 82.7(2.2향상)%  
**SNLI, SciTail 각각 1.5, 6.7% 향상  
**SOTA  



# 2 Task  
*MT-DNN 4type NLU(GLUE) 사용  
**단일 문장 분류  
**text쌍 분류  
**text 유사도  
**연관성 랭킹  

*단일문장분류   
**COLA: 문장완성도 여부  
**SST-2: 영화 긍,부정  
*text 유사도  
**리그레션 task  
**의미적 유사도  
**STS-B 유사도 스코어  
*text쌍 분류  
**두 문장 관계  
**RTE&MNLI entail, contradict, neutral 3가지 tag  
**QQP&MRPC: 파라프레이징  
*연관랭킹  
**Query&candidate answer  
**QNLI: QA데이터셋  
**sent가 정답 포함하는지 (query에 대한)  
**GLUE의 이진 분류  
**랭킹으로 해서(포함여부) 정답 확률 높임  


# 3 The Proposed MT-DNN Model  
*MT-DNN 아키텍처  
**아래 shared task layer(sent to embeddings)   
**위 task-specific layer(트랜스포머 인코더 context capture-셀프어텐션으로)
*** multi-task러닝 통해 임베딩 생성(반영)  
*Lexicon 인코더  
**인풋: {x1, ..., xm} x는 각 토큰  
**x1은 cls토큰  
**x가 센텐스 페이이면 sep토큰으로 구분  
**x를 임베딩과 매핑  
**포지션 임베딩과도 매핑  
*트랜스포머 인코더  
**멀티레이어 양방향 트랜프포머 사용  
**input벡터로부터 contextual 임베딩벡터화->다른 task들과 share됨  
**버트 + 멀티테스크 목적함수 사용  
**GLUE NLU task 예시로 아래 설명 + text generation  
*단일 문장 분류  
**token(cls)의 x는 context임베딩(l2)  
**라벨 확률로 softmax logistic regression 식 사용  
**SST-2로 처럼  
*텍스트 유사도  
**SST-B 처럼  
**x는 cls 포함, pair x1, x2의 유사도 계산  
*텍스트 쌍 분류   
**NLI task, P(p1, .., pn)과 H(h1,..,hn) 관계 R  
**SAN(Stochastic Answer Network) 따름  
**multi step reasoning  
**iterative prediction->entailment  
P concat 임베딩(트랜스포머 아웃풋) Mp로 표기  
**H->Mh  
**k번 reasoning, 메모리/라벨 관련성 K(하이퍼파라미터)  
**S는 서머리  
**alpha는 step range  
**Sk는 GRU로 사용  
**이전 state sk-1  
**메모리 Mp(knn처럼 여러 스텝)  
**마지막으로 P의 k개 avg함  
**stochastic prediciton dropout 사용  
*연관 랭킹 결과  
**x는 문맥 임베딩 [cls]와 연관된  
**g(WqnliT * X)  
***웨이트 * x, g식에 넣어 연관도 계산  
**학습 절차 MT-DNN  
**1. 프리트레인->MLM&NSP  
**2. 멀티테스크러닝->미니배치 SGD기반 사용, 파라미터 업데이트, 9GLUE로 최적화  
**분류테스크  
***크로스엔트로피 목적함수, 1(x,c)는 이진, c는 정답, Pr은 Eq 1 or 4  
**STS B유사도 task서 mse 사용  


*연관랭킹task 서 pairwise 러닝-랭크 패러다임  
**QNLI 예시  
**후보 뽑고  
**긍정 A+는 답  
**|A|-1은 부정  
**negative log likelihood로 트레이닝  
**Rel은 eq5로  
**alpha는 튜닝펙터  


# 4 Experiments  
*MT-DNN평가  
**GLUE, SciTail  
**비교는 버트모델  
**파인튜닝 GLUE -> SNLI, SciTail  


*GLUE, SNLI, SciTail 데이터셋 묘사  
*GLUE : General Language Understanding Evaluation  
**QA, 감정분석 텍스트 유사도, 텍스트 포함 여부  
**NLU평가  
*SNLI  
**570K인간 주석 문장쌍  
**Flickr30코퍼스로부터 만듬  
**NLI를 위한 유명한 entailment 데이터셋임  
*SciTail  
**텍스트 수반 데이터셋  
**Sci QA 추출  
**전제와 이론 수반 여부  
**이론은 과학지식  
**후보는 검색으로 얻음(다른 코퍼스에서)  
**문장은 어렵 유사도는 큼(전제와 이론 사이)  
**도메인 학습에 사용  
*구현 디테일  
**파이토치 버트 기반  
**l.r : 5e-5  
**32배치사이즈  
**max 5 epoch  
**linear l.r decay : 0.1  
**MNLI만으로 COLA 0.05  
**explode gradient 방지 위해 clip함  norm 1으로  
**워드피스 토크나이저 사용  
**길이제한 512  
*GLUE MT-DNN 실험 결과  
**리더보드와 비교  
*버트라지  
**베이스라인  
**파인튜닝모델 task-specific for GLUE  
*제안모델  
**프리트레인 버트라지 사용  
**GLUE로 MTL 파인튜닝  
**SOTA 성능 보임, MNLI빼고 8개에서  
**82.7%로 2.2%향상, 버트라지대비  
**MT작은 도메인 학습으로 유용  
**도메인 트레이닝 적을수록 향상에 편차 큼  
*MT-DNN 파인튠 안한 결과  
**COLA빼고 버트라지 압도  
**COLA는 작은 도메인이고 유니크해서 MT-DNN성능 떨어짐   
**MT-DNN 모든 도메인 적응에 효과적  
**task-specific에 좋음(framework도 있고)  
*ST-DNN: Single Task DNN  
**MT-DNN과 같은 구조  
**버트만 공유, MTL로 refine x  
**ST-DNN 파인튠 GLUE  
**버트 압도  
*도메인 적응 결과  
*빠른 적응 specific task에 중요  
**아주 적은 데이터로 학습 경우가 많기 때문  
**도메인 적응 실험 진행함  
**NLI, SNLI, SciTail 사용  
**과정  
***1. MT-DNN, 버트 이니셜 모델임  
***2. new task model 만듬, SNLI, SciTail  
***3. 평가  
**랜덤샘플 : 0.1%, 1%, 10% SciTail 4 set로 train, SNLI도 4set  
**랜덤샘플링 5회  
**평균으로 결과뽑음  
**MT-DNN이 버트 압도  
**82.1 vs 52.5(SNLI), 85.2 vs 78.1(SciTail)  
**MT-DNN라지가 SNLI, SciTail서 SOTA  


# 5 Conclusion  
*MT-DNN제안  
**MTL + PLM  
**NLU에서 SOTA(SNLI, SciTail GLUE)  
**적응도 굿  
*퓨처 워크  
**모델 이해 더 잘 하는 것  
**더 효과적 학습 방법(연관성 이용과 같은, pre-train과 fine-tunning 모두에)  
**언어구조 사용  
**광고에 강건한지 테스트 등  