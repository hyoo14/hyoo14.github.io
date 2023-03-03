---
layout: post
title:  "[2022]Masked Summarization to Generate Factually Inconsistent Summaries for Improved Factual Consistency Checking"
date:   2023-03-02 22:36:33 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

*추상 요약 발전에도 소스 text부터 요약이 factual 일관적인지 판단하는 것 어려움  
**최근 접근법: 분류기 학습으로 factual 일관인지 비일관인지 구분  
**데이터로 factual 일관 data 있음  
**하지만 비일관 데이터를 만드는 것은 어려움  
**특히 비일관이지만 소스와 연관된 데이터를 만드는 것은 더 어려움  
**본 논문에서는 비일관 요약 생성기를 제안  
***소스 text와 keyword가 마스킹된 요약 데이터를 사용
**7개 벤치마크 test에서 기존 모델들 압도, 인간과도 견줄만 함  
**본 논문에서는 제안 모델의 요약 특성 분석도 제공  

    
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1_1zgNf_bNOYL69_6rqo46L9N2f1IGUU9?usp=sharing)  


[Lecture link](https://aclanthology.org/2022.findings-naacl.76.mp4)  


# 단어정리  
*diverge: 발산, 나뉘어지다, 하나로 모아지지 않다  
*affinity: 관련성, 밀접성  
* augment: 늘리다, 증가시키다  
* salient: 가장 중요한, 핵심적인, 현저한  
* hallucination: 환각, 환영, 환청  



   

# 1 Introduction  
* 텍스트 데이터가 많아지고 요약에 뉴럴모델 도입하면서 성능도 많이 좋아짐  
**추출보다 요약 생성 위주로 발전 중    
***비일관성 문제가 생김  
**기존 접근은 새 리소스 없을 경우 식별이 어려움  
**유사성 척도는 ROUGE&BLEU (n-gram 유사도 기반)  
***일관성 평가에서 인간의 평가와는 낮은 연관성 보임  
**연관 task이용 NLI&fact확인은 이상적인 방법은 아님  
***두 문장 사이 관계 식별이기 때문  
***사실적 일관성은 여러 문장 관여, 길이가 김   
*남은 해결 방법은 사실 일관 분류기  
**긍정 요약 판독은 있음  
**잘못된 요약이 없고, 만들어야 함  
**단문 키워드 대체는 diverge 함  
***너무 명확해서 유용성 떨어짐   
*Masked-and-Fill 제안  
**Masked Article(소스 요약 text->masked)  
***infer로 가능한 걸로 mask 매꿈(but 사실 비일관적인)  
**7개 벤치마크서 성능 압도적  
**인간 평가와 연관  
**특성 연구도 진행함  
**요약  
***새 부정 요약 생성기(비일관 사실 분류 text 생성)  
***효과성 보임(인간평가와 연관성 높음)  
***특성(affinity&diversity) 분석 진행   

<br/>


# 2 Related Work  
## 2.1 Factual Inconsistency in Summarization Systems  
*사실적 비일관 요약 시스템  
**요약서 30% 에러->사용성 떨어짐  
**두가지 오류  
***내적 : 소스 내부 컨텐츠 혼동  
***외적 : 소스 무시 & 생성 -> 포함 안 된 내용 생성  
**이러한 다양한 에러 탐지 시스템 제안  
***에러 종류별로 포함하여 build 제안  


## 2.2 Measuring Factual Consistency  
*사실일관 평가 위해 QAGS&GuestEval 연구들은 Q gen & A gen 프레임워크 체택(사실체크 먼저하는)  
**엔티티&명사구 사용(candidate 요약서)  
**답과 Q 소스 & 요약 사이 비교  
**요약 필요 없지만 인간 판단과 높은 연관성 보임  
**복잡구조, 계산 비쌈  
***헤비 & 오류 연쇄 있음  
**요약은 소스 다큐에 수반됨에서 idea 얻음  
**연관 task NLI가 사실 일관 검사 때 이용  
**QA보다 낫지만(간단하고 직관적이어서) 단일 문장 위주임  
**그래서 관련 연구는 synthetic dataset 만드는 것 위주  
**사전 정의 룰 사용  
***엔티티 교체 mask&fill  
**본 논문이 더 일반적인 반대 요약 생성 기법 제안  
***추가적으로 masked 소스 이용  
**CoCo는 likelihood 비교(생성 요약의)  
***오리지널과 mask&align  
**본 논문은 바로 반대요약 만들고 train for 분류기  


<br/>  


# 3 Methods  
** A: article, S: summary  
**S가 사실 일관인지 아닌지 체크  
***Sc & Si 구분  
*조건  
**비일과 보장  
**소스와 연관되야  
***일고나 체크가 잘 되야함  
***소스로부터 생성  
**복잡, 혼란수런 부정 요약 위해  
***masked article 사용  
***masked refer 요약 사용  
***주요 정보 숨긴  
**요약 모델이 추론(mask를)  
**이전엔 그냥 대체  
***부정은 확실했으나 소스와 무관해짐  


## 3.1 Mask-and-Fill with Masked Article  
*MFMA  
**비일관 but 연관은 된 요약 만드는 Masked&Fill with Masked Article  
**masked 소스 & 요약 갖고 만듬   
**아티클의 명사구와 엔티티가 중요 정보라 가정하고 이를 마스킹  
**긍정 요약에서는 주요 부분(엔티티, 명사구) 특정 비율로 마스킹  
**소스 + 요약 concat  
***학습 시킴(인코더 디코더 바트로)  
***오리지널 복구하게(목적함수)  
**학습 후 부정 요약 생성->unseen&masked 요약 페어로  
**mask 비율 높으면 쉽게 복원 못 함  
**하지만 그럴듯한 채우기는 가능  


## 3.2 Masked Summarization  
*마스킹 요약  
**확장판으로 다양 버전 만듬  
***마스크 요약  
**마스크 된 소스로 요약 만듬 -> 목적함수: 마스크 없는 요약으로 만드는 것    
**더 다양 요약 만듬  
***infer guide 없이 학습하므로  


## 3.3 Training Factual Consistency Checking Model  
*사실 일관 체킹 모델 학습  
**ELECTRA F-T 함  
***classification head 추가하여 이진 Cross Entropy loss 로 학습  


<br/>


# 4 Experiments  
## 4.1 Implementation Details  
### Negative Summary Generation  
*실험  
**랜덤하게 CNN/DM 데이터셋 분리  
***반은 학습 fro 부정 요약 판단  
***나머지 반은 부정 요약 생성  
**spaCy 사용  
***entity & 명사구 찾음  
**바트 베이스 학습  
***speach for MFMA  
**F-T 없이 사용하기 위해  
**T5 small 사용(MSM 위해)  


### Training Classifier  
*분류기 학습  
**구글/electra 판별기 학습  
**5epoch, l.r=2e-5, batch size-96, Adam opti, data는 MF, MFMA, MSM서 생긴거 사용  
***DocNLI, FactCC의 경우 오리지널 데이터셋 사용(저자가 공개한)  
**평가 위해 1K 인간 주석 요약 사용  


## 4.2 Benchmark Datasets  
*벤치마크 데이터셋  
**인간 레이블 중요(이진 또는 여러 레벨로 나뉜 거)  
**이진: accuracy로 판단  
**여러레벨: correlation with 인간평가 로 판단  


### FC-Test  
*FC-Test  
**레이블 CNN/DM 요약 대한 테스트  


### XSumHall  
*BBC XSum 요약 error 체크, 2K 데이터셋 바이너리  


### SummEval  
*인간 판별 점수 1600 요약, CNN/DM test set기반, 1/2/3/4/5 점수인데 5점은 consistency, 나머지는 inconsistent로 세팅  


### QAGS-CNN/DM & XSum  
*235 요약 인가평가 from CNN/DM 데이터셋  
**3개 레이블: 1개는 inconsistence, 2개는 consistent  


### FRANK-CNN/DM & XSum  
*FRANK 데이터셋 요약, 2246개, 1250from CNN/DM, 996from XSum  
**binary로 변환  


## 4.3 Baseline Metrics  
*베이스라인  
**베이스라인 메트릭 손수 계산  

### Entailment Based Metrics  
**수반 메트릭 : MNLI&FEVER data 사용, FactCC & NLI 모델서 사용  


### QA-Based Metrics  
**QA 기반, QuestEval 사용, 서머리와 아티클 사실 score 비교  


### N-gram Similarity Metrics  
*N-gram 유사도  
**BLEU&ROuGE, METEOR 요약 평가  
***ROUGE-L: F-measure base, 긴 subseq 찾는 것으로 많이 쓰임  


### Other Metrics  
*기타  
**버트스코어, cosine 유사도 기반  
**CoCo : likelihood 차이점  


## 4.4 Results  
### Classification Accuracy  
*결과  
**데이터 불균형해서 macro-F1 & class 벨런스 정확도 보여줌  
**MFMA 나은 성능 보임, 특히 CNN/DM서  
**XSum 서는 다른 것들과 유사  
**CNN/DM만 트레이닝 셋으로 썼기 때문  
**DocNLI는 인간레이블 씀, ANLI/SQuAD 같은  
**MSM도 능력 보여줌  
***성능은 XSum 기반 보다 좀 떨어짐  
**MSM과 MFMA 사이 gap은 바로 요약 여부라고 봄(noise 판별 잘 돈지 안 된지 따라)   


### Correlation with Human Judgments  
*인간 평가와 얼마나 연관되는지   
**본 논문 제안 모델이 가장 높은 성능  
**스피어맨 상관계사 사용  
**수반 관련 계산 쉬움  
**제안 모델 성능 굿  


## 4.5 Analysis and Discussion  
### Performance among Masked Ratio  
**마스크 비율 영향 분석  
***trade-off임  
**too much mask -> performance down  
**too small mask -> 똑같은 샘플 만듬  
**optional masking ration 추론 해봄  


### Generated Samples among Masking Ratio  
*생성 부정 요약 시각화 해봄  
**masking ratil rA 너무 낮으면 원본과 거의 같음  
**rA 너무 높으면 너무 달라짐  
***아티클 없이 채운 수준  


### Performance among Masking Unit  
*마스크 유닛 따른 성능  
**masking 명사구, 엔티티에 수행 요약과 소스 둘다  
**단어 레벨 수행 및 비교   
**명사구 레벨 마스킹이 성능 좋음  


### Distance from Original Reference Summary  
*원래 요약과 평균 거리  
**버트 스코어로  
**dist 0.8 일 때 성능 최고  
**얼마나 구조적으로 부정요약이 (거리가)먼 지 보여줌  
**체킹모델로 학습때 유용지표  


### Diversity among Masked Ratio  
*본 제시 모델:  mask 위치 따라 샘플 다양하게 해봄  
**MFMA mask 비율 테스트  
**4개 부정 요약 샘플(방법 따라)  
***pairwise sim 계산 & bert score로 계산  
***다양성은 비슷  
***R**2=0.7일 때 성능 최대  


### Case Study  
*장, 단점 체크 위해 케이스 스터디  
**성공/실패 사례  
**사실 판단 때 제안 시스템이 좋음  
**완벽은 아님  
**MFMA와 MSM 더 잘 요약 되며 성능 올라감  
**reasoning과 check 더 필요  


# 5 Conclusion  
*효과적 생성 모델 제안(비일관 요약)  
**MFMA  
**소스 & refer 요약, 숨겨진 면 요약 모델이 올바른거 but 비일관 요약 생성(mask 바꾸어가며)  
**7벤치 마크서 classifier의 성능이 압도적  
**인간 평가와 연관성도 좋음  