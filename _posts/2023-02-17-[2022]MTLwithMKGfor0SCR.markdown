---
layout: post
title:  "[2022]Modularized Transfer Learning with Multiple Knowledge Graphs for Zero-shot Commonsense Reasoning"
date:   2023-02-17 20:14:33 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

* CR(Commonsense Reasoning) 여러 추론 일반화 필요하지만 sota모델들은 그렇지 못함  
* 비싼 데이터레이블에 일반화없이 오버피팅됨  
* 대안으로 제로샷 qa 제안  
* 상식 KG -> qa생성 -> 이걸로 학습  
* KG 여러개 합쳐 시너지 얻음  
    
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1E9wjgl5kX5Q8HRfq6jPRtiDq755cNOFC?usp=sharing)  


[Lecture link](https://aclanthology.org/2022.naacl-main.163.mp4)  


# 단어정리  
* neglecting: 방치      
* pitfall: 함정  
* repurpose: 용도변경  
* mitigating: 완화  



   

# 1 Introduction  
*CR은 NLP의 코어테스크 중 하나  
*연관 데이터셋으로 SocialQA, CommonseseQA, PhysicallQA 등이 개발됨   
*layer-scale neural system으로 인간 정확도 보임  
*일반화가 안 되서 비판받음  
*평가척도로 Ma의 OS제안, 이게 옳음(본 논문의 주장)  
*비지도 제로샷-다지선다 qa  
*기존 테스크 특화 cr은 제로샷 시나리오 ㅈ거용 안됨  
*레이블 없이 학습위해 atomic이나 conceptnet같은 kg 사용  
*관계추론 주로 사용, 근데 비현실적   
**현실은 선언/사회/원인/물리적 추론 필요  
*다양 추론 고려 위해 멀티소스케이스 0s 러닝  
*상식 kg들로부터  
*atomic은 사회상식  
*conceptnet은 개념 사용  
*mtl 꽤 유용하지만 이전에 배운거 잊고 다른 kg간 간섭이라는 단점이 있음  
*한계극복법:  0s fusion  
**adaptFusion사용  
**트랜스포머 블록 사이 작은 모듈(adapter)은 학습 후 합쳐지는데, 지속 통합을 허용하는것, 다시 재학습 하지 않게 하는 것  
**예를 들어, 다른 kg기반 adapter(expert)들은 어텐션같은 퓨전 모듈로 합침   
**(1)kg기반 qa로 학습 (2)kg주변은 kg분류 data로 학습 (3)밸런스 있는 혼합  
*interference도 이용(kg 사이)  
*공헌  
**1)심플,효과적 cr용 kg제안 2)adapterFusion탐구, 데이터축적에 도움->시너지 얻음 3)실험통해 제안 프레임웍 통해 성능항샹  


# 2 Related Work & Preliminaries  
## 2.1 Zero-shot Commonsense Reasoning  
*비지도 학습 주목 받는 중  
*kg가 사용되는 cr  
*대부분은 그냥 가정 안 함  
*ATOMIC 20 20 KG와 tuple(conceptNet & atomic 사용)  
*다양 kg 합쳐짐  
*kg 합쳐지는 거 연장   


## 2.2 Transfer Learning with Modular Approaches  
*KG합침 -> T.L(transfer learning)   
*plm일부 para F-T  
*share부분은 fix  
*forget 해결하는 것 focus  


## 2.3 Multi-Task Learning  
*shared 된 것 학습  
*para 축적 mtl 약점 보완    
**1) 모델 task 전부 보유해야  2) 모두 성능 좋기 어렵 3)일관성 부족 문제  
*해결 위해 Mixture of Exper parameter 일반화,앙상블기술,gate사용  
**하지만 cr용으로 부족  
*해결책  
**adaptation fusion   
**task-speckfic adapter 잘 합침  
**어텐션 이용  
**independency 학습  
**forget 방지  
**0S cross lingual transfer framework제공됨 이용  
***여기서 영감 받음  


# 3 Modularized Zero-shot Framework  
*KG로 QA generation함  
**KG triplet = (e head, r, e tail)   여기서 e는 entity, r은 relation  
**템플릿 사용( e head + r -> Q   /   e tail + e (m-1개의 distractors from others randomly) -> A  
**레이블은 정답인지 아닌지로 사용  
*KG 효과적 사용 위해 모듈화 프레이밍  
**여러 kg adapter들 학습  
**zero-shot fusion method 사용  
*AdapterFusion 기반 + KG classifier adapter(어느 kg인지 고려)  


## 3.1 KG Modularization  
*KG 모듈화  
**downstream에 잘 맞게 align  
***adapter module채용 -> plm 연계 잘 되도록 하는  
**새 layer추가해서 연계 잘 되기 함  
**PLM weight는 touch x  


## 3.2 Zero-shot Fusion  
*제로샷 퓨전  
**어텐션 같은 메커니즘으로 knwledge합침 from expert adapter서 학습된  
**각각에 맞는 KG잘 활용하도록 하는 것 목표  
**expert adapter 통해 block  
***제로샷 퓨전으로 balancing  
***generalization 굿  


## 3.3 KG-Classifier Adapter  
*KG분류기 Adapter  
**Adapter Fusion은 PLM히든 representation사용  
**본 논문: SyntheticQA mixture사용  
***다운스트림용 아님  
**그래서 kg분류기 사용  
***뭐가 kg alignment adapter인지 분류  
***어떤 kg가 도움되는지 찾음  
*새 train방법 제안  
**kg-분류기 adapter  
**맞는 kg찾아줌  
**크로스엔트로피 낮아지는 방향으로 학습  
**kg분류기 adapter를 어텐션의 query로 사용  


# 4 Experiments  
*cr의 효율적 테스트 시행  


## 4.1 Experimental Settings  
*세팅  
**오피셜 트레이닝 데이터(벤치마크의) 접근 x  
**valid set사용(test용으로만)  


### 4.1.1 Benchmarks  
*벤치마크  
**5개 qa cr 사용  


### 4.1.2 Baselines  
*벤치마크 특성 테스트  
**로버타L & GPT2L(f-t 안 한 것들) 사용  
*KG사용-베이스라인용: self-talk, COMET-DynaGen, SMLM사용  
*더 분석 위해 p-t synthetic qa로  


#### Single-Task Learning(STL)  
*다음처럼 single task 러닝 plm & plm+adapters 경우 따짐  
**atomic qa, concept net, wiki, wordnet 데이터 사용   


#### Multi-Task Learning(MTL)   
*MTL  
**P-T 여러 synthetic QA  데이터로 test  
***이어지는 synthetic qa 데이터로    
**stl과 mtl 차이 확인  


### 4.1.3 Implementations  
*구현  
**로버타 L 사용(허깅페이스)  
**Ma의 세팅 사용  
**Adapter & AdapterFusion 사용(베이스로) (AdapterHub로부터 가져옴 )  
**3개의 다른 랜덤시드로 실험  


## 4.2 Main Results  
*결과  
**일반적으로 제로샷퓨전이 베스트 결과(모든 벤치마크 중 WG제외하고)  
*경쟁력있는 결과 나옴 WG에서도 베스트는 아니었지만  
*제로샷 퓨전 일관적 성능 향상 보임  
**kG-C없는 경우 test해보면 있는 경우가 0.4% 나은 결과 보임  


## 4.3 Impact of the KG-Classifier Adapter  
*KG-C 효과 확인  
**시각화  
**t-SNE plot 확인  
*KG-C가 분리 잘 함을 확인  
**KG-C가 제로샷 퓨전 끼치는 영향 탐구  
**KG-C가 PLM을 쿼리로 사용  
**제로샷 퓨전 + KG-c는 kgc-c을 쿼리로 이용  
**시각화 결과 어두운 셀은 adapter를 나타냄  
**미묘한 것 잘 구분(kg-c + zero shot fusion)  


## 4.4. Mitigating Interference   
*간섭 경감  
**새 메트릭 제안  
**ratio임  
**잘못된 예측 비율  
**부정 영향 확인용  
**MTL이 경쟁보다 나음  
**본 모델이 약간 나음 mtl보다  


## 4.5 Visualization of Knowledge Aggregation  
*지식 축적 시각화  
**성능 증가 비교 MTL & 0s fusion + kg-c  
**초록: 이득/ 빨강: 손해(시각화비교 상에서)  
*MTL은 kg가 늘어나면 성능 떨어짐  
**본 논문 제안 모델은 MTL보다 덜 떨어짐  
**본 논문의 성능이 굿임  


# 5 Conclusion  
*많은 상식 kg에도 cr활용에 불충분  
*모듈화 transfer러닝으로 kg 잘 합침  
*kg 모듈과 expert어댑터 사용, 제로샷퓨전 + kg-cfh whgdms tjdsmd djedma  
**mtl대비 우위(5개 벤치마크에서)  
*더 많은 kg사용 기대  
*최적합으로 kg모듈화 사용하여 최적 전이학습 탐구 기대   
