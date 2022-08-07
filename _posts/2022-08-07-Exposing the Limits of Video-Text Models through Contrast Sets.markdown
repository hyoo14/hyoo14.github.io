---
layout: post
title:  "Exposing the Limits of Video-Text Models through Contrast Sets"
date:   2022-08-07 17:51:19 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

비디오-텍스트 분류에 적용할 시맨틱스 이해를 확인해 줄 평가 프레임웍 제공(자동 생성 되는 대조셋)  

텍스트 묘사 일부 변조(프리트레인LM과 휴리스틱하게 동사&사람 엔티티 사용)  

실험 결과 제시하는 set이 의미있었음(CLIP과 이전 것의 차이도 없앰..?)  

https://drive.google.com/drive/folders/1u1QFTNWJtwMTnrQpNiT9ftks47JDvypQ?usp=sharing

{% endhighlight %}


#단어정리  
*leverage: 활용하다, suffer:  저하되다, modalities: 양식/양상, perturbation: 동요하다 , antonym: 반의어 ,  foil: 은박지, pronoun: 대명사, mitigate: 완화시키다  

# 1. Intro  
비디오와 텍스트 모달리티 연결운 어려운 일(사람, 사물과 공간, 시간이 상호작용하므로)   
텍스트 서술과 비디오의 엔티티, 행동 이해는 매우 어렵  


비디오-텍스트 모델의 학습과 평가는 다음과 같음  
*크로스 모델 매칭 통해  
**탐색 문제로서 올바른 거 고르기(랜덤하게 방해요소 추가)  
*다중 선택 예측 통해  
**맞는 거 고르기(부정된 선택 match 허용하여 보기 추가)    
(탐색 세팅은 학습 때만 사용-특정 패턴, 편향 피하기 위해(평가 때는 모두 사용)  )  


최근 라지스케일 클립모델 이용 방법이 SOTA(멀티모달 트랜스포머 기반) 보다 좋음(특히 검색에서)  
그러나 성능은 과대평가임  
평가 샘플이 부족해서인데 실제로 NLP task와 image-text에서 대조세트(사람 전문가가 만든) 실험 결과는 매우 안 좋음  


그래서 본 논문에서는 평가 프레임웍을 제시  
*강한 부정 체크  
*자동 파이프라인  
*대조셋 만들어 줌  
*사람 엔티티 바꿔줌  
*유창성 지키면서 시맨틱스 바꿔줌  
*엔티티 동사에 집중  
*T5 LM써서 바꿔줌  
*휴리스틱 하게도 함(사람 엔티티 바꿀 때)  


유명 비디오-텍스트 모델들에 실험(MSR-VTT, LSMDC)  
사람도 실험, 비교  


결과적으로 프레임웍에 의미 있었음  
제시한 데이터셋에 밴치마크들의 성능이 다들 저하되었음  
벤치마크들이 엔티티, 동사 바꾸었을 시 잘 구분 못 하는 것 확인  



# 2. Related Work  
관련연구  
*Defending&generating adversarial examples  
**NLP, PLM이후  
**단어 교체(문장에서, mLM이용하여)  
**분류&entailment 성능 떨어짐  
*Template-based & manually crafted  
**textual entailment에서 연구됨  
**체크리스트 제시(NLP)모델용  
*Language-based adversarial examples  
**비전-랭기지 모델용  
**FOIL-COCO(이미지 보고 텍스트에서 가린 거 찾기) 데이터셋 포함  
**visual reforming도 연구(단어 바뀌었을 때 robust한 지)  
**비전-랭귀지 트랜스포머는 동사에 약함(명사보다)  
**new VQA 데이터셋(VQA 강건성 체크용)  
**자동 생성 대조 셋(씬 그래프로부터 VQA 구성 일관성 실험)  
*본 논문은 perturbation과 evaluation서 PLM썼다는 것이 다름  



# 3. Designing Contrast Sets  
텍스트 기반 대조셋 자동 생성  
*용어 정의  
**비디오 Vi, 묘사 Si, 대조셋 ^C1={^S1,...,^Si}  
**대조셋은 1)사람 엔티티 대조, 2)동사구 대조 두가지로 만듬  


# 3.1 Contrast Sets for Person Entities  
사람 엔티티 대조  
'이름'을 자동 변환  
*LSMDC 데이터셋은 영화 묘사 있음, 캐릭터와 함께.. 그리고 캐릭터 리스트와 성별도 잇음  
*캐릭터 ID를 같은 성의 다른 캐릭터(같은 영화에 있는)로 바꿈  
**아예 없는 걸로 바꾸면  랭귀지 통계로 알까봐  


MSR-VTT셋은 ID없음  
그러나 80%는 성별 힌트가 있음  
*그래서 언급된 성별 바꿔줌  


# 3.2 Language Model Generated Verb Contrast Sets  
동사 대조 셋  
*동사 바꾸면 비디오와 연관이 없는지 보장을 못 함  
*문장 부자연스러워지고 문맥이 없어질 수 있음  
-> 그래서 PLM으로 자동으로 바뀌게 함  


동사구 Spacy로 식별하고 ->masking->top K 뽑음(LM통계로)(이 때 T5 LM 씀, 기존 mLM보다 후보 단어가 더 많아서 씀)  
(T5모델 동사구 배우게 파인튜닝함(비지도-denoise object, G.T 비슷하게 해줌)  


K후보 문장 필터링 함    
*verb 없거나  
*rare or unseen verb  
*PP 높음(GPT2-XL 상)  


semantics 체크  
*오리지널과 관련이 없는 지  


후보 동사가 반의어이거나 임베딩 similarity낮을 때는 반의어 따로 핸들  
임베딩 유사도로 잘 못 걸러주기 때문  


# 3.3 Human-Generated Verb Contrast Sets  
사람이 만든 셋  


모델이 잘 했는 지 확인하기 위하여 머신-사람 얼마나 유사하게 반응 했는지 측정  
(AMT로 사람 설문)  


# 4. Experiments  
# 4.1 Datasets and Multiple Choice Design  
데이터셋과 다중 선택 디자인  


*MSR-VTT  
**10K 유튜브 비디오(20가지 묘사 문장을 가짐)  
**검색 성능으로 평가(1000개 V-T pairs)  


*다중 선택 버전  
**2,990 테스트 비디오 쿼리 & 4 random positive caption(다른 비디오에서 가져온)  
**총 5개 선택지 만듬  


*본 논문에서 RandomMC로 라벨링 MC확률로 negative 하나, 우리 contrast로 바꿈  
Gender SWAP or Verb_LM_SWAP or Verb_HUMAN_SWAP  


*LSMDC  
**짧은 영화 클립과 자막(캐릭터 이름이 라벨링 된) 사용(하지만 이걸로만 대조셋을 만들진 못 함)  
**그래서 캐릭터 정보 있는 다른 자막 이용   
**new split만듬(같은 영화로, 트레이닝 때 테스트)  
**LSMCD-IDs 랜덤 MC의 경우 ID바꿔줌  



# 4.2 Video-Text Models and Evaluation  
V-T 모델 평가  
트랜스포머 비디오-언어 모델 벤치마크로 실험  
*Portillo는 frozenCLIP피처 이용, zero-shot 비디오->text 검색 수행  
*MMT(MultiModalTransformer)는 텍스트와 비디오 사이 joint 표현 학습  



본 논문에서 Dzabraev에 영감 받아서 MMT+frozenCLIP(input으로) 사용  
->MMT-CLIP  


*CLIP4CLIP & CLIP2VIDEO 바로 CLIP을 파인튜닝한 SOTA들임  


VIT-B/32 모델이 모든 CLIP 실험에 이용됨  
*위 모델들 contrast loss로 학습(V-T학습)  
*MC서 맞았다 판단(GT score 제일 높은 경우)  
*사람 MC도 체크  
*V->T Recall@1으로 평가(검색 성능)  


# 4.3 Results  
MSR-VTT결과의 경우  
*CLIP 파인튠과 나머지 gap 큼(심지어 CLIP 제로샷이 MNT보다 나음)  
*RANDOM MC 거의 다 품  
*Contrast set서 다 성능 저하 보임  
*contrast set서 파인튠 CLIP 우수성 사라짐  
*검색 성능이 좋다고 단어 수준 변화를 잘 알아채지 못 함  
*frozenCLIP이 GenderMC에서 더 좋음  
*파인튠 CLIP이 genderMC서 더 둔함  
*인간의 성능이 훨씬 좋음  


LSMDC-ID 데이터셋 결과  
*RandomMC MSR-VTT처럼 못 품  
*ID swap 더 쉬움, CLIP feature가 ID서 도움 됨  
*negative ID시 성능 original 보다 최소 13.9% 감소  
**모델이 ID식별 위해 long-tail로 노력하는 것을 의미함  
*verb 대조는 MSR-VTT와 유사  
*contrast가 랜덤 보다 성능 낮게 나옴  
*인간은 90% 이상의 정확도 보임   



# Does Semantic Proximity in Verb Contrast Sets Affect the Model Performance?  
의미론적 어림잡기가 동사 대조 셋에서 성능에 영향 줄까?  
*답 위해 subset(반의어 포함) 고려  
*나머지 위해 off-the-shelf 문장 인코더 사용  
*SentBert와 CLIP text 트랜스포머 이용(semantic approximation위해)  
*original & negative & 점수 상위,하위 각각 15% 고름  
*CLIP4CLIP 모델이 반의어에서 성능 떨어짐(83.7% -> 73.6%)   
*사람의 정확도는 계속 좋음  
*Sent BERT low에서는 사람보다 좋은 성능 보여줌  
*하지만 high에서는 사람이 더 나음  
*SOTA의 실패 사례를 보면 의미적 유사한 경우에 잘 이해하지 못했고, 반의어에 약했음->fine-grain lock으로 action을 이해하지 못한 것   


# 5. Conclusion  
사람엔티티와 동사구를 바꿔주는 자동 대조셋 만드는(V-T task용) 파이프라인을 제시함  
->모델은 랜덤 부정보다 더 어려워함  
->검색 잘 하는 모델도 성능이 떨어지는 것을 보임  
->모델 성능은 의미추정과 연관(사람과 달리)  


futuer work  
*컨트레스트 셋을 잘 이해하도록 학습  
*concept&POS contrast set 구축  


# 6. Ethical Considerations  
*데이터셋에 성별관련, 동사 관련 편견 있을 수 있음을 고려해주길    

