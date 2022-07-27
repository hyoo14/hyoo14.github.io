---
layout: post
title:  "Entity Cloze By Date- What LMs Know About Unseen Entities"
date:   2022-07-27 21:59:19 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :

오래된 LM으로 새로 나오는 entity infer 할 수 있어야함(test)  

이를 위한 data를 자동으로 모아주는 pipeline 제공  

특히, 날짜 정보와 wiki 문서 정보를 포함하는 entity로 구성된 데이터를 만들도록 도와줌  

{% endhighlight %}


#단어정리  
*derive, clozed, stub

# Intro  
새로운 엔티티는 계속 생기고, 기존 엔티티 기반의 것들은 커버 못 함  
LM은 real world의 엔티티를 잘 이해할 수 있어야함  

기존 LM벤치마크들은 제한된 관계만 반영한 cloze style task임  

예들은 아래와 같음  

*LAMA : 2017년의 40개 위키데이터의 관계와 엔티티를 사용  
*Newer-Cloze benchmark : temporal aspects 합침(시간, 기간 -> cloze 문장 valid확인은 하지만 새로운 것과 기존 것 구분 못 함)  
**리얼월드의 넓은 엔티티 test&evaluate fail  

그래서 framework를 이 논문에서 제안  
LM이 아는 entity관한 지식 평가(이 때 entity 기원시간으로 분류)  

이러한 기원시간 인덱스된 엔티티(Origination Data Indexed Entity, ODIE)뽑아냄 (위키 메타데이터로부터)  

실험데이터셋 과거꺼와 달리 이 cloze 문장 테스트는 모델이 넓은 범위의 엔티티 관련 추론능력이 있나 test할 수 있음(KB관계 제약 없이)  

이를 위해 주요한 entity 주변에 masking함, 그리고 raw한 문장과 엔티티대체 문장 사이의 perplexity gap을 평가함  

본 논문에서 그래서 이를 릴리즈 했고, 실험도 해보았음->3개의 프리트레인 LM으로 실험했는데  
텍스트 정의 같은 추가정보 입력이 guess에 의미있는 결과를 보임  
즉, 이런 데이터셋 활용성이 좋았음  


# Entity Cloze by Date  
LM이 더 넓은 엔티티지식&아예 처음 본 엔티티 추론적 테스트 원함  
그래서 entity cloze문장에 date indexing과 다양 문장들 넣음  

date indexing은 pre-train corpus유무를 확인하고  
다양 문장 삽입은 엔티티 지식이 다양한 형태라 넣는 것  
(mask길이, 다양 문법적 카테고리-pos같은)  

# Task Definition  
용어 정의 - e는 엔티티, s는 문장, me는 entity mention span, mq는 masked query span, my는 gold masked span(predicted goal)  
평가척도는 perplexity  

#Data Collection  
3단계 data 수집 과정  
-엔티티 채굴  
-문장 수집  
-기간 선택  
(eng wikipedia&wikidata 사용)  


#ODIE Mining  
위키피디아 엔티티와 시작시간, 발표날짜, 발견 또는 발전시간, inception time, point시간 또는 날짜 갖고 이중 첫번째 것(시간) 사용  
다른 것들과 비교 위해 유명한 엔티티 사용  


# entity Sentence Collection  
엔티티 관련 영문위키피디아 문서 수집  
(stub의 사소한 문장 제외, 500단어 미만 제외, 첫문단(정의,축약) 제외)  
entity 이름 또는 동의어 있는 분장을 수집  
(인용제외-중첩되서 이상해질 수 있어서, 5단어 미만 제외)  


# Span Selection  
s중에서 길이 mq로 마스킹  
(a) 오버래핑x (엔티티멘션 스팬 me와)  
(b) me 뒤에 위치  
(c) 10단어 이내에서 (me이후) 시작되야  


2종류 span  
-NP span: spaCy 사용하여 span얻음(관계 지식 사용 KB triple)  
-Random span: 샘플링 1-5 중 not overlap& entity당 100span limit으로  


# Span seneitivity to entity knowledge  
실험 디자인 잘 되었나 체크 위한 실험  
(특히 span 유효성)  
span 유효시 mask->PP 상승  
(POPULAR entity 사용)  

entity remove 32.2% 하락, span mask 35.9% 하락  


# Dataset Statistics  
표1에 통계 있음  
people 카테고리만 없는데  
birth와 prominent 사이 gap이 너무 커서임  

voca 더 많고(19Kvs2K)  
data split함(dev와 test)  
이전 date가 entity 더 많음  
2021껀 없음  
NP&random 동일하게(NP상 30% proper하다고 나옴)  


# Experiments   
T5, BART, GPT-Neo로 평가(zero-shot세팅된)  
문장->3가지 버전  
*NoEnt: entity mention 부분을 글자 "the entity"로  
*RANDOM DEF: 정의 관련 부분을 랜덤으로 채워넣음  
*DEFINITION: 정의 관련 첫문장으로 채워넣음  


연도별로 나눠서 평가  
2020, 2021 T5, BART에 없음, GPT-Neo에도 거의 없음  
근데 트레이닝 Data에 2020 영어위키 있긴 함  

실험서 20,21->unseen  
17,18,19->seen  

# Evaluation Metric  
평가척도는 정규화된 PP사용  

 PP계산 달리함  
L-R인 GPT-Neo는 left context만 span으로  
나머지 둘은 right도 포함  
그래서 비교는 좀 애매함  

T5, BART는 single mask  
디코딩서 BART의 left PP계산  
T5서 특수토큰 계산  


# Result  
PP가 year마다 민감. 상대 성능 봄  
NoEnt가 ORI보다 성능 떨어짐  
mask가 context에 민감. 그렇다고 entity에 놀리지 꼭 들어가야한다는 걸 의미하는 건 아님  


DEF가 ORI보다는 나음  
지식 "더" 있는 건 Better  

RANDOM DEF가 T5D서 좀 향상 -> 이유는 model이 다른 position modeling이라서??  

# Performance on unseen entities  
20-21 ->unseen, 17~19->seen 이었는데  
3LM다 더 좋은 성능 보임을 알 수 있었음  

퍼포먼스델타 차이 조사했는데  ORI와 DEF사이서  
->20-21이 더 나음  Def가 더 유용(unseen에서)  


퍼포먼스델타 POPULAR가 20-21보다 나음  
LM이 사전지식 갖고 있고, 새거나 드문거에서도 이거 이용한다고 생각됨  
어떻게? 차후 연구해야할 듯  


# Use Cases  
이 데이터셋이 실생활에 더 유용(유통기한 지난 옛것보다)  
update쉽고 time indexed여서 평가 쉬움  
지속 학습도 좋음  
기존->한가지 fact  , robust-넓은 범위 새것 못하는데  
논문제안-> 함  


# Related Work  
프리트레인LM과 실현실 지식과 mismatch므로 좋은 연구거리임  
*Lazaridou: 코퍼스 PP로 LM넘어서는 것의 경우 성능 떨어짐  
*Dhingro: TEMPORAL LAMA, 시간의존적(valid주제, 관계, object)  
*SITUATE QA: 시간의존 QA예시를 일시정보 entity로 학습  
*ECBD: 뉴엔티티 주목(학습 때 없던)  
*TemporalWiki: 새 fact/entity인용(영어 wiki와 비교해서),  실생활 반영은 잘 안 됨  
*ECBD: 오리지네이트 타임기반 채집-실생활 시간 기반  
*나머지는 diachronic 임베딩 주목: 시간에 따른 의미변화 모델링.  
본 연구에서는 새거에 집중 but ECBD와 비슷한 채집 가짐  


비록 범용 cloze format 썼지만 orthogonal에 집중함(children book & LAMBADA) fiction, not real world에  


# Conclusion  
데이터셋 제시(언어모델 넘어 서는 것도 이해하는 걸 평가하는 엔티티(시간이 가미된))  
43K cloze sent 수집(timerlqks) -> unseen handling은 여전히 challenging 함  






