---
layout: post
title:  "Recipes for Building an Open-Domain Chatbot"
date:   2022-08-22 13:51:19 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 :  

오픈도메인 챗봇 여전히 도전적이  

기존->스케일링(데이터의 파라미터, 서비스)  

여기선->blended skilss 사용  
(-providing engaging talking point, 지식,공간,개인적 적절성 보여주도록 함)  


[https://drive.google.com/drive/folders/1enQBlhrvU9l1Rpyvropi-LVmXXPfncGr?usp=sharing]


{% endhighlight %}


#단어정리  
*takeaways: 시사점, engagingness: 몰입도, utterance: 발언, bland: 부드러운, spicy: 매운, interrogate: 정보를 얻다, startify: 계층화하다,   
gauge: 판단하다,측정하다, whittle: 깎다,줄이다, nowhere near as close: 멀었다, elucidate: 설명하다, snippeet: 토막,정보, conducted: 수행되다,  
albeit: 비록 ~일지라도, further notes: 추가참고사항, annotator: 주석자, avenue: 수단, hallucinate: 환각, seed: 유포, mitigate: 완화, 

# 1. Introduction  
서론  
*오픈 도메인 챗봇 만드는 레시피(방법론) 제공  
*어떤 방법 조합이 좋은 성능 낼지 보여줌  
(NLP & 대화에이전트는 큰 코퍼스(코퍼라)가 중요)  
*여기선  
**파인튜닝 중 개인성, 몰두성, 지식, 공간에 포커스  
***BlendedSkillTask(BST) set-up으로 트레이닝 데이터와 초기 대화컨텍스트에 이러한 특성 제공  
**PP비슷한 두 모델이라도 어떤 decoding 알고리즘 쓰느냐에 따라 성능은 달라짐  
***빔서치 성능 별로였고 샘플링이 더 좋았음->파라미터(빔 길이같은 것) 조정해서 영향 끼침->bland vs spicy 스펙트럼 영향  
*본 모델이 DialogGPT, Meena챗봇 압도함  
**75%, 25% engaging  
**65%, 35% humanness  
***더 나음 본 모델이  
*성능 좋지만 한계는 있음  
**깊은 지식 부족  
**간단한 언어 선호  
**자주 쓰는 말 반복  
*unlikelihood학습과 검색, 정제로 극복시도했으나 성공은 못 함  
**미래에 이 문제 경감 위해 다루어 보겠음  
*릴리즈된 모델이 잘 활용되게 잘 공개함  
*큰규모 SOTA대화 에이전트(코드-파인튜닝, 모델웨이트, 평가코드) 다 제공해줌  


# 2. Related Work  
관련 연구  
*오픈도메인챗봇 상당 발전해왔고 관련 연구들은 아래와 같음  
*ConvAI2 대회(데이터셋) -> 큰 프리트레인 트랜스포머 계열이 성능 앞섬  
**Books Corpus사용, PP&F1 최상 -> 더 크고 프리트레인 향상으로 성능 향상  
*다양한 데이터셋 등장(daily diologue & cornell movies) -> 본 논문도 다양 Data 멀티테스킹함  
*Meena와 DialogGPT와 비교함  
**Meena는 2.6B 파라미터 가진 트랜스포머 모델로 341GB text로 학습했고 DialogGPT는 345B 파라미터 가졌으며 147M Reddit discussion thread 멀티대화를 학습  
*SSA서 미나 평가척도는 평균 sensibleness와 specificity  
**사람 평가에 의했고 정적 or 상호작용 써서 얼마나 사람같은지 물어봄  
***하지만 empathy 평가 못함  
***test set 공개가 덜 되서 비교도 어려움  
***반면 DialogGPT는 릴리즈 되어서 평가가 쉬움  
*편견없는 크라우두워커 고용해서 실험함  
**ACUTE-EVAL 평가척도로 사용  
**인간다움, 몰두성 평가함(사람이 얼마나 봇에 흥미가 있는지-몰두성)  


# 3 Models, Training, and data  
# 3.1 Architectures  
3가지 아키텍처로 구성  
*탐색  
*생성  
*탐색&정제(트랜스포머기반 모델)  

# Retriever  
탐색  
*대화기록 -> 다음 발화(점수 매겨서 최고점 씀, 후보는 train set에 있음)  
*폴리인코더 구조 사용(두개인데 각각 256M, 622M 파라미터 갖고 N = 64 codes임)  

# Generator  
생성  
생성모델로 본 Seq2Seq 트랜스포머 씀  
*ParlAI 기반  
*Byte-Level BPE 노크나씀(허깅페이스 토크나이저 기반으로 구현&학습)  
*3가지 사이즈있음  
**90M 파라미터 - Shuster 따름  
**2.7B 파라미터 - Adiwardana 따름, 2인코더/24디코더, 256차원, 32어텐션  
**9.4B 파라미터 - 4layer, 32 디코디레이어, 4096임베딩 차원, 32어텐션  


# Retrieve and Refine  
생성모델이 멍청하고 반복하는 문제가 있고  
이를 해결해가 위해 (스케일링으로는 부족해서) 사용하는 기법(모델)  


탐색 먼저함(생성 전)->탐색&정제 모델임  
*1. 대화탐색 - 탐색기반 대화모델 사용 -> 응답 -> 같이 input에 넣음 -> 컨텍스트 가미된 input이 되는 것  
*2. 지식탐색 - KB 탐색을 컨디션으로 씀  
1, 2 합쳐서 Wiz 생성기에 다 넣어서 -> 랭킹메길 때 사용  
*트랜스포머 기반 분류기 학습시킴 -> 탐색할지 말지 정하도록(탐색 불필요한 경우 있으므로)  
(참고로 본 논문 다른 모델들은 조건부 지식 검색 안 씀)  


# 3.2 Training Objectives  
학습목적, 목적함수  
탐색모델 - Humeauet  
생성모델 - MLE  
탐색&정제 - alpha blending -> 탐색발화무시방지 위해  
실패(모델생성) 처리 -> unlikelihood loss test, 너무 많이 나오는 n-gram 구문에 패널티  


# 3.3 Decoding  
디코딩  
*빔서치(빔 크기 달리하면서)  
*top-K 샘플링  
*샘플 & 랭크  


(실험)  
*최소 길이 제약 - 끝 토큰 생성 금지(최소 길이 이전에는)  
*길이 예측 접근 - 4개 반환길이 중 하나 예측(분류기 만들어서 top뽑음)  
*subsequence blocking - 스탠다드 빔 블로킹(n-gram) 통해서(n=3)  
(생성 발화, 이전 발화자의 반환 input sequence 모두 고려)  


# 3.4 Training Data  
학습 데이터  
영어 사용  
*pre training - Reddit 토의 씀(Humeauet 따름)  
(3rd party로 획득, pushshift.io에서 이용 가능)  
*Dataset 필터링 휴리스틱하게.. Appendix 참고  
*데이터셋 (150B 코멘트, 56.8B 토큰(BPE), 88.8B context 토큰, 코퍼스-4096chunks로 동등나눔, ID로 구분(같은 post코멘트가 안 나뉘게), 마지막 2개는 test와 valid용으로 전체의 0.02%비율임)  
*파인튜닝 (작지만 집중된 연구용 사용, display desirable conversational traits 사용, ConvAI2(개인성, 집중성), EmpatheticDillogues(공감), Wizard of Wiki(지식), BlendSkillTest(위 3개 잘 섞어서), 기타 나쁜말 탐지 분류기-얼마나 자주 나오나 좀)  


# 4. Evaluation Methods  
평가방법  
ACUTE-Eval procedure로 평가  
*1. Engagingness - 누구와 지속 대화 할래?  
*2. Humanness - 누가 더 사람 같은지?  

사람 평가는 시간과 비용이 많으 듦  
그래서 스스로대화 시킴  
*BST 세팅(ACUTE-EVAL 비용 크니까 대체재로)  
**페르소나  
**토픽  
**이전발화  
*위 세가자ㅣ 줌  
(빔 디코딩 시 같은 대화 반복 피해줌  



# 5. Results & Analysis  
결과분석  
# 5.1 Automatic Evaluations  
자동 분석  
# Retriever  
탐색모델  
ConvAI2, Wizard of wiki, 공감대화 ,BST로 파인튜닝  
*hit@1/K로 평가(validation set에서)  


# Generator  
생성모델 평가  
90M, 2.7B, 9.4B 파리미터 모델들(valid set: pushshift.io Reddit 사용)  
파인튜닝 후 PP성능 향상  
탐색&정제 후 PP 조금 향상  

# Safety  
안정성  

BST파인튜닝이 더 safe한 응답을 함  
(다른 프리트레인 모델 보다)  
*분류기 or 휴리스틱 작업 했기 때문  
*본 모델은 ConvAI2 학습이 더 적었고 unsafe인 경우도 적었음    
**제시하는 모델의 파인튜닝이 unsafe가 덜 있는 이유    



# 5.2 Self-Chat Evaluations  
자가실험  




  
