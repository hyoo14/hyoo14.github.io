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
샐프챗 ACUTE-EVAL 씀   
몰두성 질문 & 140페어 비교 실험(full human 용 옥석고르기)  


최소 빔 길이 조절 실험도 함  
또 최적 길이 예측 모델도 사용  

둘 다 17%, 19% 각각 향상  


빔블락(성능 높이는 것으로 알려진)  
*크게 의미는 없었음  
*full block의 경우 top-K샘프(k=40)이고 샘플 20으로 실험  
**스윗스팟은 10일 때  
**그러나 크게 성능에 영향은 안 줌  
더 작고 집중한 BST파인튜닝 시 성능이 의미있게 향상됨(60%, 40% 향상)  
(페르소나, 지식, 공감, 대화지점에 포커스)  
페르소나가 없는 것 보다 훨씬 성능 좋음(54%), 그래서 페르소나 계속 사용  


# 5.3 Full (Human-Bot Chat) Evaluations  
사람-봇 대화 데이터 Adiwardena 따라 수집  
*open-ended chat '하이'로 시작(사람이)  
*최소 14턴 왔다갔따함  
*100개 대화 수집(crowd worker로부터)  
*Fig1은 체리픽 결과로 (Gen BST2.7B 모델임)  


# Overall ranking models   
전체 랭킹 모델들  


*인간-인간 챗로그 Adiwardana & 봇-인간 Meena log 사용  
*결과  
**파인튜닝 BST(2.7B) > 프리트레인만 한 모델  
**빔서치(min > 20) > 빔서치(no minimum)  (BST2.7B중)  
**BST Gen 큰 거(2.7B)   > 작은 거(90M)  
RetNRef(대화&지식 탐색) 성능향상 별로 없음  
(best decoding scheme 쓸 때)  
*제일 큰 BST GEn 9.4B 모델 몰두성 안 좋음 2.7B 보다  
**대신 PP는 더 좋음(낮음) ( 둘 사이 직관적 관계 없다는 것 나타냄)  
*ACUTE EVAL 몰두성하며 규명함 56%이김(작은 모델이)  
**퓨처워크 -> 결과 이해하는 것, 왜 이런지  

bot사람 대화 데이터 기반 미나봇과 대결의 경우  
*BST gen 2.7B가 미나 이김(몰두성 75%, 인간적임 65%)  
**미나는 인간적임이 몰두성 보다 높은 경향이 있었음  



# Response Length  
대답길이  


BST(2.7B) 평균 21토큰(20길이 제한일 때)  
제한 없는 경우는 9.5  

미나는 10.4, 사람은 18  

인과관계:  
평균 응답 길이(몰두성) 대답길이와 비례(관련됨)  



# 5.4 Failure Cases and Model Extensions  
실폐 사례와 모델 확장  


*ACUTE-Eval 성능 - 처음 볼 땐 49:51 비로 사람에 근접 -> 거의 오픈도매인 챗봇 문제 푼 것  
**그러나 문제점들 있고 왜 캡처하지 못 했는지 설명해보고 관련 대화 단락 제시하겠음  


# Vocabulary Usage  
단어 사용  
Gen 모델 빔서치 디코딩 -> 일반어 넘 많이 나타나고 레어단어 잘 안 씀  
(사람이 사용하는 분포와는 달랐음  )   
->기술적으로는 맞지만 몰두성이 떨어짐  
(시도)  
*샘플링 써서 likelihood 도움 되지만 문맥이 이상해지는 단점 있음  
*최소 길이 제한의 경우 답 풍부하게 하지만 여전히 일반단어에 대한 빈도가 너무 많음  
*적은 문답에서 성능 높이는 방법으로  
**unlikelihood 트레이닝 적용 -> engagingness(ACUTE-Eval) 악영향  
->humaness 올려줌(tradeoff)  


# Other issues  
다른 문제들  
*반복  
**빔블락킹으로 어느정도 방지->근데 파트너 카피함->몰두성 높아지긴 함  
**unlikelihood training으로 반복 줄임  
**페르소나 추가함  
*모순성  
**파트너 말 까먹음 -> 질문 안 해서 논리적 연결 끊김으로 생각됨  
***솔루션들 제시되고 있지만 완전히 해결 못 함  
*사실 오류는 적음  
**평가의 자연 섭리 때문일 듯..(얕은 토픽, 깊지 않은 대화로 평가)  
***토픽이 진중할 경우 약해짐  
*토픽 자꾸 바꾸는 문제  
**ConvAI2 데이터셋의 문제  
***위자드 위키가 해결책이 됨  
*ACUTE-Eval 성능 하락 문제  
**위키 지식 읽을 경우  
***깊은 지식이 별로 필요 없을 수 있기 때문  
***필요없을 때에도 지식을 사용하거나 잘못된 지식을 사용하기 때문  
*진정한 모든 도메인 대화 챗봇은 지식 효과적으로 쓸 수 있어야 하고,  
우리가 잘 평가할 수도 있어야 함!  


# Conversation Length and Memonry  
대화길이와 메모리  
*평가는 14턴 원샷 평가임(매우 짧음)  
**반복과 망각 충분히 노출되지 않음  
*128BPE 토큰으로 제한  
**확장 불가  
*최근 확장(긴 컨텍스트) 있었으나 이거 구현 안 했고 평가도 이것처럼은 못 함  


# Further Notes on Evaluation  
평가에서 추가 참고사항  
*100이상 28턴 이상에서 성능 하락  
*작은버전->통계적 의미 없음  
*길거나, 한번은 평가되어야 인간이 몰두할 만해 짐  
*지도 주기(위자드 위키 랜덤 픽 토픽)  

(사람과 비교)  
*사람-사람(크라우드) (BST paper제시)와 비교해봄  
*본 논문 BST2.7B가 ACUTE-Eval서 56%대 44%로 이김  
*employee chat 보단 약함 49:51  
*crowd가 몰두성서 56:44로 이김  
*crowd가 인간다움에서 59:41로 이김  
*crowd worker가 바로미터로 괜찮음  
*gap 줄이기 위해 다른 방법으로 매칭(일꾼) 또는 다른 셋업 또는 사전지시가 대안임  


# 6 Discussion  
논의
*어떤 방법 같이 쓰는 것이 좋을지 연구  
(scale, fine-tune, decode)  
blend together->SOTA일 때..  


*본 모델이 사람들의 평가에서 몰두성, 인간적임에 높은 평가 받음  
*약점  
**모순과 반복  
**같은 구를 다른 대화에서 반복  
**환각지식?(다른 생성모델처럼)  
***future work- unlikelihood, 지식에 조건 주기 해봤으나 불충분  
*긴 대화의 경우도 성능 떨어짐  
**트랜스포머의 한계로 대화를 기록하고 긴 메모리 기억하게 최근에 발전하는 중  
*풍성하게 하는 방향은 평가가 어렵고 긴 대화 수집의 어려움이 있음  
**대안:  
***토픽에대한 대화 유도  
***인간에게 지식 어떤 거에 맞추라고, 잘 평가하게 함  
*모델 측면:  
**봇이 긴대화->컨텍스트 골라야함  
*페르소나와 토픽(BST에서 주어진)- 대화 충분히 흥미롭게 해줌  
**하지만 더 디테일 해야함(긴경우&반복된 경우)  
***반복 피해야하고 평가 달라져야 함  
*챗봇 사용 위해 더 인간보다 잘 행동해야 하지만 편견 많아 보임  
*인간 평가와 PP사이 관련성 있따는데(디코딩 스킴 고정 시)  
*고려점 더 있음  
**트레이닝 데이터 pushshift.io Reddit vs BST  
**디코딩알고리즘(같은 PP에서)  
***2.7B 모델 > 90M gain 큼  
***9.4B, 2.7B -> 인간평가서 비슷, 2.7B PP는 더 낮음에도  
***대화 경쟁력은 PP가 낮다고 좋은 거 아님  
***PP가 낮아지면 디코딩 타임은 조금 향상됨  
****이것들 이해가 향후 중요 발전 방향이 될 듯  



  
