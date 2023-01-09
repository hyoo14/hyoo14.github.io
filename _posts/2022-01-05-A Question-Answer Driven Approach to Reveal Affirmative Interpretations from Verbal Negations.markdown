---
layout: post
title:  "A Question-Answer Driven Approach to Reveal Affirmative Interpretations from Verbal Negations"
date:   2023-01-05 13:34:19 +0900
categories: study
---





{% highlight ruby %}
짧은 요약 :  

QA 접근법으로 부정형 동사를 사용하는 문장을 같은 의미를 지녔으나 긍정형 동사를 사용하는 문장으로 바꿔주는 모델 제안  

-새코퍼스는 4,472개의 동사로 구성  
-67.1%는 실제 나열  
-태그하는 사람들이 7,277개 답변, 이 중 3,001개가 부정형 동사를 사용  
-NLI 분류의 분제로 풀었으나 기존 코퍼스는 충분하지 않았음  
-파인튜닝해도 성능 상승 적음  
-T5 사용  
-사람의 정확도 따라가지 못 함  

    
{% endhighlight %}


[Paper with my notes](https://drive.google.com/drive/folders/1uKU-bbKbfqbHyUlLX4WyRWn0NeONC0zH?usp=sharing)  


[Lecture link](https://aclanthology.org/2022.findings-naacl.37.mp4)  


# 단어정리  
*predicate: 서술부(동사 이하 부분)  
rephrase: 바꾸어 말하다  
endeavor: 하려고 노력하다, 애쓰다, 노력, 시도  
spearheaded: 선봉, 진두지휘하다  
bypass: 우회 도로, 우회하다, 건너뛰다  
auxiliary: 보조의, 예비의, 조동사, 보조원  
conjugation: 활용, 동사 활용형  
   

# 1 Introduction  
*부정형은 연산자로 볼 수 있음  
**하지만 긍정형보다 정보 적음  
**이해가 어려울 수도 있음(이중부정 등으로)  
*과거에는 scope and focus 방법으로 접근  
**scope는 부정형이 있는 부분  
**focus는 부정형의 타겟 부분  
**이 방법은 이해에는 좋으나 긍정형 해석을 도출하지 못함  
**부정형 부분에 속하는지, 속하지 않는지만 체크  
*본 논문에서는 QA 방법론을 사용  
**동일한 의미의 긍정형을 만들어줌  
**QA-SRL 사용  
*태그자들은 다음과 같은 방식으로 데이터 만듬  
**태그자들은 질문 만들고 답변 만듬    
**긍정형 동사를 사용하는 답변을 다시 만듬  
*본 논문의 공헌  
**AFIN 코퍼스 생성  
**어떤 predicate가 rephrase(서술부가 긍정형으로 다시 쓰임)  
**긍정형으로 바꾸어서 해석  
**긍정형 변환을 생성문제로 풀고 T5사용(성능은 사람보다는 떨어짐)  


# 2 Related Work  
*긍정형 해석 드러내기 어려움(생성 어려움)  
**이전에는 대체로 scope and focus 찾기 문제로 접근함  
**ML, off-the-shelf semantic parser, NN, PB-FOC등 사용됨  
**최근은 graph 기반으로 구성하고 담화데이터 사용  
**단어기반 NN, 토픽기반 어텐션 NN 사용  
**S&F는 부정형이 뭔지 찾기 쉬우나 긍정형 해석 못 얻음  
*Sarabiet의 연구가 본 논문과 비슷  
**부정형 코퍼스 + 동치의 긍정형 해석   
**약점  
***심플 위키 부정형 문장 사용(세련되지 못 함)  
***제한이 들어갔는데 6-25토큰의 문장 길이만 커버하고 특정 단어는 없음  
***긍정형 해석이 제한적인데 오로지 부정형 서술부만 다시 씀  
**반면 본 논문은 QA기반 접근론으로 다양한 긍정 해석 가능하고 복잡한 것도 처리 가능  
*Jiang 상식 내포 식별 연구  
**if-then rule 적용  
**일반 상식, 지식 차용  
**본 논문은 이와는 다르게 자연적 발생 문장(부정형 포함) 서술부가 문법적으로 부정, fact까지 드러냄  


# 3 A Question-Answer Driven Approach to Collect Affirmative Interpretations  
*QA기반 접근 긍정형 해석 수집  
**AFIN 코퍼스 만듬(부정형과 동치의 긍정형 포함된)  
**코퍼스 원천 앞으로 다룰 것임(논문에서)  
**템플릿 기반 QA 생성으로 긍정형, 부정형 만듬  
**QA에서 어떻게 자연어 긍정형 해석이 나오는지 설명할 예정(논문서)  


# 3.1 Collecting Sentences Containing Negation  
*부정형 수집  
**QA-SRL 뱅크 코퍼스 사용(wiki, wikinews, science textbooks)  
**부정어 체크(not, n't, no, never)  
**두가지만 제한(의문문 제외, 부정형은 동사로)  
**SpaCy로 직, 간접 체크  
**타겟 동삼(부정형) 사용  


# 3.2 Generating and Answering Questions  
*답변 생성  
**전체 질문의 답변으로 긍정형(반대급부인) 만들고 fact인지 체킹  
***맞으면 타겟 value 있나 체크  

# Template-Based Question Generation  
*템플릿 기반 질문 생성  
**선호단어로 질문 만들게시킴  
**일치&빠르게  
**7-slot template technique 사용  
**WH 추가 사용(WH: Who, What, Whom, When, Where, etc / AUX: is, was, does, did, has, had, can, etc / SUB: something, or someone / VERB: full conjugation of the target / OBJ1: something, or someone / PREP: by, to, for, with, about, of, or from / OBJ2: someone, something, somewhere, do, doing, etc /  
*템플릿이 주석자들 질문 생성 다양하게 해줌  
**자연어 긍정형 해석 쉽게 해줌  


# Answering Questions and Assigning Confidence  
*질문응답과 자신감 점수 측정  
**직전 질문에 응답(질문에 있을수도, 상식 또는 지식에 잇는지)  


*점수척도  
**4: 극도로 자신있는(정확, 예: when? not!)  
**3. 매우 자신있는(예: How often? usually do not)  
**2. 보통 자신있는(예: What? Not release)  
**1. 조금 자신있는(예: How explain? not be explaiend by data)   


# Scaling the Annotation Process  
*주석 작업 스케일링  
**웹 이용 자동화  


# Annotation Quality  
*주석 퀄리티는 영어가 모국어인 학부생 5명이 참여하여 체크  
*수동평가(자동의 경우 한계가 있음, 다른 두 문장 같은 질의에 대답이 모두 올바를 수 있음, 이를 놓침)  


# 3.3 Generating Affirmative Interpretations from Questions and Answers  
*QA서 긍정문 생성  
**질문 생성, 답변 생성(주석자에 의한, 룰베이스 기반, 동사시제 수일치 의존적)  


# 4 Corpus Analysis  
*QA 접근 응답 3,001/4,472 (부정형)  
*target verb 당 2.4 질문에 대답(생성)  
*길이는 Q5, A3.5 정도  
**부정형 25.8, 긍정형 11.2  
*극도로 자신있는 85%, 매우 자신있는 까지 포함할 경우 97.7%  
*다음의 특징이 있음(다음 범주의 단어들 포함)  
**인내24%, 매너 23%, 양10%, 시간10%, 이유5%, 대리인8%, 그 외 16%  


# 5 Experiments and Discussion  
*AFIN은 부정형 + 긍정형 해석(부정형과 동치인)으로 구성  


# 5.1 Affirmative Interpretations and Natural Language Inference Classification  
*긍정 해석 & NLI 분류로 target&긍정 사용  
**전제와 가설 in NLI 처럼 매핑  


# Transformers and Existing NLI Benchmarks  
*트랜스포머와 SOTA NLI 벤치마크 기록(AFIN에서)  
**RoBERTa와 XLNet 실험  
**MNLI, SNLI, RTE 파인튜닝  
**트랜스포머 성능 비슷(entailment만 사용하거나 3가지 테그 모두 사용)  
**RTE서 drop이 XLNet이 낮았지만 성능은 오히려 안 좋음(contradiction 없어서 인 것으로 예상)  


# Fine-tuning with AFIN  
*AFIN 파인튜닝 성능 향상 확인  
**근데 향상폭이 아주 크지는 않음  


# 5.2 Generating Affirmative Interpretatinos  
*NLI로 casting 가치 있지만 현실적인 방법은 아님  
**긍정형 체크 하기가 어렵기 때문  
**부정형으로 긍정형 생성 시도  
**AFIN 70% 학습, 15% 개발용, 15% 테스트용  


# Experimental Setup  
*T5 large 트랜스포머로 실험  
**input: 부정형 동사 포함된 문장 또는 문장+target  


# Results and Analysis  
*BLEU-2, chrf++, METEOR로 평가  
*사람과 T5로 테스트  
*T5의 유용성 확인  
**그러나 부분적이고 수동 평가가 필요  
**target을 합친 경우가 더 효과가 좋음 확인  
**T5의 한계도 확인  


# Qualitative Analysis  
*질적 분석  
**T5한계: 랜덤 150인스턴스로 실험  
**T5로 certainly true 학습이 쉬웠음  
**T5가 패턴을 학습했기 때문  
**T5가 부정어를 단순히 지우기도 함  


# 6 Conclusion  
*QA 기반 접근으로 긍정형 해석 드러냄(부정형과 같은 의미의)  
*주석자가 질의 및 응답 생성(부정형과 의미는 같은 긍정형)  
*67.1% 부정형 이벤트 fact로 판정  
*많은 카테고리가 긍정형으로 바뀜(인내, 매너 등..)  
*NLI로 cast해도 트랜스포머가 해석하는 것 확인  
**하지만 아주 적절하지는 않음  
**제한적 성공이긴 했음  
*긍정형 생성이 사실적, 현실적  
**상식, 세계지식 등을 조합하는 것이 SOTA 를 뛰어넘는 모델이 되게하는 길이라고 예상  














