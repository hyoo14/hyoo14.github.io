---
layout: post
title:  "[2024]Decoding the Narratives: Analyzing Personal Drug Experiences Shared on Reddit"  
date:   2024-11-20 19:42:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

챗지피티로... 다중 라벨 분류...하도록 프로세스(파이프라인) 만들어서 좋은 결과...ㅎㅎ  

짧은 요약(Abstract) :    



이 연구는 Reddit과 같은 온라인 커뮤니티에서 약물 사용자(PWUD)가 공유한 경험을 분석하기 위한 새로운 다층적 다중 라벨 분류 모델을 개발합니다. 연구자들은 게시물의 의도(Inquisition 또는 Disclosure), 주제(회복, 의존 등), 구체적인 목표(재발, 안전성 등)를 평가하는 새로운 분류 체계를 소개했습니다. GPT-4 모델은 다른 모델을 능가하며, 사용자 생성 텍스트를 정확하게 분류하는 데 우수한 성능을 보였습니다. 분석 결과, "안전성", "물질 조합", "정신 건강"에 대한 논의가 개인적인 경험과 피해 감소 노력을 중점으로 하고, "생리학적 효과"는 피해 감소에 초점을 맞추는 경향이 있음을 발견했습니다. 이 연구는 약물 사용자 경험에 대한 이해를 증진하며, 약물 사용 장애(SUD) 및 약물 사용에 대한 지식 기반을 풍부하게 합니다.

---


This study develops a novel multi-level, multi-label classification model to analyze drug use experiences shared on platforms like Reddit. Researchers introduced a taxonomy to classify posts based on intent (Inquisition or Disclosure), subject (e.g., Recovery, Dependency), and specific objectives (e.g., Relapse, Safety). GPT-4 outperformed other models in accurately classifying user-generated text. The analysis revealed that discussions about "Safety," "Combination of Substances," and "Mental Health" often focus on personal experiences and harm reduction, while "Physiological Effects" discussions emphasize harm reduction. This work enhances understanding of drug user experiences and enriches the broader knowledge base on Substance Use Disorder (SUD) and drug use.



* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  







 
<br/>
# Methodology    




이 연구의 방법(GPT-4 기반 모델)은 기존의 머신러닝 및 트랜스포머 기반 모델과 비교하여 뛰어난 성능을 보였습니다. 주요 결과와 테스트 세부 사항은 다음과 같습니다:

#### 결과 비교
1. **GPT-4 모델 성능**:
   - 세 가지 레벨(연결, 주제, 목표) 모두에서 GPT-4가 가장 높은 F1 점수(0.91)를 기록했습니다.
   - 특히 `Instruction + Definition + Example`(I + D + E) 방식의 프롬프트에서 가장 우수한 성능을 보였습니다.

2. **기타 모델과의 비교**:
   - DeBERTa 모델: 연결 수준에서는 정밀도 0.95로 높은 성능을 보였으나, 전체적인 F1 점수에서는 GPT-4에 미치지 못함.
   - 전통적인 머신러닝 모델(Logistic Regression, SVM 등)은 F1 점수가 0.75 이하로 상대적으로 낮음.

#### 테스트 메트릭
- **평가 지표**:
  - 정밀도(Precision), 재현율(Recall), F1 점수(F1 Score)로 모델 성능 평가.
  - 다중 레이블 분류에서 각 레벨별로 평균 가중치 점수 사용.

#### 데이터셋
- **데이터 출처**:
  - Reddit의 약물 관련 서브레딧(r/opiates, r/benzodiazepines, r/stims, r/cocaine)에서 2017~2022년 사이 게시된 게시물.
  - 총 267,748개의 게시물 중 1,600개를 랜덤 샘플링하여 분류 체계 개발 및 테스트에 사용.
- **훈련 및 테스트 세트**:
  - 500개의 주석된 게시물을 훈련(400개)과 테스트(100개)에 사용.
  - 최종적으로 1,000개의 추가 게시물에 모델을 적용하여 결과 분석 수행.

---


The proposed GPT-4-based method demonstrated superior performance compared to traditional machine learning and transformer-based models. Key results and test details are as follows:

#### Result Comparison
1. **Performance of GPT-4 Model**:
   - Achieved the highest F1 score (0.91) across all three levels (Connection, Subject, Objective).
   - Particularly excelled when using `Instruction + Definition + Example` (I + D + E) prompting style.

2. **Comparison with Other Models**:
   - DeBERTa: Performed well in the Connection level with a precision of 0.95 but fell short in overall F1 scores compared to GPT-4.
   - Traditional ML models (e.g., Logistic Regression, SVM) showed relatively lower F1 scores, below 0.75.

#### Evaluation Metrics
- **Metrics Used**:
  - Precision, Recall, and F1 Score were utilized to evaluate model performance.
  - Weighted average scores were calculated for multi-label classification across levels.

#### Dataset
- **Data Source**:
  - Posts from drug-related subreddits (r/opiates, r/benzodiazepines, r/stims, r/cocaine) on Reddit, dated between 2017 and 2022.
  - Out of 267,748 posts, 1,600 were randomly sampled for taxonomy development and testing.
- **Training and Test Sets**:
  - 500 annotated posts were used for training (400 posts) and testing (100 posts).
  - An additional 1,000 posts were analyzed using the best-performing model for further insights.


   
 
<br/>
# Results  


이 연구의 방법(GPT-4 기반 모델)은 기존의 머신러닝 및 트랜스포머 기반 모델과 비교하여 뛰어난 성능을 보였습니다. 주요 결과와 테스트 세부 사항은 다음과 같습니다:

#### 결과 비교
1. **GPT-4 모델 성능**:
   - 세 가지 레벨(연결, 주제, 목표) 모두에서 GPT-4가 가장 높은 F1 점수(0.91)를 기록했습니다.
   - 특히 `Instruction + Definition + Example`(I + D + E) 방식의 프롬프트에서 가장 우수한 성능을 보였습니다.

2. **기타 모델과의 비교**:
   - DeBERTa 모델: 연결 수준에서는 정밀도 0.95로 높은 성능을 보였으나, 전체적인 F1 점수에서는 GPT-4에 미치지 못함.
   - 전통적인 머신러닝 모델(Logistic Regression, SVM 등)은 F1 점수가 0.75 이하로 상대적으로 낮음.

#### 테스트 메트릭
- **평가 지표**:
  - 정밀도(Precision), 재현율(Recall), F1 점수(F1 Score)로 모델 성능 평가.
  - 다중 레이블 분류에서 각 레벨별로 평균 가중치 점수 사용.

#### 데이터셋
- **데이터 출처**:
  - Reddit의 약물 관련 서브레딧(r/opiates, r/benzodiazepines, r/stims, r/cocaine)에서 2017~2022년 사이 게시된 게시물.
  - 총 267,748개의 게시물 중 1,600개를 랜덤 샘플링하여 분류 체계 개발 및 테스트에 사용.
- **훈련 및 테스트 세트**:
  - 500개의 주석된 게시물을 훈련(400개)과 테스트(100개)에 사용.
  - 최종적으로 1,000개의 추가 게시물에 모델을 적용하여 결과 분석 수행.

---


The proposed GPT-4-based method demonstrated superior performance compared to traditional machine learning and transformer-based models. Key results and test details are as follows:

#### Result Comparison
1. **Performance of GPT-4 Model**:
   - Achieved the highest F1 score (0.91) across all three levels (Connection, Subject, Objective).
   - Particularly excelled when using `Instruction + Definition + Example` (I + D + E) prompting style.

2. **Comparison with Other Models**:
   - DeBERTa: Performed well in the Connection level with a precision of 0.95 but fell short in overall F1 scores compared to GPT-4.
   - Traditional ML models (e.g., Logistic Regression, SVM) showed relatively lower F1 scores, below 0.75.

#### Evaluation Metrics
- **Metrics Used**:
  - Precision, Recall, and F1 Score were utilized to evaluate model performance.
  - Weighted average scores were calculated for multi-label classification across levels.

#### Dataset
- **Data Source**:
  - Posts from drug-related subreddits (r/opiates, r/benzodiazepines, r/stims, r/cocaine) on Reddit, dated between 2017 and 2022.
  - Out of 267,748 posts, 1,600 were randomly sampled for taxonomy development and testing.
- **Training and Test Sets**:
  - 500 annotated posts were used for training (400 posts) and testing (100 posts).
  - An additional 1,000 posts were analyzed using the best-performing model for further insights.




<br/>
# 예제  




이 연구에서 GPT-4 모델은 경쟁 모델들보다 더 정밀하고 구체적인 결과를 제공했습니다. 테스트 사례와 이를 처리한 방법을 통해 경쟁 모델과의 차이를 설명합니다.

#### 테스트 사례
- 게시물 예시 1 (연결 수준, Connection Level):
  - **게시물 내용**: "나는 이 약물을 처음 사용해 보려 하는데, 이것이 안전한 방법인지 알고 싶습니다."
  - **실제 라벨**: Inquisition (정보 요청)

##### 처리 결과
1. **GPT-4의 처리 결과**:
   - 라벨: Inquisition
   - 이유: 질문 구조와 정보 요청을 나타내는 단어("안전한 방법", "알고 싶습니다")를 정확히 분석.
   - 정확도 높은 결과 제공.

2. **DeBERTa 모델의 결과**:
   - 라벨: Disclosure
   - 이유: 질문 내용 대신 약물 관련 "경험 공유" 단어에 과대 반응, 잘못된 라벨 지정.

---

- 게시물 예시 2 (목표 수준, Objective Level):
  - **게시물 내용**: "이 약물 조합이 안전한지 알고 싶습니다. 복용 방법도 알려주세요."
  - **실제 라벨**: Safety, Methods of Ingestion

##### 처리 결과
1. **GPT-4의 처리 결과**:
   - 라벨: Safety, Methods of Ingestion
   - 이유: 다중 라벨을 정확히 인식하고, "안전", "복용 방법"이라는 구체적 목표를 정확히 분류.
   - I + D + E 프롬프트를 통해 세부적인 문맥 파악.

2. **DeBERTa 모델의 결과**:
   - 라벨: Safety
   - 이유: 주요 단어("안전")는 인식했으나, "복용 방법"이라는 추가 정보를 놓침.
   - 다중 라벨 처리 능력 부족.

#### 결론
GPT-4 모델은 문맥과 세부적인 의미를 더 잘 이해하여, 다층적 분류 체계에서 경쟁 모델보다 뛰어난 성능을 보여줍니다. 특히 복잡한 게시물에서도 다중 라벨을 정확히 처리할 수 있었습니다.

---



In this study, the GPT-4 model provided more precise and detailed results than its competitors. Test cases demonstrate how it handles data better than other models.

#### Test Case
- **Example 1 (Connection Level)**:
  - **Post Content**: "I'm trying this drug for the first time and want to know if this is a safe method."
  - **True Label**: Inquisition (Information-seeking)

##### Results
1. **GPT-4's Result**:
   - Label: Inquisition
   - Reason: Correctly identified question structure and key phrases ("safe method," "want to know"), providing accurate classification.

2. **DeBERTa's Result**:
   - Label: Disclosure
   - Reason: Misclassified due to overemphasis on "experience-sharing" words rather than recognizing the question intent.

---

- **Example 2 (Objective Level)**:
  - **Post Content**: "I want to know if this drug combination is safe. Please tell me the method of ingestion."
  - **True Label**: Safety, Methods of Ingestion

##### Results
1. **GPT-4's Result**:
   - Label: Safety, Methods of Ingestion
   - Reason: Accurately identified multiple labels, capturing both "safety" and "method of ingestion" objectives through nuanced context understanding with I + D + E prompts.

2. **DeBERTa's Result**:
   - Label: Safety
   - Reason: Identified the primary term "safety" but failed to recognize additional context about ingestion methods, highlighting its limitations in multi-label tasks.

#### Conclusion
The GPT-4 model outperformed competitors by better understanding context and capturing nuanced meanings, particularly in complex posts requiring multi-label classifications.


<br/>  
# 요약   



이 연구는 Reddit 게시물을 분석하기 위해 새로운 다층적 분류 체계를 도입했으며, GPT-4 모델이 이를 통해 최고 성능(F1 점수 0.91)을 기록했습니다. 특히, `Instruction + Definition + Example`(I + D + E) 프롬프트 방식을 활용해 다중 라벨 분류에서도 뛰어난 정확도를 보였습니다. 테스트 사례에서 GPT-4는 "안전성"과 "복용 방법" 같은 세부 목표를 정확히 분류해 경쟁 모델을 능가했습니다. DeBERTa 등 다른 모델은 특정 키워드에 의존하여 다중 라벨을 인식하지 못했습니다. GPT-4는 문맥과 세부 의미를 잘 이해하여 약물 사용자 경험의 분석에 적합성을 입증했습니다.

---


This study introduced a novel multi-level taxonomy to analyze Reddit posts, with the GPT-4 model achieving the highest performance (F1 score of 0.91). Using the `Instruction + Definition + Example` (I + D + E) prompting method, GPT-4 excelled in multi-label classification tasks. In test cases, GPT-4 accurately identified specific objectives such as "Safety" and "Methods of Ingestion," outperforming competitors. Models like DeBERTa relied too heavily on keywords, failing to recognize multiple labels. GPT-4 demonstrated superior understanding of context and nuances, proving effective for analyzing drug user experiences.


<br/>  
# 기타  


<br/>
# refer format:     



@inproceedings{Bouzoubaa2024,
  author    = {Layla Bouzoubaa and Elham Aghakhani and Max Song and Minh Trinh and Rezvaneh Rezapour},
  title     = {Decoding the Narratives: Analyzing Personal Drug Experiences Shared on Reddit},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2024},
  pages     = {6131--6148},
  year      = {2024},
  publisher = {Association for Computational Linguistics},
  address   = {August 11-16, 2024},
  doi       = {},
  url       = {}
}




Bouzoubaa, Layla, Elham Aghakhani, Max Song, Minh Trinh, and Rezvaneh Rezapour. "Decoding the Narratives: Analyzing Personal Drug Experiences Shared on Reddit." Findings of the Association for Computational Linguistics: ACL 2024, August 11–16, 2024, 6131–48. Association for Computational Linguistics.




