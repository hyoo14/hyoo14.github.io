---
layout: post
title:  "[2024]Understanding Student Sentiment on Mental Health Support in Colleges Using Large Language Models"  
date:   2025-01-11 15:56:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    

대학의 정신 건강 지원은 학생들의 복지와 학업 성공에 중요한 역할을 합니다. 하지만 데이터 수집의 어려움과 표준화된 평가 기준의 부족으로 인해 이를 평가하는 연구는 제한적입니다. 본 논문은 대규모 언어 모델(LLM)을 활용하여 학생들이 대학의 정신 건강 지원에 대해 갖는 감정을 분석합니다. 이를 위해 SMILE-College라는 감정 분석 데이터셋을 생성했으며, 인간과 기계가 협력하여 데이터를 주석 처리했습니다. 전통적인 기계 학습 방법과 최신 LLM(GPT-3.5, BERT)을 비교한 결과, GPT-3.5와 BERT가 가장 우수한 성능을 보였습니다. 연구 결과는 감정 예측의 어려움과 LLM이 정신 건강 연구 및 대학 정신 건강 서비스 개선에 기여할 수 있는 방법을 제시합니다. 이러한 데이터 기반 접근법은 효율적인 정신 건강 지원 평가와 관리, 의사결정을 지원합니다.  


Mental health support in colleges is vital for student well-being and academic success. However, evaluating its effectiveness faces challenges due to difficulties in data collection and a lack of standardized metrics. This paper analyzes student sentiments on mental health support in colleges using large language models (LLMs). The SMILE-College dataset was created through human-machine collaboration. A comparison of traditional machine learning methods and state-of-the-art LLMs, such as GPT-3.5 and BERT, revealed that these models performed best. The study highlights challenges in sentiment prediction and offers practical insights into how LLMs can enhance mental health research and improve college mental health services. This data-driven approach facilitates efficient evaluation, management, and decision-making in mental health support.  



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


이 연구에서는 대학생들의 정신 건강 지원에 대한 감정을 분석하기 위해 **SMILE-College 데이터셋**을 생성하고, 이를 기반으로 여러 모델을 비교하였습니다. 주요 방법론은 다음과 같습니다:

1. **사용된 모델**:
   - **GPT-3.5**: 대규모 언어 모델로 가장 우수한 성능(F1 점수 0.80)을 보였으며, 다양한 감정 범주에 대해 균형 잡힌 예측 성능을 제공했습니다.
   - **BERT**: 사전 학습된 언어 모델로, 특히 불만족(Dissatisfied)과 중립(Neutral) 범주에서 높은 성능을 보였습니다(F1 점수 각각 0.86, 0.80).
   - **기타 모델**: Logistic Regression(LR), Support Vector Machine(SVM), Mistral, Orca 2, Llama 2와 같은 다양한 모델도 비교에 포함되었으며, GPT-3.5와 BERT에 비해 상대적으로 낮은 성능을 보였습니다.

2. **특이점**:
   - **혼합 감정(Mixed)**이라는 새로운 감정 범주를 도입하여, 만족과 불만족이 동시에 나타나는 복잡한 감정 표현을 반영했습니다.
   - **사람-기계 협력 주석**: 대규모 언어 모델을 사용하여 초기 감정 분류를 자동화한 후, 인간 주석자가 이를 검토 및 보완하는 과정을 통해 데이터 정확도를 높였습니다.

3. **사용된 데이터셋**:
   - **Student Voice Survey(SVS)** 데이터: College Pulse에서 제공한 학생 설문 데이터를 기반으로 하며, 총 793개의 주석 처리된 텍스트 응답으로 구성되어 있습니다. 이 데이터셋은 대학 정신 건강 지원에 대한 학생들의 경험을 다루며, 만족(Satisfied), 불만족(Dissatisfied), 혼합(Mixed), 중립(Neutral)의 4가지 감정 범주로 주석 처리되었습니다.


---


This study analyzed student sentiments on mental health support in colleges by creating the **SMILE-College dataset** and comparing various models. The main methodology includes:

1. **Models Used**:
   - **GPT-3.5**: A large language model that achieved the highest performance (F1 score: 0.80) with balanced predictions across various sentiment categories.
   - **BERT**: A pre-trained language model that excelled in the Dissatisfied and Neutral categories (F1 scores: 0.86 and 0.80, respectively).
   - **Other Models**: Logistic Regression (LR), Support Vector Machine (SVM), Mistral, Orca 2, and Llama 2 were also included in the comparison but performed relatively lower than GPT-3.5 and BERT.

2. **Key Features**:
   - **Introduction of the Mixed Sentiment Category**: This new category captures complex responses expressing both satisfaction and dissatisfaction simultaneously.
   - **Human-Machine Collaboration**: Large language models were used for initial automated sentiment annotation, which was later refined and validated by human annotators to enhance accuracy.

3. **Dataset Used**:
   - **Student Voice Survey (SVS)** Data: This dataset, provided by College Pulse, consists of 793 annotated text responses. It reflects students' experiences with college mental health support and includes four sentiment categories: Satisfied, Dissatisfied, Mixed, and Neutral.




   
 
<br/>
# Results  




이 연구의 결과는 SMILE-College 데이터셋의 테스트 세트를 기준으로 모델 성능을 비교하여 도출되었습니다. 주요 결과는 다음과 같습니다:

1. **테스트 데이터셋**: 
   - SMILE-College 데이터셋에서 학습, 개발, 테스트 데이터를 각각 75:5:20의 비율로 분리하여 사용하였습니다.
   - 각 감정 범주(만족, 불만족, 혼합, 중립)에 대해 모델의 정밀도(Precision), 재현율(Recall), F1 점수를 평가하였습니다.

2. **모델 성능**:
   - **GPT-3.5**: 전체 F1 점수 0.80으로 가장 우수한 성능을 보였습니다. 이는 만족(F1: 0.76), 불만족(F1: 0.86), 혼합(F1: 0.69), 중립(F1: 0.85) 범주에서 모두 균형 잡힌 결과를 보여줍니다.
   - **BERT**: 전체 F1 점수 0.78로 두 번째로 높은 성능을 보였으며, 특히 불만족(F1: 0.86)과 중립(F1: 0.80) 범주에서 높은 점수를 기록했습니다.
   - **기타 모델**: 
     - 전통적 기계 학습 모델(LR, SVM)은 전체 F1 점수가 각각 0.61, 0.60으로 상대적으로 낮은 성능을 보였습니다.
     - Mistral, Orca 2, Llama 2 등의 LLM은 특정 범주에서 강점을 보였으나 전체적인 F1 점수는 GPT-3.5와 BERT에 미치지 못했습니다.

3. **성능 향상**:
   - GPT-3.5는 LR에 비해 전체 F1 점수에서 약 **31% 향상**을 보였으며, BERT와 비교했을 때도 **2% 향상**을 나타냈습니다.
   - 특히 혼합(Mixed) 범주에서 GPT-3.5는 다른 모델 대비 약 4-16% 높은 F1 점수를 기록하여, 복잡한 감정 표현 분석에 강점을 보였습니다.

---



The results of this study are based on the evaluation of models using the test set from the SMILE-College dataset. Key findings are as follows:

1. **Test Dataset**:
   - The SMILE-College dataset was split into training, development, and test sets in a 75:5:20 ratio.
   - The performance was evaluated across sentiment categories (Satisfied, Dissatisfied, Mixed, Neutral) using metrics such as Precision, Recall, and F1 score.

2. **Model Performance**:
   - **GPT-3.5**: Achieved the highest overall F1 score of 0.80, with balanced performance across categories: Satisfied (F1: 0.76), Dissatisfied (F1: 0.86), Mixed (F1: 0.69), and Neutral (F1: 0.85).
   - **BERT**: Recorded the second-best overall F1 score of 0.78, with strong results in the Dissatisfied (F1: 0.86) and Neutral (F1: 0.80) categories.
   - **Other Models**:
     - Traditional machine learning models (LR, SVM) achieved relatively lower overall F1 scores of 0.61 and 0.60, respectively.
     - LLMs like Mistral, Orca 2, and Llama 2 showed strengths in specific categories but were outperformed by GPT-3.5 and BERT in overall metrics.

3. **Performance Improvement**:
   - GPT-3.5 showed an approximately **31% improvement** in overall F1 score compared to LR and a **2% improvement** over BERT.
   - In the Mixed category, GPT-3.5 outperformed other models by 4-16% in F1 score, demonstrating its strength in analyzing complex sentiment expressions.



<br/>
# 예제  


테스트 데이터셋에서의 구체적인 예시와 모델 간의 성능 차이를 보여줍니다.

#### 예시 샘플:
**학생 응답**:
> "내 대학의 정신 건강 지원 팀은 매우 도움이 됩니다. 하지만 학업에서 오는 스트레스가 학생들의 정신 건강에 미치는 부정적인 영향을 더 많이 다뤄야 한다고 생각합니다."

#### 결과 비교:
- **모든 모델의 예측 결과**:
  - **Logistic Regression (LR)**: "Neutral"로 분류. 학업 스트레스에 대한 언급이 감정을 드러내는 것으로 간주되지 않았습니다.
  - **Support Vector Machine (SVM)**: "Satisfied"로 분류. 긍정적인 언급만 초점으로 맞추었습니다.
  - **BERT**: "Mixed"로 분류. 응답의 긍정적 및 부정적 요소를 모두 고려했으나, 세부적인 맥락 이해는 부족했습니다.
  - **GPT-3.5**: "Mixed"로 정확히 분류. 긍정적인 지원 경험과 학업 스트레스의 부정적 영향을 모두 반영하여 이 범주에 맞게 예측했습니다.

#### GPT-3.5의 강점:
- GPT-3.5는 응답의 첫 번째 문장에서 "도움이 됩니다"와 같은 긍정적 표현을 감지하고, 두 번째 문장에서 "더 많이 다뤄야 한다"고 강조된 부족한 부분도 포함하여 **혼합(Mixed)** 감정을 정확히 분류했습니다.
- 반면, LR과 SVM은 응답의 복합적인 성격을 감지하지 못해 각각 "Neutral" 또는 "Satisfied"로 잘못 분류했습니다.
- BERT는 복합적인 감정을 이해했으나, 텍스트의 맥락을 완전히 파악하지 못해 학업 스트레스의 중요도를 다소 간과했습니다.

---


Here is a concrete example from the test dataset illustrating differences in model performance:

#### Sample Response:
**Student Response**:
> "The mental health support team at my college has been very helpful. However, I think more attention should be paid to the negative impact of stress from school on students' mental health."

#### Comparison of Results:
- **Predicted Results Across Models**:
  - **Logistic Regression (LR)**: Classified as "Neutral." It did not consider the mention of stress as sentimentally significant.
  - **Support Vector Machine (SVM)**: Classified as "Satisfied," focusing only on the positive mention of support.
  - **BERT**: Classified as "Mixed," considering both positive and negative elements but lacked nuanced understanding.
  - **GPT-3.5**: Correctly classified as "Mixed." It captured the positive experience with support services and the negative sentiment about school stress.

#### Strength of GPT-3.5:
- GPT-3.5 accurately detected the positive sentiment in the first sentence ("very helpful") and the negative concern about school stress in the second sentence. It successfully categorized this as **Mixed**.
- In contrast, LR and SVM failed to recognize the dual nature of the response, misclassifying it as "Neutral" or "Satisfied," respectively.
- While BERT identified the mixed sentiment, it lacked the depth to fully grasp the importance of the concern regarding school stress. 

This example highlights GPT-3.5's superior ability to handle nuanced and context-rich student feedback.


<br/>  
# 요약   




이 연구는 대학생들의 정신 건강 지원에 대한 감정을 분석하기 위해 SMILE-College라는 데이터셋을 구축하고, 대규모 언어 모델(GPT-3.5, BERT)을 포함한 다양한 모델을 비교했습니다. 데이터셋은 College Pulse의 학생 설문 데이터를 기반으로, 만족, 불만족, 혼합, 중립의 네 가지 감정 범주로 주석 처리되었습니다. 결과적으로, GPT-3.5는 전체 F1 점수 0.80으로 가장 우수한 성능을 보였으며, 특히 복잡한 감정인 혼합 범주에서 다른 모델 대비 뛰어난 성능을 보였습니다. 예를 들어, 긍정적인 지원 경험과 학업 스트레스를 모두 언급한 응답에서 GPT-3.5는 정확히 혼합으로 분류한 반면, 전통적 모델은 잘못 분류했습니다. 이러한 연구는 인간-기계 협력 주석과 데이터 기반 접근법을 통해 대학 정신 건강 서비스 개선에 기여할 수 있는 새로운 가능성을 제시합니다.

---



This study analyzed student sentiments on college mental health support by creating the SMILE-College dataset and comparing various models, including GPT-3.5 and BERT. The dataset, based on student survey data from College Pulse, was annotated into four sentiment categories: Satisfied, Dissatisfied, Mixed, and Neutral. GPT-3.5 achieved the highest overall F1 score of 0.80, excelling in the complex Mixed category compared to other models. For instance, in responses mentioning both positive experiences with support and concerns about academic stress, GPT-3.5 accurately classified them as Mixed, whereas traditional models misclassified them. This research demonstrates the potential of human-machine collaborative annotation and data-driven approaches to enhance college mental health services.


<br/>  
# 기타  


논문에 포함된 주요 테이블과 피규어를 요약하고 설명합니다.

#### 1. **Table I: 감정 범주의 대표 예시**
- **설명**: SMILE-College 데이터셋에서 "만족(Satisfied)", "불만족(Dissatisfied)", "혼합(Mixed)", "중립(Neutral)"의 각 감정 범주를 대표하는 예시를 보여줍니다.
- **의의**: 각 범주에 대한 주석 처리 기준을 명확히 하고, 모델이 분류해야 하는 텍스트의 복잡성을 강조합니다.

#### 2. **Table III: 데이터셋 통계**
- **설명**: SMILE-College 데이터셋의 감정별 레코드 수, 응답 길이(최소, 최대, 평균 단어 수), 문장 수(최소, 최대, 평균)를 제시합니다.
  - 예: "불만족" 범주의 평균 단어 수는 33.01개로 가장 길며, "중립" 범주는 18.15개로 가장 짧습니다.
- **의의**: 데이터의 다양성과 불균형을 보여주며, 모델 평가 시 데이터 불균형의 영향을 고려해야 함을 시사합니다.

#### 3. **Table IV: 모델 성능 비교 (테스트 세트 기준)**
- **설명**: GPT-3.5, BERT, LR, SVM 등 주요 모델의 정밀도(Precision), 재현율(Recall), F1 점수를 감정 범주별로 비교합니다.
  - **결과**: GPT-3.5가 전체 F1 점수 0.80으로 가장 우수, BERT가 0.78로 그 뒤를 이었습니다.
- **의의**: GPT-3.5가 모든 범주에서 균형 잡힌 성능을 보였음을 강조합니다.

#### 4. **Table VI: 감정 분석을 통해 식별된 정신 건강 서비스의 주요 한계**
- **설명**: 불만족 응답을 바탕으로 GPT-3.5와 클러스터링 기법을 활용해 대학 정신 건강 서비스의 주요 문제점을 분류하고 빈도를 나열했습니다.
  - 주요 문제: 상담 서비스의 품질(157건), 접근성(76건), 다양성과 포용성(22건) 등.
- **의의**: 대학들이 집중해야 할 주요 개선 영역을 데이터 기반으로 도출합니다.

#### 5. **Figure 1: 감정 주석 프레임워크**
- **설명**: LLM과 인간 협력을 통한 데이터 주석 처리 과정을 단계별로 시각화합니다.
  - 1단계: LLM을 사용한 초기 감정 분류
  - 2단계: "혼합(Mixed)" 범주의 추가
  - 3단계: 인간 검증 및 협업
- **의의**: 인간-기계 협력의 중요성을 시각적으로 보여줍니다.

#### 6. **Figure 2: 혼동 행렬(Confusion Matrix)**
- **설명**: GPT-3.5, BERT, Mistral, Orca 2 등 주요 모델이 SMILE-College 데이터셋에서 보여준 분류 오류 패턴을 비교합니다.
  - 예: GPT-3.5는 "만족"과 "혼합" 사이의 혼동이 적은 반면, Mistral은 "혼합"을 "중립"으로 잘못 분류하는 경우가 많았습니다.
- **의의**: 각 모델의 약점과 강점을 시각적으로 분석하여 모델 개선 방향성을 제공합니다.

---



Key tables and figures from the paper are summarized below:

#### 1. **Table I: Representative Examples for Sentiment Categories**
- **Description**: Displays representative examples from the SMILE-College dataset for each sentiment category: "Satisfied," "Dissatisfied," "Mixed," and "Neutral."
- **Significance**: Clarifies the annotation criteria for each category and highlights the complexity of text that the models must classify.

#### 2. **Table III: Dataset Statistics**
- **Description**: Provides statistics for the SMILE-College dataset, including the number of records per sentiment, response length (minimum, maximum, average words), and sentence count (minimum, maximum, average).
  - Example: The "Dissatisfied" category has the longest average response length (33.01 words), while "Neutral" has the shortest (18.15 words).
- **Significance**: Highlights the diversity and imbalance in the dataset, which must be considered during model evaluation.

#### 3. **Table IV: Model Performance Comparison (Test Set)**
- **Description**: Compares Precision, Recall, and F1 scores across sentiment categories for major models like GPT-3.5, BERT, LR, and SVM.
  - **Results**: GPT-3.5 achieved the highest overall F1 score (0.80), followed by BERT (0.78).
- **Significance**: Emphasizes GPT-3.5's balanced performance across all sentiment categories.

#### 4. **Table VI: Identified Limitations in Mental Health Services**
- **Description**: Lists major issues in college mental health services identified from "Dissatisfied" responses using GPT-3.5 and clustering techniques.
  - Key Issues: Quality of counseling services (157 mentions), accessibility (76 mentions), diversity and inclusivity (22 mentions).
- **Significance**: Provides data-driven insights into areas that colleges should prioritize for improvement.

#### 5. **Figure 1: Sentiment Annotation Framework**
- **Description**: Visualizes the step-by-step process of annotating data through LLM and human collaboration:
  - Step 1: Initial sentiment classification using LLMs.
  - Step 2: Introduction of the "Mixed" sentiment category.
  - Step 3: Human validation and collaboration.
- **Significance**: Demonstrates the importance of human-machine collaboration in ensuring high-quality annotations.

#### 6. **Figure 2: Confusion Matrices**
- **Description**: Compares error patterns of major models (GPT-3.5, BERT, Mistral, Orca 2) on the SMILE-College dataset.
  - Example: GPT-3.5 shows minimal confusion between "Satisfied" and "Mixed," whereas Mistral frequently misclassifies "Mixed" as "Neutral."
- **Significance**: Visually analyzes each model's strengths and weaknesses, providing insights for future improvements.


<br/>
# refer format:     

@inproceedings{sood2024student_sentiment,
  author    = {Palak Sood and Chengyang He and Divyanshu Gupta and Yue Ning and Ping Wang},
  title     = {Understanding Student Sentiment on Mental Health Support in Colleges Using Large Language Models},
  booktitle = {Proceedings of the 2024 IEEE International Conference on Big Data (IEEE BigData)},
  address   = {Washington D.C.},
  month     = {December},
  year      = {2024},
  pages     = {To appear}
}



Sood, Palak, Chengyang He, Divyanshu Gupta, Yue Ning, and Ping Wang. "Understanding Student Sentiment on Mental Health Support in Colleges Using Large Language Models." In Proceedings of the 2024 IEEE International Conference on Big Data (IEEE BigData), Washington, D.C., December 15–18, 2024. Forthcoming.









