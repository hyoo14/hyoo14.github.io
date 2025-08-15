---
layout: post
title:  "[2025]Knowledge Tracing in Programming Education: Integrating Students’ Questions"
date:   2025-08-15 19:44:40 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 



학생의 질문과 자동으로 추출된 기술 정보를 활용하여 학생의 향후 문제 해결 성과를 예측하는 SQKT(학생 질문 기반 지식 추적) 모델을 소개(질문의 표면적 내용뿐만 아니라 학생의 숙련도와 개념적 이해를 포착하는 의미적으로 풍부한 임베딩을 생성)   

이를 통해 학생의 과제 완료 예측  


짧은 요약(Abstract) :


이 논문에서는 프로그래밍 교육에서의 지식 추적(Knowledge Tracing, KT)의 독특한 도전 과제를 다루고 있습니다. 코딩 과제의 복잡성과 학생들이 문제를 해결하는 다양한 방법으로 인해 KT 모델은 학생들의 이해도와 오해를 반영하는 질문을 입력으로 통합하지 않는 경우가 많습니다. 본 논문에서는 학생의 질문과 자동으로 추출된 기술 정보를 활용하여 학생의 향후 문제 해결 성과를 예측하는 SQKT(학생 질문 기반 지식 추적) 모델을 소개합니다. 이 방법은 질문의 표면적 내용뿐만 아니라 학생의 숙련도와 개념적 이해를 포착하는 의미적으로 풍부한 임베딩을 생성합니다. 실험 결과, SQKT는 다양한 난이도의 파이썬 프로그래밍 과정에서 학생의 과제 완료 예측에서 기존 모델보다 33.1%의 절대적인 개선을 보였습니다. 또한, SQKT는 교차 도메인 설정에서도 강력한 일반화 능력을 보여주며, 고급 프로그래밍 과정에서 데이터 부족 문제를 효과적으로 해결합니다. SQKT는 개인의 학습 요구에 맞춘 교육 콘텐츠를 조정하고 컴퓨터 과학 교육에서 적응형 학습 시스템을 설계하는 데 사용될 수 있습니다.



This paper addresses the unique challenges of knowledge tracing (KT) in programming education. Due to the complexity of coding tasks and the diverse methods students use to solve problems, traditional KT models often neglect to incorporate students' questions as inputs, which contain valuable signals about their understanding and misconceptions. We introduce SQKT (Students’ Question-based Knowledge Tracing), a knowledge tracing model that leverages students' questions and automatically extracted skill information to enhance the accuracy of predicting students' performance on subsequent problems in programming education. Our method creates semantically rich embeddings that capture not only the surface-level content of the questions but also the student's mastery level and conceptual understanding. Experimental results demonstrate SQKT's superior performance in predicting student completion across various Python programming courses of differing difficulty levels, achieving a 33.1% absolute improvement in AUC compared to baseline models. The model also exhibited robust generalization capabilities in cross-domain settings, effectively addressing data scarcity issues in advanced programming courses. SQKT can be used to tailor educational content to individual learning needs and design adaptive learning systems in computer science education.


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


**SQKT (Students’ Question-based Knowledge Tracing) 모델 개요**

SQKT는 프로그래밍 교육에서 학생의 성과를 예측하기 위해 설계된 지식 추적 모델입니다. 이 모델은 학생의 질문과 자동으로 추출된 기술 정보를 통합하여 학생의 이해도를 보다 정확하게 반영합니다. SQKT의 주요 구성 요소와 방법론은 다음과 같습니다.

1. **다양한 입력 특징 통합**: SQKT는 학생의 문제 해결 이력, 코드 제출, 학생 질문 및 문제에 필요한 기술 정보를 포함한 다양한 입력 특징을 통합합니다. 이를 통해 학생의 지식 상태를 보다 포괄적으로 모델링할 수 있습니다.

2. **학생 질문의 통합**: 학생의 질문은 그들의 이해도와 혼란을 나타내는 중요한 신호를 제공합니다. SQKT는 CodeT5 모델을 사용하여 학생 질문을 임베딩하고, 이를 통해 학생의 질문 내용이 모델의 예측 정확도를 높이는 데 기여합니다.

3. **기술 추출**: SQKT는 학생 질문에서 기술 정보를 자동으로 추출하는 시스템을 사용합니다. 이 시스템은 GPT-4o를 기반으로 하여 학생 질문과 관련된 기술을 식별하고, 이를 통해 학생이 어떤 개념에서 어려움을 겪고 있는지를 파악합니다.

4. **임베딩 및 융합 레이어**: SQKT는 학생의 코드 제출, 문제 설명, 질문 및 기술 정보를 각각 임베딩한 후, 이를 융합하여 통합된 표현 공간으로 변환합니다. 이 과정에서 트리플 손실(triplet loss)을 사용하여 서로 다른 입력 특징 간의 관계를 강화합니다.

5. **다중 헤드 자기 주의 메커니즘**: 모든 임베딩은 다중 헤드 자기 주의 메커니즘을 통해 처리되어, 학생의 성공 또는 실패를 예측하는 데 필요한 복잡한 상호작용을 캡처합니다.

6. **훈련 및 평가**: SQKT는 다양한 프로그래밍 코스에서 수집된 데이터를 사용하여 훈련됩니다. 모델의 성능은 AUC, 정확도 및 F1 점수와 같은 지표를 통해 평가됩니다. SQKT는 기존의 지식 추적 모델에 비해 성능이 우수하며, 특히 학생 질문을 통합함으로써 예측 정확도가 크게 향상됩니다.

이러한 방법론을 통해 SQKT는 프로그래밍 교육에서 학생의 개별 학습 요구에 맞춘 맞춤형 교육 콘텐츠를 제공하고, 적응형 학습 시스템을 설계하는 데 기여할 수 있습니다.

---


**Overview of the SQKT (Students’ Question-based Knowledge Tracing) Model**

SQKT is a knowledge tracing model designed to predict student performance in programming education. This model integrates students' questions and automatically extracted skill information to more accurately reflect students' understanding. The main components and methodologies of SQKT are as follows:

1. **Integration of Diverse Input Features**: SQKT integrates various input features, including the student's problem-solving history, code submissions, student questions, and the skill information required for the problems. This allows for a more comprehensive modeling of the student's knowledge state.

2. **Incorporation of Student Questions**: Students' questions provide valuable signals about their understanding and confusion. SQKT uses the CodeT5 model to embed student questions, contributing to improved prediction accuracy by effectively utilizing the content of these questions.

3. **Skill Extraction**: SQKT employs a system to automatically extract skill information from student questions. This system is based on GPT-4o and identifies skills related to student questions, helping to pinpoint the concepts where students struggle.

4. **Embedding and Fusion Layer**: SQKT embeds the student's code submissions, problem descriptions, questions, and skill information, then fuses them into a unified representation space. During this process, a triplet loss is used to strengthen the relationships among different input features.

5. **Multi-Head Self-Attention Mechanism**: All embeddings are processed through a multi-head self-attention mechanism, capturing the complex interactions necessary to predict student success or failure.

6. **Training and Evaluation**: SQKT is trained using data collected from various programming courses. The model's performance is evaluated using metrics such as AUC, accuracy, and F1 score. SQKT consistently outperforms existing knowledge tracing models, particularly by integrating student questions, which significantly enhances prediction accuracy.

Through these methodologies, SQKT can provide personalized educational content tailored to individual learning needs in programming education and contribute to the design of adaptive learning systems.


<br/>
# Results


이 논문에서는 SQKT(Students’ Question-based Knowledge Tracing) 모델을 제안하며, 이 모델이 기존의 지식 추적(Knowledge Tracing, KT) 모델에 비해 학생의 프로그래밍 문제 해결 능력을 예측하는 데 있어 얼마나 효과적인지를 실험적으로 검증하였습니다. SQKT는 학생의 질문과 자동으로 추출된 기술 정보를 통합하여 학생의 성과를 예측하는 데 필요한 정보를 보다 풍부하게 제공합니다.

#### 실험 설정
SQKT 모델은 한국의 온라인 프로그래밍 교육 플랫폼에서 수집된 데이터를 사용하여 평가되었습니다. 데이터는 네 가지 Python 프로그래밍 과정에서 수집되었으며, 각 과정은 서로 다른 난이도를 가지고 있습니다. 실험은 두 가지 주요 설정으로 나뉘어 진행되었습니다: 
1. **인도메인 실험**: 동일한 과정에서 훈련 및 테스트를 수행하여 모델의 성능을 평가합니다.
2. **크로스 도메인 실험**: 서로 다른 과정 간의 일반화 능력을 평가하기 위해 훈련된 모델을 다른 과정에서 테스트합니다.

#### 결과
SQKT 모델은 인도메인 실험에서 AUC(Area Under the Curve) 메트릭에서 87.1%에서 93.4%의 성과를 기록하며, 기존의 경쟁 모델(KTMFF, OKT 등)에 비해 12.6%에서 20.8%의 절대 개선을 보였습니다. 특히, SQKT는 학생의 질문을 통합함으로써 예측 정확도를 크게 향상시켰습니다.

크로스 도메인 실험에서는 SQKT가 다른 과정의 데이터에 대해서도 뛰어난 일반화 능력을 보여주었습니다. 예를 들어, "Python Introduction" 과정에서 훈련된 모델이 "First Python" 과정에서 45.3%의 AUC 개선을 보였고, 데이터가 부족한 "Algorithm" 과정에서도 11.4%의 성과 향상을 기록했습니다. 이는 학생의 질문이 다양한 교육 콘텐츠에 대한 일반화된 통찰력을 제공함을 나타냅니다.

#### 결론
SQKT 모델은 학생의 질문을 통합하여 프로그래밍 교육에서 학생의 성과를 예측하는 데 있어 기존 모델보다 더 효과적임을 입증하였습니다. 이 연구는 SQKT가 개인화된 학습 경험을 제공하고, 학생의 이해도를 보다 정확하게 평가할 수 있는 가능성을 제시합니다.

---




This paper introduces the SQKT (Students’ Question-based Knowledge Tracing) model and experimentally validates its effectiveness in predicting students' programming problem-solving abilities compared to existing Knowledge Tracing (KT) models. SQKT integrates students' questions and automatically extracted skill information to provide richer insights necessary for predicting student performance.

#### Experimental Setup
The SQKT model was evaluated using data collected from a Korean online programming education platform. The data covered four distinct Python programming courses, each with varying difficulty levels. The experiments were conducted in two main settings:
1. **In-domain experiments**: The model's performance was evaluated by training and testing within the same course.
2. **Cross-domain experiments**: The model's generalization ability was assessed by testing it on different courses.

#### Results
In the in-domain experiments, the SQKT model achieved an AUC (Area Under the Curve) metric ranging from 87.1% to 93.4%, demonstrating an absolute improvement of 12.6% to 20.8% over existing competitive models (KTMFF, OKT, etc.). Notably, SQKT significantly enhanced prediction accuracy by incorporating students' questions.

In the cross-domain experiments, SQKT exhibited remarkable generalization capabilities across different course data. For instance, the model trained on the "Python Introduction" course showed a 45.3% improvement in AUC when tested on the "First Python" course, and it achieved an 11.4% performance increase on the data-scarce "Algorithm" course. This indicates that students' questions provide generalized insights applicable across diverse educational content.

#### Conclusion
The SQKT model has been proven to be more effective than existing models in predicting student performance in programming education by integrating students' questions. This research suggests that SQKT has the potential to offer personalized learning experiences and accurately assess students' understanding.


<br/>
# 예제



이 논문에서는 프로그래밍 교육에서 학생의 성과를 예측하기 위한 새로운 지식 추적 모델인 SQKT(Students’ Question-based Knowledge Tracing)를 소개합니다. 이 모델은 학생들이 제출한 코드와 질문을 통합하여 학생의 이해도를 더 정확하게 평가합니다. 

#### 데이터셋 예시

1. **Python Basic - 문제 예시**
   - **문제 설명**: 자연수를 입력받아 1부터 입력받은 수까지의 합의 제곱과 제곱의 합의 차이를 출력하는 프로그램을 작성하시오.
   - **문제 해결 코드**:
     ```python
     N = int(input())
     i_square = 0
     i_list = list(range(1, N + 1))
     for i in i_list:
         i_square += i**2
     sum_square = sum(i_list)**2
     print(sum_square - i_square)
     ```
   - **학생의 코드 제출**:
     ```python
     summation = 0
     while num > 0:
         summation = summation + 1
         num = num - 1
     print(summation)
     ```
   - **학생의 질문**: "왜 summation 변수가 출력할 때 올바른 값을 생성하지 않나요?"
   - **스킬**: While 루프, Print 함수, 연산자
   - **교육자의 응답**: "While 루프가 각 반복에서 summation을 1씩 증가시키므로, 10을 입력하면 summation의 최종 값은 10이 됩니다."

2. **Python Introduction - 문제 예시**
   - **문제 설명**: 사용자가 입력한 값을 받아서 출력하는 프로그램을 작성하시오.
   - **문제 해결 코드**:
     ```python
     var = input()
     print('Parrot:', var)
     ```
   - **학생의 코드 제출**:
     ```python
     var = raw_input("Enter a value:")
     print('Parrot:', var)
     ```
   - **학생의 질문**: "input(”) 함수에 무엇을 입력해야 하나요?"
   - **스킬**: 변수 할당, 연산자, input 함수, print 함수
   - **교육자의 응답**: "입력할 때 문자열 'Parrot'를 따옴표로 묶어야 합니다. 예를 들어, var = input('Parrot')와 같이 입력할 수 있습니다."

이와 같은 방식으로, SQKT 모델은 학생의 질문과 코드 제출을 분석하여 학생의 이해도를 평가하고, 이를 통해 맞춤형 학습 경험을 제공할 수 있습니다.

---




This paper introduces a new knowledge tracing model called SQKT (Students’ Question-based Knowledge Tracing) aimed at predicting student performance in programming education. The model integrates students' code submissions and questions to more accurately assess their understanding.

#### Dataset Example

1. **Python Basic - Problem Example**
   - **Problem Description**: Write a program that takes a natural number as input and outputs the difference between the square of the sum and the sum of the squares for numbers from 1 to the given input.
   - **Problem Solution Code**:
     ```python
     N = int(input())
     i_square = 0
     i_list = list(range(1, N + 1))
     for i in i_list:
         i_square += i**2
     sum_square = sum(i_list)**2
     print(sum_square - i_square)
     ```
   - **Student's Code Submission**:
     ```python
     summation = 0
     while num > 0:
         summation = summation + 1
         num = num - 1
     print(summation)
     ```
   - **Student's Question**: "Why doesn't the summation variable produce the correct value when printed?"
   - **Skill**: While loop, Print function, Operator
   - **Educator's Response**: "Since the while loop increments summation by 1 in each iteration, if you input 10, the final value stored in summation will be 10."

2. **Python Introduction - Problem Example**
   - **Problem Description**: Write a program that takes user input and prints it.
   - **Problem Solution Code**:
     ```python
     var = input()
     print('Parrot:', var)
     ```
   - **Student's Code Submission**:
     ```python
     var = raw_input("Enter a value:")
     print('Parrot:', var)
     ```
   - **Student's Question**: "What should I enter in the input(”) function?"
   - **Skill**: Variable assignment, Operator, input function, print function
   - **Educator's Response**: "You need to enclose the string 'Parrot' in quotation marks when entering it. For example, you can input it as follows: var = input('Parrot')."

In this way, the SQKT model analyzes students' questions and code submissions to evaluate their understanding, providing a tailored learning experience.

<br/>
# 요약


이 논문은 SQKT(학생 질문 기반 지식 추적) 모델을 제안하여 프로그래밍 교육에서 학생의 성과를 예측하는 데 학생의 질문을 통합하는 방법을 설명합니다. SQKT는 학생의 질문과 자동으로 추출된 기술 정보를 활용하여 예측 정확도를 향상시키며, 다양한 Python 프로그래밍 과정에서 우수한 성능을 보입니다. 실험 결과, SQKT는 기존 모델에 비해 AUC에서 최대 33.1%의 절대 개선을 달성했습니다.



This paper introduces the SQKT (Students' Question-based Knowledge Tracing) model, which integrates students' questions to predict performance in programming education. SQKT enhances prediction accuracy by leveraging students' questions and automatically extracted skill information, demonstrating superior performance across various Python programming courses. Experimental results show that SQKT achieves up to a 33.1% absolute improvement in AUC compared to baseline models.

<br/>
# 기타



#### 1. 다이어그램 및 피규어
- **Figure 1**: SQKT의 프로세스를 보여주는 다이어그램으로, 학생의 문제 해결 이력, 질문, 그리고 다음 문제의 기술 정보를 통합하여 학생의 성공 여부를 예측하는 과정을 설명합니다. 이 다이어그램은 SQKT 모델이 어떻게 다양한 입력을 처리하여 예측을 수행하는지를 시각적으로 나타냅니다.
  
- **Figure 2**: SQKT의 전체 아키텍처를 보여주는 다이어그램으로, 문제 설명, 코드 제출, 학생 질문, 기술 정보의 세 가지 임베딩 레이어를 통해 정보를 처리하는 과정을 설명합니다. 이 구조는 다양한 입력을 통합하여 학생의 지식 상태를 예측하는 데 중요한 역할을 합니다.

#### 2. 테이블
- **Table 4**: 다양한 모델의 성능 비교를 보여줍니다. SQKT는 세 가지 데이터셋에서 모든 기준(AUC, 정확도, F1 점수)에서 다른 모델들보다 우수한 성능을 보였습니다. 이는 학생 질문을 통합한 SQKT 모델이 학생의 성과 예측에 효과적임을 나타냅니다.

- **Table 5**: SQKT의 구성 요소에 대한 기여도를 평가하기 위한 아블레이션 연구 결과를 보여줍니다. 질문 임베딩과 기술 임베딩을 각각 제거했을 때 성능이 감소하는 것을 통해 두 요소의 중요성을 강조합니다.

- **Table 6**: 보조 손실 함수의 영향을 분석한 결과를 보여줍니다. 질문 손실과 트리플렛 손실이 모델 성능에 긍정적인 영향을 미친다는 것을 나타냅니다.

- **Table 12**: 오류 분석 결과를 보여줍니다. 오류 유형별로 발생 빈도를 나열하고, 각 오류의 예시와 원인을 설명합니다. 이는 모델의 개선이 필요한 영역을 식별하는 데 도움이 됩니다.

#### 3. 어펜딕스
- **Appendix C**: 데이터셋의 예시를 제공합니다. 각 문제에 대한 설명, 학생의 코드 제출, 학생 질문, 교육자의 응답이 포함되어 있습니다. 이는 SQKT 모델이 어떻게 학생의 질문과 코드 제출을 통해 지식을 추적하는지를 보여줍니다.




#### 1. Diagrams and Figures
- **Figure 1**: A diagram illustrating the process of SQKT, showing how it integrates a student's problem-solving history, questions, and skill information to predict the success of the next problem. This diagram visually represents how the SQKT model processes various inputs to make predictions.

- **Figure 2**: A comprehensive architecture diagram of SQKT, detailing how it processes problem descriptions, code submissions, student questions, and skill information through three embedding layers. This structure plays a crucial role in integrating diverse inputs to predict the student's knowledge state.

#### 2. Tables
- **Table 4**: A performance comparison of various models, showing that SQKT consistently outperforms all baseline models across three datasets in all metrics (AUC, accuracy, F1 score). This indicates the effectiveness of the SQKT model in predicting student performance by incorporating student questions.

- **Table 5**: Results from an ablation study assessing the contribution of each component in SQKT. The performance drops when either question or skill embeddings are removed, highlighting the importance of both elements.

- **Table 6**: Analysis of the impact of auxiliary loss functions, showing that both question loss and triplet loss positively influence model performance.

- **Table 12**: Error analysis results detailing the occurrence of different error types, along with examples and reasons for each. This helps identify areas for improvement in the model.

#### 3. Appendix
- **Appendix C**: Provides examples from the dataset, including problem descriptions, student code submissions, student questions, and educator responses. This illustrates how the SQKT model tracks knowledge through student questions and code submissions.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{kim2025knowledge,
  title={Knowledge Tracing in Programming Education: Integrating Students’ Questions},
  author={Kim, Doyoun and Kim, Suin and Jo, Yohan},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={27703--27718},
  year={2025},
  month={July},
  publisher={Association for Computational Linguistics},
  url={https://github.com/holi-lab/SQKT}
}
```

### Chicago Style Citation

Kim, Doyoun, Suin Kim, and Yohan Jo. "Knowledge Tracing in Programming Education: Integrating Students’ Questions." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 27703–27718. July 27 - August 1, 2025. Association for Computational Linguistics. https://github.com/holi-lab/SQKT.
