---
layout: post
title:  "[2025]Machine Learning Meets Algebraic Combinatorics: A Suite of Datasets Capturing Research-level Conjecturing Ability in Pure Mathematics"  
date:   2025-07-25 13:59:40 +0900
categories: study
---

{% highlight ruby %}


한줄 요약: 

수학용 LLM테스트셋 새로이 제공(대수적 기초 이론 반영 및 난이도 분화)  

짧은 요약(Abstract) :    



최근 인공지능의 능력이 급격히 향상되면서, 수학처럼 고차원적 추론이 요구되는 분야에 머신러닝을 활용하려는 관심이 높아졌습니다. 하지만 기존의 수학 관련 데이터셋은 대부분 고등학교, 대학 학부 또는 대학원 수준에 머물러 있으며, 실제 수학자들이 다루는 수준의 개방형 문제를 반영한 자원은 거의 없습니다. 이를 해결하기 위해, 본 논문은 대수적 조합론(Algebraic Combinatorics) 분야에서의 기초 이론이나 미해결 문제들을 다룬 ACD Repo (Algebraic Combinatorics Dataset Repository) 라는 새로운 데이터셋 모음을 소개합니다. 이 데이터셋은 수백만 개의 예제를 포함하며, 각 데이터셋은 연구 수준의 개방형 문제를 기반으로 구성되어 있습니다. 특히, 수학적 추측(conjecture) 을 생성하는 과정을 중점적으로 다루고 있으며, 해석 가능한 모델 분석이나 LLM 기반 코드 생성을 통해 모델을 적용할 수 있는 다양한 방법도 함께 제시됩니다. 이러한 데이터셋은 머신러닝이 수학 탐구에 기여할 수 있는 새로운 가능성을 열어줍니다.



With recent dramatic increases in AI system capabilities, there has been growing interest in utilizing machine learning for reasoning-heavy, quantitative tasks, particularly mathematics. While there are many resources capturing mathematics at the high-school, undergraduate, and graduate level, there are far fewer resources available that align with the level of difficulty and open endedness encountered by professional mathematicians working on open problems. To address this, we introduce a new collection of datasets, the Algebraic Combinatorics Dataset Repository (ACD Repo), representing either foundational results or open problems in algebraic combinatorics, a subfield of mathematics that studies discrete structures arising from abstract algebra. Further differentiating our dataset collection is the fact that it aims at the conjecturing process. Each dataset includes an open-ended research level question and a large collection of examples (up to 10M in some cases) from which conjectures should be generated. We describe all nine datasets, the different ways machine learning models can be applied to them (e.g., training with narrow models followed by interpretability analysis or program synthesis with LLMs), and discuss some of the challenges involved in designing datasets like these.





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



이 논문은 새로운 모델 아키텍처를 제안하기보다는, 대수적 조합론(Algebraic Combinatorics)에서 등장하는 수학적 추측(conjecture) 생성 능력을 평가하고 도전할 수 있도록 설계된 **9개의 데이터셋(A.C.D Repo)**을 소개합니다. 각 데이터셋은 다음과 같은 방식으로 구성되어 있습니다:

데이터셋 구성:

각 데이터셋은 실제 수학자들이 관심 갖는 기초 이론 또는 미해결 문제를 중심으로 제작됨.

문제는 대부분 정수 분할, 순열, 영 타블로(Young tableaux) 등 이산적 수학 구조로 표현되며, 이는 컴퓨터에서 다루기 용이함.

모델 아키텍처:

실험에 사용된 모델은 로지스틱 회귀, MLP (다층 퍼셉트론), Transformer 등 기본적인 신경망 모델들로 구성됨.

일부 태스크에 대해선 GPT-4o, Claude, GPT-mini 등의 LLM 기반 프로그램 생성(program synthesis) 방식도 사용.

학습 전략:

일부 태스크에서는 단순한 분류나 회귀 문제가 아니라, 수학적 직관이나 패턴을 이해하는 능력이 요구됨.

특히 "interpretability"를 활용한 분석이나, 프로그래밍 코드 생성 기반의 추론 전략이 강조됨.

특별한 기법:

모델 성능보다도 수학적 통찰을 도출할 수 있는지 여부를 중시.

예: 구조 상수(structure constant) 예측 문제에서는 LLM이 데이터 생성 과정에서의 패턴(예: permutation length의 짝수/홀수 여부)을 역추론함으로써, 사람이 명시하지 않은 수학적 규칙을 발견하는 사례도 있음.




Rather than introducing new model architectures, this paper focuses on the construction and application of a suite of nine datasets—the Algebraic Combinatorics Dataset Repository (ACD Repo)—designed to evaluate machine learning models’ ability to engage in research-level conjecture generation in pure mathematics.

Dataset Construction:

Each dataset is built around a foundational or open mathematical problem in algebraic combinatorics.

The problems involve discrete structures such as partitions, permutations, Young tableaux, making them well-suited to digital representation and ML processing.

Model Architectures:

The authors experiment with logistic regression, multi-layer perceptrons (MLPs), and Transformers as baseline models.

Additionally, large language models (LLMs) like GPT-4o, Claude, and Mini GPT-4o are applied using program synthesis approaches.

Training Strategy:

Tasks go beyond standard classification/regression and require models to grasp mathematical patterns or intuitions.

The study emphasizes interpretability-driven analysis and code-generating models as key tools for uncovering deeper mathematical insight.

Special Techniques:

The focus is not solely on prediction accuracy but rather on whether the model can aid in generating meaningful mathematical conjectures.

For example, in predicting Schubert polynomial structure constants, LLMs reverse-engineered dataset generation rules based on permutation parity—demonstrating implicit learning of unprovided mathematical properties.




   
 
<br/>
# Results  




논문에서는 총 9개의 데이터셋에 대해 기본 머신러닝 모델(logistic regression, MLP, Transformer) 및 **LLM 기반 접근(GPT-4o, Claude 등)**의 성능을 비교합니다. 주요 결과는 다음과 같습니다:

모델 성능 비교:

대부분의 분류 태스크에서는 MLP가 가장 안정적이고 높은 정확도를 보임.

Transformer는 일부 태스크(예: 격자 경로)에서 성능이 불안정하거나 낮음.

LLM 기반 program synthesis는 일부 경우(예: 구조 상수 예측)에서 놀랍도록 정확하게 정답을 유도함 (100% accuracy).

성능 예시 (Table 1):

예: mHeight 문제 (n=10)

Logistic: 94.2%

MLP: 99.9%

Transformer: 99.9%

예: 슈베르트 다항식 구조 상수 (n=6)

Logistic: 89.7%

MLP: 99.8%

Transformer: 91.3%

어려운 태스크:

대칭군(Sn)의 character 예측 및 RSK 대응 같은 회귀 문제는 전통적인 모델에서 성능이 매우 낮음 (예: MSE 기준 수십억 이상).

예: Sn characters (n=20)의 MSE

Linear regression: 4.20×10¹²

MLP: 4.22×10¹²

Transformer: 5.39×10¹²

모델 해석의 중요성:

높은 정확도를 달성한 경우라도, 그 결과가 실제로 수학적 통찰을 주는지 여부가 핵심임.

예: LLM이 데이터셋 구성 방식(짝수/홀수 길이 합 규칙)을 파악해 정확도는 높았지만, 실제 수학적 의미는 부족했던 사례도 있음.




The paper evaluates baseline models and large language models (LLMs) across 9 datasets in algebraic combinatorics, reporting their performance using metrics such as accuracy (for classification tasks) and mean squared error (MSE) (for regression tasks). Key findings include:

Model Comparison:

MLPs consistently outperform other baseline models across most classification tasks.

Transformers occasionally underperform or exhibit instability (e.g., on lattice path tasks).

LLM-based program synthesis approaches (e.g., with GPT-4o or Claude) sometimes yield perfect predictions (100% accuracy) on combinatorial tasks, suggesting strong symbolic reasoning capabilities.

Performance Examples (from Table 1):

mHeight task (n = 10)

Logistic Regression: 94.2%

MLP: 99.9%

Transformer: 99.9%

Schubert Polynomial Structure Constants (n = 6)

Logistic Regression: 89.7%

MLP: 99.8%

Transformer: 91.3%

Challenging Tasks:

Regression-based tasks like Sn character prediction and RSK correspondence show extremely poor performance in all traditional models.

Example: MSE for Sn characters (n = 20)

Linear regression: 4.20×10¹²

MLP: 4.22×10¹²

Transformer: 5.39×10¹²

Interpretability over Raw Accuracy:

The authors emphasize that accurate predictions alone are not sufficient—mathematical insight must also be extractable.

For instance, an LLM correctly reverse-engineered the data sampling rule (based on permutation length parity), achieving high performance but with limited mathematical value.





<br/>
# 예제  





논문에서 소개된 데이터셋들은 실제 수학 연구에서 등장하는 문제를 기계학습 태스크로 바꾼 사례입니다. 주요 예시는 다음과 같습니다:



예시 1: 대칭군 불가약 표현의 캐릭터 값 예측 (Section 4.1)
입력(Input): 두 개의 정수 분할 (예: λ = (4,2,2), μ = (3,3,2))

출력(Output): 두 분할에 대응하는 대칭군 
𝑆
𝑛
S 
n
​
 의 불가약 캐릭터 값 
𝜒
𝜇
𝜆
χ 
μ
λ
​
 

형식: 회귀 문제 (정수 예측)

예시:

입력: λ = (3,2,1), μ = (2,2,2)

출력: χ = 4



예시 2: mHeight 함수 예측 (Section 4.2)
입력: 하나의 순열 (예: σ = 3 1 4 2)

출력: 해당 순열에 존재하는 3412 패턴 중 최소 높이 (예: mHeight = 1)

형식: 분류 문제 (소수 개의 정수 값 중 선택)

예시:

입력: σ = 3 1 4 2

출력: 1



예시 3: Schubert 다항식 구조 상수 예측 (Section 4.6)
입력: 세 개의 순열 (예: α = 1 2 3, β = 2 1 3, γ = 2 3 1)

출력: 구조 상수 
𝑐
𝛼
,
𝛽
𝛾
c 
α,β
γ
​
  (예: 0 또는 1)

형식: 분류 문제 (정수값 클래스 예측)

예시:

입력: (α, β, γ) = (1 2 3, 2 1 3, 2 3 1)

출력: 1



예시 4: 클러스터 가변수 식별 (Section 4.3)
입력: 3×4 형태의 Semistandard Young Tableau
예:

Copy
Edit
1 1 2 3  
2 3 4 5  
4 5 6 7  
출력: 이 tableau가 Grassmannian 클러스터 가변수를 정의하는지 여부 (True/False)

형식: 이진 분류 (binary classification)

예시:

입력: 위와 같은 Young tableau

출력: True




The paper provides a number of dataset-specific tasks that frame abstract mathematical reasoning into machine learning problems. Here are concrete examples:



Example 1: Predicting Characters of Irreducible Representations of 
𝑆
𝑛
S 
n
​
  (Section 4.1)
Input: Two integer partitions (e.g., λ = (4,2,2), μ = (3,3,2))

Output: The character value 
𝜒
𝜇
𝜆
χ 
μ
λ
​
  of the symmetric group representation

Task: Regression (predicting an integer)

Example:

Input: λ = (3,2,1), μ = (2,2,2)

Output: χ = 4



Example 2: Predicting mHeight of a Permutation (Section 4.2)
Input: A single permutation (e.g., σ = 3 1 4 2)

Output: The minimum height among all 3412-patterns in the permutation

Task: Classification

Example:

Input: σ = 3 1 4 2

Output: 1


Example 3: Schubert Polynomial Structure Constant Prediction (Section 4.6)
Input: A triple of permutations (e.g., α = 1 2 3, β = 2 1 3, γ = 2 3 1)

Output: The structure constant 
𝑐
𝛼
,
𝛽
𝛾
c 
α,β
γ
​
  in Schubert polynomial multiplication

Task: Classification (predicting integer constants)

Example:

Input: (α, β, γ) = (1 2 3, 2 1 3, 2 3 1)

Output: 1


Example 4: Identifying Cluster Variables (Section 4.3)
Input: A 3×4 semistandard Young tableau
Example:

Copy
Edit
1 1 2 3  
2 3 4 5  
4 5 6 7  
Output: Boolean indicating whether the tableau corresponds to a valid cluster variable

Task: Binary classification

Example:

Input: The tableau above

Output: True



<br/>  
# 요약   



이 논문은 대수적 조합론의 연구 수준 문제를 기계학습으로 다룰 수 있도록 9개의 데이터셋(ACD Repo)을 설계하고, 기본 모델(MLP, Transformer)과 LLM 기반 방법(코드 생성 등)을 활용하는 다양한 접근법을 제시한다. 대부분의 분류 문제에서 MLP가 안정적으로 높은 정확도를 보였으며, 일부 태스크에서는 LLM이 데이터 생성 패턴까지 학습하여 100% 예측을 달성하기도 했다. 예를 들어, 순열을 입력받아 구조 상수를 예측하거나, 두 분할에서 대칭군 캐릭터 값을 예측하는 문제 등이 포함된다.




This paper introduces the ACD Repo, a suite of nine datasets designed to model research-level problems in algebraic combinatorics using machine learning, incorporating both narrow models (e.g., MLPs, Transformers) and LLM-based approaches like program synthesis. MLPs performed consistently well across classification tasks, while LLMs occasionally achieved perfect prediction by implicitly learning dataset generation rules. Tasks include predicting structure constants from permutation triples or computing symmetric group characters from two integer partitions.


<br/>  
# 기타  





Table 1: 분류 태스크에 대한 모델 정확도 비교
내용: MLP, Transformer, Logistic Regression 등 모델이 여러 조합론적 태스크에 대해 얼마나 잘 작동하는지를 비교한 테이블.

결과: 대부분의 태스크에서 MLP가 가장 높은 정확도를 기록, 특히 작은 입력 크기에서 성능이 뛰어남.

인사이트: 간단한 MLP조차 잘 설계된 조합론 문제에서는 매우 강력하며, 입력 표현의 단순성도 성능에 큰 영향을 준다.



Table 3: 회귀 태스크에 대한 평균제곱오차 (MSE)
내용: 대칭군 캐릭터 계산이나 RSK 대응 문제처럼 정수 예측이 필요한 태스크에 대해 MSE로 성능을 평가.

결과: 모든 모델에서 매우 큰 오차 발생 (예: 10¹² 수준), 학습이 거의 되지 않았음을 시사.

인사이트: 이러한 문제는 복잡도가 높고, 모델이 수학적 구조나 연산규칙을 이해하지 못할 경우 단순 학습으로는 어려움.



Figure 3~5: 캐릭터 분포의 롱테일 시각화
내용: 출력값(예: 캐릭터 값)의 분포가 얼마나 불균형한지를 시각화한 히스토그램/그래프.

결과: 많은 값이 0 또는 작은 범위에 집중되고, 일부 매우 큰 값이 존재 → long-tail distribution.

인사이트: 분포가 매우 치우쳐 있어 모델이 대다수의 평범한 케이스만 학습하고 중요한 극단값은 무시할 가능성 높음.



Appendix B: 각 데이터셋의 생성 방식, 하이퍼파라미터, 벤치마크 모델 세부 정보
내용: 각 데이터셋의 정확한 구성법, 문제 정의, 모델 학습 방법, 샘플 수 등을 정리.

인사이트: 단순히 모델을 적용하는 것이 아니라, 수학적으로 의미 있는 문제를 ML-friendly하게 구성하는 과정 자체가 핵심 기여로 작용함.



Supplementary Materials – English Version


Table 1: Accuracy Comparison on Classification Tasks
Content: Comparison of MLPs, Transformers, and logistic regression across classification problems in algebraic combinatorics.

Findings: MLPs consistently achieve the highest accuracy, especially on problems with small input size.

Insight: Even simple neural networks can perform remarkably well when the input representation aligns with underlying combinatorial structure.



Table 3: Mean Squared Error on Regression Tasks
Content: Evaluation of tasks such as computing symmetric group characters or RSK correspondences.

Findings: All models exhibit extremely high MSEs (e.g., on the order of 10¹²), indicating poor learning.

Insight: These problems are structurally complex, and traditional models struggle without incorporating deeper mathematical reasoning.



Figures 3–5: Visualization of Long-Tailed Character Distributions
Content: Histograms and distribution plots showing how outputs (e.g., character values) are distributed.

Findings: Sharp imbalance with many values concentrated near 0 and a few extremely large outliers.

Insight: Models may overfit to frequent trivial cases while missing rare but mathematically significant outputs.



Appendix B: Dataset Generation, Model Details, Hyperparameters
Content: Detailed explanation of how datasets were constructed, including problem context, instance counts, and training protocols.

Insight: The design of mathematically meaningful, ML-compatible datasets is itself a major contribution, bridging theoretical math and practical ML.




<br/>
# refer format:     




@inproceedings{chau2025ml4algcomb,
  title     = {Machine Learning Meets Algebraic Combinatorics: A Suite of Datasets Capturing Research-level Conjecturing Ability in Pure Mathematics},
  author    = {Herman Chau and Helen Jenne and Davis Brown and Jesse He and Mark Raugas and Sara C. Billey and Henry Kvinge},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  volume    = {267},
  series    = {Proceedings of Machine Learning Research},
  address   = {Vancouver, Canada},
  publisher = {PMLR},
  url       = {https://github.com/pnnl/ML4AlgComb},
  note      = {Equal contribution by first two authors}
}




Chau, Herman, Helen Jenne, Davis Brown, Jesse He, Mark Raugas, Sara C. Billey, and Henry Kvinge. “Machine Learning Meets Algebraic Combinatorics: A Suite of Datasets Capturing Research-Level Conjecturing Ability in Pure Mathematics.” In Proceedings of the 42nd International Conference on Machine Learning, PMLR 267, Vancouver, Canada, 2025. https://github.com/pnnl/ML4AlgComb.




