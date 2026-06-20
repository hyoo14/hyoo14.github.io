---
layout: post
title:  "[2026]Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning"
date:   2026-06-20 15:14:26 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 예측 프로세스 모니터링을 위한 신경-기호적 접근 방식을 제안하며, 도메인 지식을 차별화 가능한 논리적 제약으로 통합하는 두 단계 최적화 전략을 사용합니다.


짧은 요약(Abstract) :



이 논문은 사기 탐지 및 의료 모니터링을 위한 예측 모델링에서의 문제를 다루고 있습니다. 기존의 데이터 기반 접근 방식은 역사적 데이터에서 상관관계를 학습하지만, 이벤트 관계를 지배하는 도메인 특정 순차 제약 및 논리 규칙을 통합하지 못해 정확성과 규제 준수에 한계를 보입니다. 예를 들어, 의료 절차는 특정 순서를 따라야 하고, 금융 거래는 준수 규칙을 따라야 합니다. 저자들은 Logic Tensor Networks (LTNs)를 사용하여 도메인 지식을 미분 가능한 논리 제약으로 통합하는 신경-상징적 접근 방식을 제안합니다. 이들은 제어 흐름, 시간적, 그리고 페이로드 지식을 선형 시간 논리 및 1차 논리를 사용하여 형식화합니다. 주요 기여는 LTNs가 논리 공식을 만족시키는 경향이 예측 정확성을 희생하는 문제를 해결하기 위한 두 단계 최적화 전략을 제안하는 것입니다. 이 접근 방식은 사전 훈련 중 가중된 공리 손실을 사용하여 데이터 학습을 우선시하고, 만족 동역학에 따라 일관되고 기여하는 공리만을 유지하는 규칙 가지치기를 수행합니다. 네 가지 실제 이벤트 로그에 대한 평가 결과, 도메인 지식의 주입이 예측 성능을 크게 향상시키며, 두 단계 최적화가 필수적임을 보여줍니다. 이 방법은 특히 제한된 준수 훈련 예제가 있는 준수 제약 시나리오에서 뛰어난 성능을 발휘하며, 순수 데이터 기반 기준보다 우수한 성능을 보장합니다.



This paper addresses the challenges in predictive modeling for fraud detection and healthcare monitoring. Existing data-driven approaches learn correlations from historical data but fail to incorporate domain-specific sequential constraints and logical rules governing event relationships, limiting accuracy and regulatory compliance. For instance, healthcare procedures must follow specific sequences, and financial transactions must adhere to compliance rules. The authors present a neuro-symbolic approach that integrates domain knowledge as differentiable logical constraints using Logic Tensor Networks (LTNs). They formalize control-flow, temporal, and payload knowledge using Linear Temporal Logic and first-order logic. The key contribution is a two-stage optimization strategy that addresses the tendency of LTNs to satisfy logical formulas at the expense of predictive accuracy. This approach uses weighted axiom loss during pretraining to prioritize data learning, followed by rule pruning that retains only consistent, contributive axioms based on satisfaction dynamics. Evaluation on four real-world event logs shows that domain knowledge injection significantly improves predictive performance, with the two-stage optimization proving essential. The approach excels particularly in compliance-constrained scenarios with limited compliant training examples, achieving superior performance compared to purely data-driven baselines while ensuring adherence to domain constraints.


* Useful sentences :

기호들을 사용해서 로직으로 일종의 매핑?  (사람이 이해하도록)  


{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology



이 논문에서 제안하는 방법은 예측 프로세스 모니터링을 위한 신경-기호적(neuro-symbolic) 접근 방식으로, 도메인 지식을 차별화 가능한 논리적 제약으로 통합하는 Logic Tensor Networks (LTNs)를 활용합니다. 이 방법은 세 가지 주요 도전 과제를 해결합니다: (1) 다양한 프로세스 제약을 논리적 표현으로 형식화하는 방법, (2) 도메인 규칙과 역사적 데이터 패턴 간의 충돌을 처리하는 방법, (3) 데이터와 지식 제약의 영향을 균형 있게 유지하면서 훈련 중 규칙의 신뢰성을 유지하는 방법입니다.

#### 모델 아키텍처
제안된 모델은 두 단계의 최적화 전략을 사용합니다. 첫 번째 단계에서는 가중치가 부여된 공리 손실(weighted axiom loss)을 통해 데이터 학습을 우선시합니다. 이 단계에서 모델은 데이터 축적을 통해 예측 정확도를 높이는 데 집중합니다. 두 번째 단계에서는 규칙 가지치기(rule pruning) 메커니즘을 통해 비일관적이거나 기여하지 않는 규칙을 필터링하여 정제된 지식 기반을 생성합니다. 이 과정은 만족도 동역학(satisfaction dynamics)을 기반으로 하여, 유용한 규칙만을 남기고 나머지는 제거합니다.

#### 훈련 데이터 및 기법
훈련 데이터는 실제 이벤트 로그에서 추출된 것으로, 세 가지 주요 특징 범주(제어 흐름, 시간적 관계, 페이로드)를 포함합니다. 제어 흐름 특징은 순차적 구조를 나타내며, 시간적 특징은 시간 기반 관계를 나타냅니다. 페이로드 특징은 사건 수준 및 케이스 수준의 속성 데이터를 포함합니다. 이러한 특징들은 Linear Temporal Logic (LTL) 및 일차 논리(first-order logic)로 형식화되어 LTN에 통합됩니다.

훈련 과정은 두 단계로 나뉘며, 첫 번째 단계에서는 데이터와 지식 공리를 모두 고려하여 모델을 훈련합니다. 두 번째 단계에서는 정제된 지식 기반을 사용하여 모델을 미세 조정(fine-tuning)합니다. 이 과정에서 하이퍼파라미터(α, β, λ 등)를 조정하여 최적의 성능을 달성합니다.

이러한 접근 방식은 특히 규정 준수가 중요한 시나리오에서 효과적이며, 제한된 훈련 예제에서도 높은 예측 성능을 유지합니다. 결과적으로, 이 방법은 도메인 지식의 주입이 예측 성능을 향상시키고, 두 단계 최적화가 필수적임을 보여줍니다.

---




The method proposed in this paper is a neuro-symbolic approach for predictive process monitoring that utilizes Logic Tensor Networks (LTNs) to integrate domain knowledge as differentiable logical constraints. This approach addresses three key challenges: (1) how to formalize diverse types of process constraints using logical representations, (2) how to handle conflicts between domain rules and historical data patterns, and (3) how to balance the influence of data and knowledge constraints while maintaining rule reliability during training.

#### Model Architecture
The proposed model employs a two-stage optimization strategy. In the first stage, a weighted axiom loss is used to prioritize data learning. During this phase, the model focuses on enhancing predictive accuracy through data accumulation. The second stage involves a rule pruning mechanism that filters out inconsistent or non-contributive rules, resulting in a refined knowledge base. This process is based on satisfaction dynamics, retaining only useful rules while discarding the rest.

#### Training Data and Techniques
The training data is extracted from real-world event logs and includes three main feature categories: control-flow, temporal relationships, and payload. Control-flow features represent sequential structures, while temporal features indicate time-based relationships. Payload features encompass case-level and event-level attribute data. These features are formalized using Linear Temporal Logic (LTL) and first-order logic for integration into the LTN.

The training process is divided into two stages. In the first stage, the model is trained considering both data and knowledge axioms. In the second stage, the model is fine-tuned using the refined knowledge base. During this process, hyperparameters (α, β, λ, etc.) are adjusted to achieve optimal performance.

This approach is particularly effective in compliance-critical scenarios, maintaining high predictive performance even with limited training examples. Ultimately, the method demonstrates that the injection of domain knowledge improves predictive performance and that the two-stage optimization is essential.


<br/>
# Results



이 논문에서는 예측 프로세스 모니터링을 위한 신경-기호적 접근 방식을 제안하고, 이를 통해 도메인 지식을 통합하여 예측 성능을 향상시키는 방법을 설명합니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서는 LSTM 및 Transformer와 같은 순수 데이터 기반 모델을 비교 대상으로 설정했습니다. 이들 모델은 도메인 지식 없이 데이터 축적만으로 학습합니다. 또한, LTN-Data 및 LTN-NoP와 같은 LTN 기반 모델도 비교했습니다. LTN-Data는 데이터 축적만을 사용하고, LTN-NoP는 지식 축적을 포함하지만 규칙 가지치기를 적용하지 않았습니다.

2. **테스트 데이터**: 연구에서는 두 가지 평가 설정을 사용했습니다. 첫 번째는 일반적인 시간 분할 방식으로, 과거 데이터를 기반으로 미래 사건을 예측하는 방식입니다. 두 번째는 규정 준수 인식 테스트 세트로, 이 세트는 규칙을 준수하는 사례와 무작위 샘플을 결합하여 모델이 규정 준수 행동을 일반화할 수 있는지를 평가합니다.

3. **메트릭**: 성능 평가는 정확도(Accuracy)와 F1 점수(F1 Score)를 사용하여 측정되었습니다. F1 점수는 정밀도와 재현율의 조화 평균으로, 불균형 데이터셋에서 모델의 성능을 평가하는 데 유용합니다.

4. **비교 결과**: 
   - **RQ1 (도메인 지식 주입의 영향)**: Two-Stage-L 및 Two-Stage-T 모델은 모든 데이터셋에서 순수 신경망 모델보다 일관되게 높은 성능을 보였습니다. 특히, Sepsis 데이터셋에서는 LSTM에서 Two-Stage-L로의 F1 점수가 5.23% 향상되었습니다. 
   - **RQ2 (두 단계 최적화의 기여)**: LTN-NoP 모델은 성능이 크게 저하되었으며, 이는 지식 축적이 제대로 관리되지 않았기 때문입니다. 반면, Two-Stage 모델은 F1 점수가 70.59%로 회복되었습니다. 
   - **RQ3 (규정 준수 제약에 대한 일반화)**: Two-Stage 모델은 규정 준수 인식 테스트 세트에서 LSTM보다 16.7% 높은 F1 점수를 기록했습니다. 이는 도메인 지식이 부족한 경우에도 모델이 규정 준수 패턴을 잘 일반화할 수 있음을 보여줍니다.

이러한 결과는 도메인 지식이 예측 성능을 향상시키는 데 중요한 역할을 하며, 두 단계 최적화가 지식 주입의 부정적인 영향을 방지하는 데 필수적임을 강조합니다.

---




This paper presents a neuro-symbolic approach for predictive process monitoring, explaining how to integrate domain knowledge to enhance predictive performance. The key findings of the study are as follows:

1. **Competing Models**: The study compares purely data-driven models such as LSTM and Transformer. These models learn solely from data accumulation without incorporating domain knowledge. Additionally, LTN-based models like LTN-Data and LTN-NoP are compared. LTN-Data uses only data axioms, while LTN-NoP includes knowledge axioms but does not apply rule pruning.

2. **Test Data**: The study employs two evaluation settings. The first is a standard temporal split, where past data is used to predict future events. The second is a compliance-aware test set, which combines rule-compliant cases with random samples to assess the model's ability to generalize compliance behavior.

3. **Metrics**: Performance is measured using accuracy and F1 score. The F1 score, which is the harmonic mean of precision and recall, is particularly useful for evaluating model performance on imbalanced datasets.

4. **Comparison Results**: 
   - **RQ1 (Impact of Domain Knowledge Injection)**: The Two-Stage-L and Two-Stage-T models consistently outperformed their purely neural counterparts across all datasets. Notably, in the Sepsis dataset, the F1 score improved by 5.23% from LSTM to Two-Stage-L.
   - **RQ2 (Contribution of Two-Stage Optimization)**: The LTN-NoP model showed a significant performance drop, indicating that knowledge injection without proper control can degrade performance. In contrast, the Two-Stage model recovered to an F1 score of 70.59%.
   - **RQ3 (Generalization to Compliance Constraints)**: The Two-Stage models achieved an F1 score that was 16.7% higher than LSTM on the compliance-aware test set, demonstrating that domain knowledge helps the model generalize compliance patterns even when training examples are scarce.

These results highlight the critical role of domain knowledge in improving predictive performance and emphasize that the two-stage optimization is essential to mitigate the negative effects of knowledge injection.


<br/>
# 예제



이 논문에서는 예측 프로세스 모니터링을 위한 신경-기호적 접근 방식을 제안합니다. 이 접근 방식은 도메인 지식을 차별화 가능한 논리적 제약으로 통합하여 예측 성능을 향상시키는 것을 목표로 합니다. 연구에서는 두 가지 주요 단계로 구성된 최적화 전략을 사용하여, 데이터 학습과 논리적 제약 간의 균형을 유지합니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 의료 프로세스에서 수집된 이벤트 로그. 각 이벤트는 다음과 같은 속성을 가집니다:
     - 활동 (예: "수술", "항생제 투여")
     - 사례 식별자 (예: 환자 ID)
     - 타임스탬프 (예: 이벤트 발생 시간)
     - 속성 (예: 환자의 나이, 산소 포화도 등)
   - **출력**: 각 이벤트 시퀀스에 대한 예측 결과 (예: "합병증 발생" 또는 "합병증 미발생").

   예를 들어, 트레이닝 데이터의 한 샘플은 다음과 같습니다:
   - 입력: [("환자 입원", "P123", "2023-01-01 10:00"), ("수술", "P123", "2023-01-01 12:00"), ("항생제 투여", "P123", "2023-01-01 12:30")]
   - 출력: "합병증 발생" (환자가 수술 후 2시간 이내에 항생제를 투여받지 않았을 경우).

2. **테스트 데이터**:
   - **입력**: 새로운 환자에 대한 이벤트 로그. 이 데이터는 트레이닝 데이터와 유사하지만, 합병증 발생 여부가 포함되어 있지 않습니다.
   - **출력**: 모델이 예측한 결과 (예: "합병증 발생" 또는 "합병증 미발생").

   예를 들어, 테스트 데이터의 한 샘플은 다음과 같습니다:
   - 입력: [("환자 입원", "P456", "2023-01-02 09:00"), ("수술", "P456", "2023-01-02 11:00"), ("항생제 투여", "P456", "2023-01-02 11:30")]
   - 출력: 모델의 예측 결과 (예: "합병증 미발생").

이러한 방식으로, 연구는 도메인 지식을 통합하여 예측 성능을 향상시키고, 특히 제한된 훈련 데이터에서의 일반화 능력을 평가합니다.

---



This paper proposes a neuro-symbolic approach for predictive process monitoring, aiming to enhance predictive performance by integrating domain knowledge as differentiable logical constraints. The study employs a two-stage optimization strategy to balance data learning and logical constraints.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: Event logs collected from a healthcare process. Each event has the following attributes:
     - Activity (e.g., "Surgery", "Antibiotic Administration")
     - Case Identifier (e.g., Patient ID)
     - Timestamp (e.g., time of event occurrence)
     - Attributes (e.g., patient age, oxygen saturation, etc.)
   - **Output**: Predicted outcomes for each event sequence (e.g., "Complication Occurred" or "No Complication").

   For example, a sample from the training data might look like this:
   - Input: [("Patient Admitted", "P123", "2023-01-01 10:00"), ("Surgery", "P123", "2023-01-01 12:00"), ("Antibiotic Administered", "P123", "2023-01-01 12:30")]
   - Output: "Complication Occurred" (if the patient did not receive antibiotics within 2 hours post-surgery).

2. **Test Data**:
   - **Input**: Event logs for new patients. This data is similar to the training data but does not include the outcome of complications.
   - **Output**: Predicted results from the model (e.g., "Complication Occurred" or "No Complication").

   For example, a sample from the test data might look like this:
   - Input: [("Patient Admitted", "P456", "2023-01-02 09:00"), ("Surgery", "P456", "2023-01-02 11:00"), ("Antibiotic Administered", "P456", "2023-01-02 11:30")]
   - Output: Model's prediction (e.g., "No Complication").

In this way, the study evaluates the integration of domain knowledge to improve predictive performance, particularly assessing generalization capabilities in scenarios with limited training data.

<br/>
# 요약


이 논문에서는 예측 프로세스 모니터링을 위한 신경-기호적 접근 방식을 제안하며, 도메인 지식을 차별화 가능한 논리적 제약으로 통합하는 두 단계 최적화 전략을 사용합니다. 실험 결과, 도메인 지식을 주입한 모델이 순수 데이터 기반 모델보다 예측 성능이 향상되었으며, 특히 제한된 훈련 데이터에서 두 단계 최적화가 필수적임을 보여주었습니다. 예를 들어, Sepsis 데이터셋에서 두 단계 접근 방식은 F1 점수에서 5.23%의 개선을 달성했습니다.

---

This paper presents a neuro-symbolic approach for predictive process monitoring, utilizing a two-stage optimization strategy that integrates domain knowledge as differentiable logical constraints. Experimental results show that models incorporating domain knowledge outperform purely data-driven models, particularly in scenarios with limited training data, highlighting the necessity of the two-stage optimization. For instance, on the Sepsis dataset, the two-stage approach achieved a 5.23% improvement in F1 score.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **파이프라인 다이어그램 (Fig. 2)**: 이 다이어그램은 제안된 접근 방식의 전체 프로세스를 시각적으로 나타냅니다. 이벤트 로그에서 특징과 규칙을 추출하고, 지식 기반을 구축한 후, 이를 LTN 기반의 이진 분류기에 주입하는 과정을 보여줍니다. 이 과정은 도메인 지식을 통합하여 예측 성능을 향상시키는 방법을 명확히 합니다.
   - **LTN 구조 (Fig. 1)**: 이 피규어는 LTN을 사용한 이진 분류의 구조를 설명합니다. 대출 신청서가 신경망을 통해 평가되어 승인 여부를 결정하는 과정을 보여줍니다. 이는 LTN의 작동 방식을 이해하는 데 도움을 줍니다.

2. **테이블**
   - **RQ1: 도메인 지식 주입의 영향 (Table 1)**: 이 테이블은 순수 데이터 기반 모델과 지식 강화 모델 간의 성능을 비교합니다. 결과는 도메인 지식이 예측 성능을 향상시키는 데 중요한 역할을 한다는 것을 보여줍니다. 특히 작은 데이터셋에서 성능 향상이 두드러지며, 이는 도메인 지식이 부족한 데이터에서 모델의 학습을 제약할 수 있음을 시사합니다.
   - **RQ2: 두 단계 최적화의 기여 (Table 2)**: 이 테이블은 LTN 변형 모델 간의 성능을 비교합니다. 지식 축적 없이 단순히 지식을 주입하는 것이 성능을 저하시킬 수 있음을 보여줍니다. 두 단계 최적화가 없을 경우, 모델이 의미 있는 예측 패턴을 학습하기보다는 논리적 단축을 이용할 수 있음을 강조합니다.
   - **RQ3: 준수 제약에 대한 일반화 (Table 3)**: 이 테이블은 준수 제약이 있는 테스트 세트에서 모델의 성능을 평가합니다. 두 단계 모델이 준수 제약이 있는 시나리오에서 성능이 크게 향상되었음을 보여줍니다. 이는 지식이 부족한 예제에서 중요한 유도 편향을 제공함을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스는 추가적인 실험 결과나 데이터, 알고리즘 세부사항 등을 포함할 수 있습니다. 이는 연구의 재현성을 높이고, 제안된 방법의 유효성을 검증하는 데 도움을 줍니다.

---




1. **Diagrams and Figures**
   - **Pipeline Diagram (Fig. 2)**: This diagram visually represents the entire process of the proposed approach. It shows the extraction of features and rules from event logs, the construction of a knowledge base, and the injection of this knowledge into an LTN-based binary classifier. This process clearly illustrates how domain knowledge is integrated to enhance predictive performance.
   - **LTN Structure (Fig. 1)**: This figure explains the structure of binary classification using LTN. It depicts how loan applications are evaluated by a neural network to determine approval. This helps in understanding the operational mechanism of LTN.

2. **Tables**
   - **RQ1: Impact of Domain Knowledge Injection (Table 1)**: This table compares the performance of purely data-driven models with knowledge-enhanced models. The results show that domain knowledge plays a crucial role in improving predictive performance, especially in smaller datasets, indicating that knowledge can constrain model learning when statistical evidence is insufficient.
   - **RQ2: Contribution of Two-Stage Optimization (Table 2)**: This table compares the performance of LTN variant models. It highlights that simply injecting knowledge without proper control can severely degrade performance. The absence of the two-stage optimization allows models to exploit logical shortcuts rather than learning meaningful predictive patterns.
   - **RQ3: Generalization to Compliance Constraints (Table 3)**: This table evaluates model performance on a compliance-aware test set. It shows that the two-stage model significantly outperforms baselines in compliance-constrained scenarios, indicating that logical rules provide crucial inductive bias when compliant examples are scarce.

3. **Appendix**
   - The appendix may include additional experimental results, data, or algorithmic details. This enhances the reproducibility of the research and helps validate the effectiveness of the proposed method.

<br/>
# refer format:




```bibtex  
@inproceedings{DeSantis2026,
  author    = {Fabrizio De Santis and Gyunam Park and Francesco Zanichelli},
  title     = {Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning},
  booktitle = {Proceedings of the 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2026)},
  pages     = {104--118},
  year      = {2026},
  publisher = {Springer},
  address   = {Singapore},
  doi       = {10.1007/978-981-92-1462-4_9}
}   
```   




Fabrizio De Santis, Gyunam Park, and Francesco Zanichelli. 2026. "Neuro-Symbolic Learning for Predictive Process Monitoring via Two-Stage Logic Tensor Networks with Rule Pruning." In *Proceedings of the 29th Pacific-Asia Conference on Knowledge Discovery and Data Mining (PAKDD 2026)*, 104-118. Singapore: Springer. https://doi.org/10.1007/978-981-92-1462-4_9.   
