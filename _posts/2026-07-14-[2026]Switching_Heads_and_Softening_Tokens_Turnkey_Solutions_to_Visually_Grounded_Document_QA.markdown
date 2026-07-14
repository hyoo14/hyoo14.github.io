---
layout: post
title:  "[2026]Switching Heads and Softening Tokens: Turnkey Solutions to Visually Grounded Document QA"
date:   2026-07-14 00:55:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 시각적으로 기반한 문서 질문 응답(VGDQA)을 위한 두 가지 아키텍처, 단일 헤드 모델과 이중 헤드 모델을 제안합니다.
(일종의 OCR을 위한, OCR스러운 방법임.. 바운딩박스를 비롯한  )  

짧은 요약(Abstract) :


이 논문의 초록에서는 시각적으로 기반한 문서 질문 응답(Visually Grounded Document Question Answering, VGDQA) 시스템의 한계와 이를 해결하기 위한 두 가지 턴키 LLM 아키텍처를 제안하고 있습니다. 첫 번째로, 단일 헤드 아키텍처를 소개하며, 이 아키텍처는 좌표를 통합된 어휘 내의 특수 토큰으로 표현합니다. 그러나 이 방법은 이산적인 감독의 한계로 인해 공간적 정밀도가 떨어집니다. 이를 해결하기 위해 "소프트 토큰" 방법을 제안하여 토큰 확률에 대한 미분 가능한 평균 제곱 오차 손실을 적용합니다. 두 번째로, 이중 헤드 아키텍처를 제안하여 텍스트 생성과 회귀 기반 경계 상자 예측을 번갈아 수행합니다. 이 방법은 회귀 헤드를 통해 높은 공간적 정밀도를 제공하며, 교차 영역 비율 손실을 도입하여 안정성을 높입니다. 마지막으로, 단일 헤드 모델의 구조적 강인성과 이중 헤드 모델의 높은 정밀도를 결합한 앙상블 방법을 제안하여 각 구성 요소의 성능을 초과하는 성능 향상을 달성합니다.



The abstract of this paper discusses the limitations of visually grounded document question answering (VGDQA) systems and proposes two turnkey LLM architectures to address these issues. First, a single-head architecture is introduced, which represents coordinates as special tokens within a unified vocabulary. However, this approach suffers from the limitations of discrete supervision, leading to reduced spatial precision. To overcome this, a "softening token" method is proposed that enables differentiable Mean-Squared-Error loss over token probabilities. Second, a dual-head architecture is proposed that alternates between text generation and regression-based bounding box prediction. This method offers high spatial precision through a regression head, further stabilized by the introduction of an Intersection-over-Union loss. Finally, an ensemble method is proposed that combines the structural robustness of the single-head model with the high precision of the dual-head model, achieving significant performance gains beyond each individual component.


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



이 논문에서는 시각적으로 기반한 문서 질문 응답(Visually Grounded Document Question Answering, VGDQA) 문제를 해결하기 위해 두 가지 주요 아키텍처를 제안합니다: 단일 헤드 아키텍처와 이중 헤드 아키텍처입니다. 이 두 아키텍처는 각각의 장점과 단점을 가지고 있으며, 최종적으로는 이들을 결합한 앙상블 방법을 통해 성능을 극대화합니다.

1. **단일 헤드 아키텍처**:
   - 이 아키텍처는 답변과 바운딩 박스 생성을 통합된 토큰 생성 작업으로 처리합니다. 이를 위해 1,001개의 특별한 토큰(<B0>부터 <B1000>까지)을 사용하여 바운딩 박스의 좌표를 표현합니다. 이 구조는 모델이 답변을 생성한 후 즉시 바운딩 박스를 생성할 수 있도록 하여 복잡한 아키텍처 변경 없이도 안정성을 제공합니다.
   - 그러나 이 아키텍처는 이산적인 감독의 한계로 인해 공간적 정밀도가 제한됩니다. 이를 해결하기 위해 "소프트 토큰" 방법을 도입하여, 토큰 확률에 대한 평균 제곱 오차(Mean-Squared-Error, MSE) 손실을 적용할 수 있도록 합니다. 이 방법은 공간적 근접성을 고려하여 모델의 성능을 향상시킵니다.

2. **이중 헤드 아키텍처**:
   - 이 아키텍처는 텍스트 생성과 바운딩 박스 예측을 위한 별도의 디코더 헤드를 사용합니다. 텍스트 헤드가 답변을 생성한 후, <SWITCH_HEAD>라는 제어 토큰을 통해 바운딩 박스 예측을 위한 박스 헤드로 전환됩니다. 이 구조는 바운딩 박스의 공간적 정밀도를 높이기 위해 회귀 기반의 예측을 사용합니다.
   - 이중 헤드 아키텍처는 IoU(Intersection-over-Union) 손실을 도입하여 작은 바운딩 박스의 정확성을 높이고, 최적화의 안정성을 보장합니다. 그러나 헤드 전환 메커니즘에 의존하기 때문에 잘못된 전환으로 인해 불필요한 답변이 생성될 수 있는 단점이 있습니다.

3. **앙상블 방법**:
   - 두 아키텍처의 장점을 결합하기 위해 앙상블 방법을 제안합니다. 단일 헤드 모델의 구조적 안정성을 활용하여 답변의 집합을 결정하고, 이중 헤드 모델의 출력을 사용하여 공간 좌표를 정제합니다. 이를 통해 각 개별 구성 요소의 성능을 초과하는 성과를 달성할 수 있습니다.

이러한 방법론은 OCR(Optical Character Recognition) 기술의 발전을 활용하여 문서의 시각적 정보를 효과적으로 처리하고, 복잡한 질문에 대한 정확한 답변을 제공하는 데 기여합니다.




In this paper, we propose two main architectures to address the problem of Visually Grounded Document Question Answering (VGDQA): a single-head architecture and a dual-head architecture. Each of these architectures has its own advantages and disadvantages, and ultimately, we combine them into an ensemble method to maximize performance.

1. **Single-Head Architecture**:
   - This architecture treats the generation of answers and bounding boxes as a unified token generation task. To achieve this, it utilizes 1,001 special tokens (from <B0> to <B1000>) to represent the coordinates of the bounding boxes. This structure allows the model to generate a bounding box immediately after producing an answer, providing stability without the need for complex architectural changes.
   - However, this architecture is limited by the constraints of discrete supervision, which caps its spatial precision. To address this, we introduce a "softening token" method that enables the application of Mean-Squared-Error (MSE) loss over token probabilities. This method enhances the model's performance by considering spatial proximity.

2. **Dual-Head Architecture**:
   - This architecture features separate decoder heads for text generation and bounding box prediction. After the text head generates an answer, it switches to the box head for bounding box prediction via a control token called <SWITCH_HEAD>. This structure uses regression-based predictions to enhance the spatial precision of the bounding boxes.
   - The dual-head architecture incorporates Intersection-over-Union (IoU) loss to improve accuracy for small bounding boxes and ensure optimization stability. However, it relies on the head-switching mechanism, which can lead to the generation of extraneous answers due to erroneous switching.

3. **Ensemble Method**:
   - To leverage the strengths of both architectures, we propose an ensemble method. This method uses the structural robustness of the single-head model to determine the set of answers while refining the spatial coordinates using the output from the dual-head model. This approach achieves performance gains that exceed those of each individual component.

These methodologies effectively utilize advancements in Optical Character Recognition (OCR) technology to process visual information from documents and provide accurate answers to complex queries.


<br/>
# Results



이 논문에서는 Visually Grounded Document Question Answering (VGDQA) 문제를 해결하기 위해 두 가지 아키텍처를 제안하고, 이를 기존의 경쟁 모델들과 비교하여 성능을 평가했습니다. 실험은 BoundingDocs v2.0 데이터셋을 사용하여 진행되었으며, 이 데이터셋은 다양한 문서 유형(영수증, 계약서 등)에서 질문-답변 쌍과 해당 답변의 경계 상자(bounding box) 정보를 포함하고 있습니다.

#### 경쟁 모델
제안된 모델은 DLaV A, DocExplainerV0, DOGR, LayTextLLM, Qwen3-VL-8B, InternVL3.5-8B와 같은 여러 최신 모델과 비교되었습니다. 이들 모델은 VGDQA 문제를 해결하기 위해 설계되었지만, 제안된 모델에 비해 성능이 떨어졌습니다. 특히, DLaV A와 DocExplainerV0는 end-to-end 최적화를 제공하지 않아 낮은 정확도를 보였고, LayTextLLM은 경계 상자 예측을 위한 훈련 목표가 부족하여 grounding accuracy가 0%에 달했습니다.

#### 테스트 데이터
BoundingDocs v2.0 데이터셋은 48,151개의 문서와 249,016개의 질문-답변 쌍으로 구성되어 있으며, 각 질문에 대해 정답과 그에 해당하는 경계 상자 정보가 제공됩니다. 이 데이터셋은 VGDQA 모델의 성능을 평가하기 위한 강력한 벤치마크로 사용되었습니다.

#### 메트릭
모델의 성능은 Grounding Accuracy (Acc)와 Normalized Cardinality Error (NCE)로 평가되었습니다. Grounding Accuracy는 생성된 텍스트가 정답과 일치하고, 예측된 경계 상자가 실제 경계 상자와의 IoU(Intersection over Union)가 0.5를 초과할 때 정확하다고 간주됩니다. NCE는 생성된 답변의 수가 얼마나 정확한지를 평가하는 지표로, 추가된 답변 수와 누락된 답변 수를 기반으로 계산됩니다.

#### 비교 결과
제안된 단일 헤드(single-head) 모델은 0.796의 Grounding Accuracy를 기록했으며, 이중 헤드(dual-head) 모델은 0.854로 더 높은 정확도를 보였습니다. 앙상블 방법은 두 모델의 장점을 결합하여 0.866의 최고 성능을 달성했습니다. 반면, DLaV A와 DocExplainerV0는 각각 0.168과 0.030의 낮은 정확도를 기록했습니다. LayTextLLM은 0%의 Grounding Accuracy를 보였으며, 일반적인 비전-언어 모델인 Qwen3-VL-8B와 InternVL3.5-8B는 각각 0.252와 0.001의 낮은 성능을 보였습니다.

이러한 결과는 제안된 모델이 기존의 경쟁 모델들에 비해 VGDQA 문제를 해결하는 데 있어 더 효과적임을 보여줍니다. 특히, 앙상블 방법은 구조적 안정성과 공간적 정밀성을 모두 갖춘 최적의 솔루션으로 평가되었습니다.

---




In this paper, two architectures are proposed to address the Visually Grounded Document Question Answering (VGDQA) problem, and their performance is evaluated against existing competitive models. The experiments were conducted using the BoundingDocs v2.0 dataset, which includes question-answer pairs and corresponding bounding box information from various document types (invoices, contracts, etc.).

#### Competitive Models
The proposed models were compared with several state-of-the-art models, including DLaV A, DocExplainerV0, DOGR, LayTextLLM, Qwen3-VL-8B, and InternVL3.5-8B. While these models were designed to tackle the VGDQA problem, they generally underperformed compared to the proposed models. Notably, DLaV A and DocExplainerV0 exhibited low accuracy due to their lack of end-to-end optimization, while LayTextLLM achieved a grounding accuracy of 0% due to insufficient training objectives for bounding box prediction.

#### Test Data
The BoundingDocs v2.0 dataset consists of 48,151 documents and 249,016 question-answer pairs, with each question accompanied by the correct answer and its corresponding bounding box information. This dataset serves as a robust benchmark for evaluating the performance of VGDQA models.

#### Metrics
The performance of the models was evaluated using Grounding Accuracy (Acc) and Normalized Cardinality Error (NCE). Grounding Accuracy is considered accurate when the generated text matches the ground truth and the predicted bounding box has an Intersection over Union (IoU) exceeding 0.5. NCE assesses the accuracy of the number of generated answers based on the count of extra and missing answers.

#### Comparison Results
The proposed single-head model achieved a Grounding Accuracy of 0.796, while the dual-head model performed even better with an accuracy of 0.854. The ensemble method, which combines the strengths of both models, reached the highest performance of 0.866. In contrast, DLaV A and DocExplainerV0 recorded low accuracies of 0.168 and 0.030, respectively. LayTextLLM showed a grounding accuracy of 0%, and general vision-language models like Qwen3-VL-8B and InternVL3.5-8B had low performances of 0.252 and 0.001, respectively.

These results demonstrate that the proposed models are more effective in addressing the VGDQA problem compared to existing competitive models. In particular, the ensemble method is evaluated as the optimal solution, combining structural robustness with spatial precision.


<br/>
# 예제



이 논문에서는 시각적으로 기반한 문서 질문 응답(VGDQA) 시스템을 위한 두 가지 아키텍처를 제안합니다. 이 시스템은 복잡한 질문에 대한 답변을 시각적으로 정확하게 찾는 것을 목표로 합니다. 아래는 트레이닝 데이터와 테스트 데이터의 구체적인 입력 및 출력 예시와 함께 이 시스템의 작동 방식을 설명합니다.

#### 트레이닝 데이터 예시
- **입력**: 
  - 문서 이미지: 예를 들어, 항공권 예약 확인서의 이미지.
  - 질문: "내 비행기의 출발 날짜와 시간을 알려줘."
  
- **출력**: 
  - 답변: "2023년 5월 10일, 오전 10시 30분."
  - 바운딩 박스: [x1, y1, x2, y2] 형식으로, 답변이 위치한 문서 내의 좌표. 예를 들어, [0.1, 0.2, 0.3, 0.4]는 문서의 특정 영역을 나타냅니다.

#### 테스트 데이터 예시
- **입력**: 
  - 문서 이미지: 예를 들어, 청구서의 이미지.
  - 질문: "청구서의 총 금액은 얼마인가?"
  
- **출력**: 
  - 답변: "150,000원."
  - 바운딩 박스: [0.5, 0.6, 0.7, 0.8] 형식으로, 답변이 위치한 문서 내의 좌표.

이 시스템은 두 가지 아키텍처를 사용합니다. 첫 번째는 단일 헤드 아키텍처로, 답변과 바운딩 박스를 동시에 생성합니다. 두 번째는 이중 헤드 아키텍처로, 텍스트 생성과 바운딩 박스 예측을 분리하여 수행합니다. 이 두 아키텍처는 앙상블 방식으로 결합되어 최종 출력을 생성합니다. 이 과정에서 각 아키텍처의 장점을 활용하여 더 높은 정확도를 달성합니다.




This paper proposes two architectures for a visually grounded document question answering (VGDQA) system, aimed at accurately finding visual answers to complex questions. Below is a detailed explanation of the input and output examples for training and testing data, along with how the system operates.

#### Training Data Example
- **Input**: 
  - Document Image: For instance, an image of a flight reservation confirmation.
  - Question: "What is the departure date and time for my flight?"
  
- **Output**: 
  - Answer: "May 10, 2023, at 10:30 AM."
  - Bounding Box: Coordinates in the format [x1, y1, x2, y2], indicating the area in the document where the answer is located. For example, [0.1, 0.2, 0.3, 0.4] represents a specific region in the document.

#### Testing Data Example
- **Input**: 
  - Document Image: For instance, an image of a bill.
  - Question: "What is the total amount on the bill?"
  
- **Output**: 
  - Answer: "150,000 won."
  - Bounding Box: Coordinates in the format [0.5, 0.6, 0.7, 0.8], indicating the area in the document where the answer is located.

The system employs two architectures. The first is a single-head architecture that generates both the answer and the bounding box simultaneously. The second is a dual-head architecture that separates text generation and bounding box prediction. These two architectures are combined using an ensemble method to produce the final output. This process leverages the strengths of each architecture to achieve higher accuracy.

<br/>
# 요약

이 논문에서는 시각적으로 기반한 문서 질문 응답(VGDQA)을 위한 두 가지 아키텍처, 단일 헤드 모델과 이중 헤드 모델을 제안합니다. 단일 헤드 모델은 특수 토큰을 사용하여 구조적 안정성을 제공하고, 이중 헤드 모델은 회귀 기반의 경계 상자 예측을 통해 높은 공간 정밀도를 달성합니다. 최종적으로, 두 모델의 장점을 결합한 앙상블 방법이 기존 방법들보다 우수한 성능을 보였습니다.

---

This paper proposes two architectures for visually grounded document question answering (VGDQA): a single-head model and a dual-head model. The single-head model offers structural stability using special tokens, while the dual-head model achieves high spatial precision through regression-based bounding box prediction. Ultimately, an ensemble method that combines the strengths of both models outperforms existing approaches.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **다이어그램 (Figure 1)**: 단일 헤드 솔루션과 이중 헤드 솔루션의 구조적 차이를 보여줍니다. 단일 헤드 솔루션은 텍스트와 바운딩 박스 생성을 통합하여 간단한 구조를 유지하는 반면, 이중 헤드 솔루션은 텍스트 헤드와 박스 헤드로 나뉘어 각각의 작업을 수행합니다. 이 구조적 차이는 각 모델의 성능에 영향을 미치며, 단일 헤드 모델은 구조적 일관성을, 이중 헤드 모델은 공간적 정밀성을 제공합니다.

2. **질적 결과 (Figures 2 & 3)**: 두 개의 질문에 대한 모델의 응답과 시각적 기초를 비교합니다. 단일 헤드 모델은 구조적 일관성을 보여주며, 이중 헤드 모델은 더 정확한 바운딩 박스를 생성하지만, 헤드 전환 메커니즘으로 인해 잘못된 응답이 발생할 수 있습니다. 앙상블 방법은 두 모델의 장점을 결합하여 더 나은 성능을 보여줍니다.

#### 테이블
1. **정량적 결과 (Table 1)**: BoundingDocs v2.0 데이터셋에서의 성능을 비교합니다. 앙상블 방법이 가장 높은 정확도(Acc=0.866)를 기록하며, 단일 헤드 모델과 이중 헤드 모델의 장점을 결합하여 기존 방법들보다 우수한 성능을 보여줍니다. 기존 방법들은 전반적으로 성능이 낮아, 특히 DLaV A와 DocExplainerV0는 0.17 이하의 정확도를 기록했습니다.

2. **소거 연구 (Table 2)**: 각 모델의 구성 요소가 성능에 미치는 영향을 분석합니다. 단일 헤드 모델에서 Softening Token을 제거하면 정확도가 감소하고, 이중 헤드 모델에서 IoU 손실을 제거하면 성능이 급격히 저하됩니다. 이는 두 손실 함수가 모델의 성능에 필수적임을 보여줍니다.

3. **추가 성능 분석 (Table 4)**: 긴 문서와 짧은 문서에서의 성능을 비교합니다. 긴 문서에서 성능이 저하되는 경향이 있으며, 이는 긴 입력 시퀀스에 대한 주의 집중의 어려움 때문입니다.

#### 어펜딕스
- **구현 세부사항 (A.1)**: 모델의 훈련 하이퍼파라미터와 LoRA 구성에 대한 정보를 제공합니다. 이는 모델의 효율적인 훈련을 위한 중요한 요소입니다.
- **질적 결과 (A.2)**: 다양한 방법의 응답과 시각적 기초를 비교하여 각 모델의 강점과 약점을 시각적으로 보여줍니다.
- **추론 비용 (A.3)**: 각 모델의 추론 시간과 FLOPs를 비교하여 성능과 효율성을 평가합니다. 이중 헤드 모델이 단일 헤드 모델보다 빠르며, 앙상블 방법은 성능 향상을 위해 더 많은 비용이 소요됩니다.

---




#### Diagrams and Figures
1. **Diagram (Figure 1)**: Illustrates the structural differences between the single-head and dual-head solutions. The single-head solution maintains a simple structure by integrating text and bounding box generation, while the dual-head solution bifurcates into a text head and a box head, each performing distinct tasks. This structural difference impacts the performance of each model, with the single-head model providing structural consistency and the dual-head model offering spatial precision.

2. **Qualitative Results (Figures 2 & 3)**: Compares the model's responses and visual grounding for two questions. The single-head model demonstrates structural consistency, while the dual-head model generates more accurate bounding boxes but may produce incorrect responses due to the head-switching mechanism. The ensemble method combines the strengths of both models, resulting in improved performance.

#### Tables
1. **Quantitative Results (Table 1)**: Compares performance on the BoundingDocs v2.0 dataset. The ensemble method achieves the highest accuracy (Acc=0.866), demonstrating superior performance compared to existing methods by leveraging the strengths of both the single-head and dual-head models. Existing methods generally underperform, with DLaV A and DocExplainerV0 recording accuracies below 0.17.

2. **Ablation Study (Table 2)**: Analyzes the impact of each component on model performance. Removing the Softening Token from the single-head model leads to a decrease in accuracy, while removing the IoU loss from the dual-head model results in a drastic drop in performance. This indicates that both loss functions are essential for the model's effectiveness.

3. **Additional Performance Analysis (Table 4)**: Compares performance on long and short documents. There is a tendency for performance to degrade on longer documents, attributed to the difficulty of maintaining attention over extended input sequences.

#### Appendix
- **Implementation Details (A.1)**: Provides information on training hyperparameters and LoRA configuration, which are crucial for efficient model training.
- **Qualitative Results (A.2)**: Visually compares the responses and visual grounding of different methods, highlighting the strengths and weaknesses of each model.
- **Inference Cost (A.3)**: Compares the inference time and FLOPs of each model to evaluate performance and efficiency. The dual-head model is faster than the single-head model, while the ensemble method incurs higher costs for performance gains.

<br/>
# refer format:
### BibTeX Citation

```bibtex
@inproceedings{wen2026switching,
  title={Switching Heads and Softening Tokens: Turnkey Solutions to Visually Grounded Document QA},
  author={Ximing Wen and Wenbo Li and Sudipta Paul and Yashas Malur and Saidutta Kalpa Gunaratna and Srinivas Chappidi},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2026},
  pages={36490--36503},
  year={2026},
  month={July},
  publisher={Association for Computational Linguistics},


}
```

### Chicago Style Citation

Wen, Ximing, Wenbo Li, Sudipta Paul, Yashas Malur, Saidutta Kalpa Gunaratna, and Srinivas Chappidi. "Switching Heads and Softening Tokens: Turnkey Solutions to Visually Grounded Document QA." In *Findings of the Association for Computational Linguistics: ACL 2026*, 36490–36503. Association for Computational Linguistics, July 2026.
    