---
layout: post
title:  "[2025]PICLe: Pseudo-Annotations for In-Context Learning in Low-Resource Named Entity Detection"
date:   2025-08-15 15:47:09 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


노이즈를 넣은 경우 더 좋은 효과  


짧은 요약(Abstract) :


PICLe는 저자원 환경에서의 Named Entity Detection(NED)을 위한 새로운 프레임워크로, LLM을 활용하여 노이즈가 있는 의사 주석을 생성하고 이를 클러스터링하여 인컨텍스트 학습을 수행한다. 실험 결과, PICLe는 인간 주석 없이도 기존의 gold-labeled demonstration을 사용하는 ICL보다 더 나은 성능을 보였다. 특히, 부분적으로 올바른 주석이 포함된 경우에도 효과적인 성능을 유지하는 것으로 나타났다.




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


PICLe는 저자원 환경에서의 Named Entity Detection(NED)을 위한 새로운 프레임워크로, LLM을 활용하여 노이즈가 있는 의사 주석을 생성하고 이를 클러스터링하여 인컨텍스트 학습을 수행한다. 실험 결과, PICLe는 인간 주석 없이도 기존의 gold-labeled demonstration을 사용하는 ICL보다 더 나은 성능을 보였다. 특히, 부분적으로 올바른 주석이 포함된 경우에도 효과적인 성능을 유지하는 것으로 나타났다.

노이즈를 넣은 경우 더 좋은 효과  



<br/>
# Results


이 논문에서는 저자들이 제안한 PICLe(의사 주석을 통한 인컨텍스트 학습) 프레임워크의 성능을 평가하기 위해 여러 실험을 수행했습니다. PICLe는 저자들이 제안한 방법으로, 저자들은 이 방법이 기존의 인컨텍스트 학습(ICL) 방식보다 더 효과적일 수 있다고 주장합니다. 

#### 경쟁 모델
PICLe는 세 가지 대조 모델과 비교되었습니다:
1. **Zero-shot 모델**: 주어진 데이터에 대해 사전 학습된 모델을 사용하여 주석 없이 예측을 수행합니다.
2. **10-shot ICL**: 10개의 금본위 주석을 사용하여 인컨텍스트 학습을 수행합니다.
3. **100-shot ICL**: 100개의 금본위 주석을 사용하여 인컨텍스트 학습을 수행합니다.

#### 테스트 데이터
저자들은 다섯 개의 생물 의학 NED(명명된 개체 탐지) 데이터셋을 사용하여 PICLe의 성능을 평가했습니다. 이 데이터셋은 ChemProt, BC5CDR, BC2GM 등으로 구성되어 있으며, 각 데이터셋은 다양한 화학 물질, 질병 및 유전자에 대한 주석을 포함하고 있습니다.

#### 메트릭
성능 평가는 마이크로 평균 정밀도, 재현율 및 F1 점수를 사용하여 측정되었습니다. 이 메트릭은 모델이 얼마나 정확하게 화학 개체를 탐지하는지를 평가하는 데 사용됩니다.

#### 비교 결과
PICLe는 zero-shot 모델보다 평균 10.7% 더 높은 F1 점수를 기록했습니다. 또한, PICLe는 100개의 금본위 주석을 사용하는 ICL보다도 더 나은 성능을 보였습니다. 특히, BC5-Chem 데이터셋에서 PICLe는 77.7%의 평균 F1 점수를 기록하여 가장 높은 성능을 보였습니다. 반면, BC2GM 데이터셋에서는 50%의 F1 점수를 기록하여 상대적으로 낮은 성능을 보였습니다. 

이러한 결과는 PICLe가 저자들이 주장한 대로, 금본위 주석이 부족한 상황에서도 효과적으로 작동할 수 있음을 보여줍니다. PICLe는 저자들이 제안한 대로, 인컨텍스트 학습에서 부분적으로 올바른 주석이 금본위 주석과 유사한 성능을 발휘할 수 있음을 입증했습니다.





In this paper, the authors conducted several experiments to evaluate the performance of their proposed framework, PICLe (Pseudo-Annotations for In-Context Learning). They argue that this method could be more effective than traditional In-Context Learning (ICL) approaches.

#### Competing Models
PICLe was compared against three contrasting models:
1. **Zero-shot Model**: This model uses a pre-trained model to make predictions without any annotations on the given data.
2. **10-shot ICL**: This model performs In-Context Learning using 10 gold-standard annotations.
3. **100-shot ICL**: This model performs In-Context Learning using 100 gold-standard annotations.

#### Test Data
The authors evaluated the performance of PICLe using five biomedical Named Entity Detection (NED) datasets. These datasets include ChemProt, BC5CDR, and BC2GM, each containing annotations for various chemical substances, diseases, and genes.

#### Metrics
Performance was measured using micro-averaged precision, recall, and F1 scores. These metrics are used to assess how accurately the model detects chemical entities.

#### Comparison Results
PICLe achieved an average F1 score that was 10.7% higher than the zero-shot model. Additionally, PICLe outperformed the ICL using 100 gold-standard annotations. Notably, on the BC5-Chem dataset, PICLe recorded an average F1 score of 77.7%, marking the highest performance. In contrast, the BC2GM dataset yielded a lower performance with a 50% F1 score.

These results demonstrate that PICLe, as the authors claimed, can effectively operate even in scenarios where gold-standard annotations are scarce. PICLe also validated the authors' assertion that partially correct annotations can perform similarly to gold-standard annotations in In-Context Learning.


<br/>
# 예제


이 논문에서는 저자들이 제안한 PICLe(Pseudo-annotated In-Context Learning) 프레임워크를 사용하여 저자들이 연구한 화학 물질 인식(Named Entity Detection, NED) 작업을 수행하는 방법을 설명합니다. 이 작업은 주어진 텍스트에서 화학 물질을 식별하는 것입니다. 

#### 예시 설명

1. **트레이닝 데이터 예시**:
   - **입력**: "NRA0160은 인간 클론 도파민 D3 수용체에 대해 미미한 친화력을 가지고 있으며 (Ki = 39 nM), 쥐 세로토닌 (5-HT) 2A 수용체 (Ki = 180 nM) 및 쥐 알파1 아드레날린 수용체 (Ki = 237 nM)와도 마찬가지입니다."
   - **출력**: 
     ```
     NRA0160 | yes | because it is a chemical
     dopamine | yes | because it is a chemical
     serotonin | yes | because it is a chemical
     5-HT | yes | because it is a chemical
     ```

   이 예시에서 모델은 입력된 문장에서 화학 물질을 식별하고, 각 화학 물질에 대해 "yes" 또는 "no"로 응답하며, 그 이유를 설명합니다.

2. **테스트 데이터 예시**:
   - **입력**: "p75NTR 발현은 쥐의 요도 방광 감각 신경과 척수에서 cyclophosphamide 유도 방광염과 함께 나타납니다."
   - **출력**: 
     ```
     cyclophosphamide | yes | because it is a chemical
     ```

   이 테스트 예시에서 모델은 주어진 문장에서 "cyclophosphamide"라는 화학 물질을 식별하고, 그것이 화학 물질임을 확인합니다.

이러한 방식으로, PICLe 프레임워크는 주어진 문장에서 화학 물질을 효과적으로 추출하고, 이를 통해 NED 작업을 수행합니다. 이 과정에서 모델은 사전 훈련된 대형 언어 모델(LLM)을 사용하여 주어진 문맥에 따라 화학 물질을 식별하는 데 필요한 정보를 학습합니다.

---




In this paper, the authors describe how to perform Named Entity Detection (NED) tasks, specifically identifying chemical entities, using their proposed framework called PICLe (Pseudo-annotated In-Context Learning).

#### Example Explanation

1. **Training Data Example**:
   - **Input**: "NRA0160 has negligible affinity for the human cloned dopamine D3 receptor (Ki = 39 nM), rat serotonin (5-HT) 2A receptors (Ki = 180 nM), and rat alpha1 adrenoceptor (Ki = 237 nM)."
   - **Output**: 
     ```
     NRA0160 | yes | because it is a chemical
     dopamine | yes | because it is a chemical
     serotonin | yes | because it is a chemical
     5-HT | yes | because it is a chemical
     ```

   In this example, the model identifies chemical entities from the input text and responds with "yes" or "no" for each entity, providing a reason for its classification.

2. **Test Data Example**:
   - **Input**: "p75NTR expression in rat urinary bladder sensory neurons and spinal cord with cyclophosphamide-induced cystitis."
   - **Output**: 
     ```
     cyclophosphamide | yes | because it is a chemical
     ```

   In this test example, the model identifies the chemical entity "cyclophosphamide" from the given text and confirms that it is a chemical.

Through this process, the PICLe framework effectively extracts chemical entities from the provided text, enabling the execution of NED tasks. The model leverages a pre-trained large language model (LLM) to learn the necessary information for identifying chemical entities based on the given context.

<br/>
# 요약

PICLe는 저자원 환경에서의 Named Entity Detection(NED)을 위한 새로운 프레임워크로, LLM을 활용하여 노이즈가 있는 의사 주석을 생성하고 이를 클러스터링하여 인컨텍스트 학습을 수행한다. 실험 결과, PICLe는 인간 주석 없이도 기존의 gold-labeled demonstration을 사용하는 ICL보다 더 나은 성능을 보였다. 특히, 부분적으로 올바른 주석이 포함된 경우에도 효과적인 성능을 유지하는 것으로 나타났다.


PICLe is a novel framework for Named Entity Detection (NED) in low-resource settings, leveraging LLMs to generate noisy pseudo-annotations and performing in-context learning through clustering. Experimental results show that PICLe outperforms standard ICL using gold-labeled demonstrations without requiring human annotation. Notably, it maintains effective performance even with partially correct annotations included.

<br/>
# 기타



이 논문에서는 PICLe(Pseudo-annotated In-Context Learning)라는 새로운 프레임워크를 제안하여, 저자들은 저자원 환경에서의 Named Entity Detection(NED) 성능을 향상시키기 위해 다양한 실험을 수행했습니다. 다음은 주요 결과와 인사이트입니다.

1. **부분적으로 올바른 주석의 효과**: 저자들은 부분적으로 올바른 주석이 완전한 주석과 유사한 성능을 발휘할 수 있음을 발견했습니다. 이는 NED에서 주석의 정확성이 낮더라도, 충분한 수의 올바른 엔티티가 포함되어 있다면 효과적인 전이 학습이 가능하다는 것을 의미합니다.

2. **PICLe 프레임워크**: PICLe는 LLM을 활용하여 많은 예시를 제로샷으로 주석 처리하고, 이를 클러스터링하여 각 클러스터에서 특정 세트를 샘플링하여 NED를 수행합니다. 이 과정에서 자기 검증(self-verification)을 통해 최종 엔티티를 선택합니다. PICLe는 5개의 생물 의학 NED 데이터셋에서 평가되었으며, 인간 주석 없이도 ICL보다 우수한 성능을 보였습니다.

3. **실험 결과**: PICLe는 10, 50, 100개의 금본 예시를 사용하는 ICL과 비교하여, 자원 부족 환경에서도 경쟁력 있는 성능을 보였습니다. 특히, PICLe는 금본 예시가 부족한 상황에서도 ICL보다 평균 10.7% 더 높은 성능을 기록했습니다.

4. **자기 검증의 중요성**: 자기 검증 단계는 PICLe의 성능을 향상시키는 데 중요한 역할을 했습니다. 자기 검증을 통해 잘못된 주석을 필터링하고, 최종 예측의 정확성을 높였습니다.

5. **다양한 모델의 성능 비교**: Mistral, Llama2, GPT-3.5-Turbo와 같은 다양한 LLM을 사용하여 PICLe의 성능을 평가했습니다. Mistral 모델이 가장 높은 성능을 보였으며, GPT-3.5-Turbo는 주석 처리 단계에서 높은 품질의 주석을 생성하는 데 기여했습니다.



In this paper, the authors propose a new framework called PICLe (Pseudo-annotated In-Context Learning) to enhance Named Entity Detection (NED) performance in low-resource settings. Here are the key results and insights:

1. **Effect of Partially Correct Annotations**: The authors found that partially correct annotations can perform similarly to fully correct annotations. This suggests that in NED, even with lower accuracy in annotations, effective transfer learning is possible as long as a sufficient number of correct entities are included.

2. **PICLe Framework**: PICLe leverages LLMs to pseudo-annotate many examples in a zero-shot manner, clusters these examples, and samples specific sets from each cluster to perform NED. The process includes a self-verification step to select the final entities. PICLe was evaluated on five biomedical NED datasets and outperformed ICL without any human annotation.

3. **Experimental Results**: PICLe demonstrated competitive performance compared to ICL using 10, 50, and 100 gold examples, achieving an average of 10.7% higher performance even in resource-scarce settings.

4. **Importance of Self-Verification**: The self-verification step played a crucial role in enhancing the performance of PICLe. It helped filter out incorrect annotations and improved the accuracy of final predictions.

5. **Performance Comparison of Different Models**: The performance of PICLe was evaluated using various LLMs, including Mistral, Llama2, and GPT-3.5-Turbo. The Mistral model showed the highest performance, while GPT-3.5-Turbo contributed to generating high-quality annotations during the annotation phase.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{Mamooler2025,
  author = {Sepideh Mamooler and Syrielle Montariol and Alexander Mathis and Antoine Bosselut},
  title = {PICLe: Pseudo-Annotations for In-Context Learning in Low-Resource Named Entity Detection},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies},
  volume = {1},
  pages = {10314--10331},
  year = {2025},
  publisher = {Association for Computational Linguistics},
  address = {April 29 - May 4, 2025}
}
```

### 시카고 스타일 인용
Mamooler, Sepideh, Syrielle Montariol, Alexander Mathis, and Antoine Bosselut. "PICLe: Pseudo-Annotations for In-Context Learning in Low-Resource Named Entity Detection." In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies*, 1:10314–10331. April 29 - May 4, 2025. Association for Computational Linguistics.
