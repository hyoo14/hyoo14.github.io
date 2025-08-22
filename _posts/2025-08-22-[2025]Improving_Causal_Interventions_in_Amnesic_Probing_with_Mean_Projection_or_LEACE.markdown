---
layout: post
title:  "[2025]Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE"
date:   2025-08-22 01:50:11 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 


언어 모델의 특정 언어 정보가 모델의 동작에 미치는 영향을 조사하는 방법인 '기억 상실 탐사(Amnesic Probing)'에 대해 다룸

기존의 정보 제거 기법인 '반복적 영공간 투영(Iterative Nullspace Projection, INLP)'은 목표 정보를 제거할 때 표현에 무작위 변화를 초래할 수 있다는 문제가 있음->

'평균 투영(Mean Projection, MP, MP는 각 클래스의 데이터 포인트의 평균을 사용하여 투영 방향을 찾는 방법)'과 'LEACE(MP처럼 하위공간 사용, 모든 선형 정보를 제거하는 것을 이론적으로 보장, MP보다 계산적으로 더 복잡하지만, 보다 정교한 제거를 제공)'라는 두 가지 대안이 정보를 보다 정확하게 제거하여 기억 상실 탐사를 통한 행동 설명을 얻는 잠재력을 향상시킨다고 주장



짧은 요약(Abstract) :



이 논문은 언어 모델의 특정 언어 정보가 모델의 동작에 미치는 영향을 조사하는 방법인 '기억 상실 탐사(Amnesic Probing)'에 대해 다루고 있습니다. 이 방법은 특정 정보를 식별하고 제거한 후, 모델의 주요 작업 성능이 어떻게 변하는지를 평가합니다. 기존의 정보 제거 기법인 '반복적 영공간 투영(Iterative Nullspace Projection, INLP)'은 목표 정보를 제거할 때 표현에 무작위 변화를 초래할 수 있다는 문제가 있습니다. 이 논문에서는 '평균 투영(Mean Projection, MP)'과 'LEACE'라는 두 가지 대안이 정보를 보다 정확하게 제거하여 기억 상실 탐사를 통한 행동 설명을 얻는 잠재력을 향상시킨다고 주장합니다.




This paper discusses "Amnesic Probing," a technique used to examine the influence of specific linguistic information on a model's behavior. This involves identifying and removing relevant information and then assessing changes in the model's performance on the main task. If the removed information is relevant, the model's performance should decline. The challenge with this approach is removing only the target information while leaving other information unchanged. It has been shown that Iterative Nullspace Projection (INLP), a widely used removal technique, introduces random modifications to representations when eliminating target information. We demonstrate that Mean Projection (MP) and LEACE, two proposed alternatives, remove information in a more targeted manner, thereby enhancing the potential for obtaining behavioral explanations through Amnesic Probing.


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



이 논문에서는 언어 모델의 특정 언어적 속성을 제거하여 모델의 행동을 분석하는 "망각적 탐색(Amnesic Probing)" 기법을 개선하기 위한 두 가지 대안적인 방법, 즉 평균 투영(Mean Projection, MP)과 LEACE를 제안합니다. 이 두 방법은 기존의 반복적 영공간 투영(Iterative Nullspace Projection, INLP) 방법의 한계를 극복하고자 합니다.

#### 망각적 탐색(Amnesic Probing)

망각적 탐색은 모델이 특정 언어적 정보를 사용하고 있는지를 확인하기 위한 방법입니다. 이 방법은 다음과 같은 절차로 진행됩니다:

1. **속성 식별 및 제거**: 모델이 특정 언어적 속성을 포함하고 있는지를 확인하고, 해당 속성을 제거합니다.
2. **모델 행동 평가**: 속성을 제거한 후 모델의 주된 작업(예: 다음 단어 예측)에서의 성능 변화를 평가합니다. 만약 제거된 정보가 중요하다면, 모델의 성능은 저하될 것입니다.
3. **정보 통제**: 제거된 정보가 실제로 모델의 성능에 영향을 미쳤는지를 확인하기 위해, 동일한 양의 무작위 정보를 제거했을 때의 성능 변화와 비교합니다.
4. **선택적 통제**: 제거된 정보를 다시 추가했을 때 모델의 성능이 원래대로 회복되는지를 확인합니다.

#### 평균 투영(Mean Projection, MP)

MP는 각 클래스의 데이터 포인트의 평균을 사용하여 투영 방향을 찾는 방법입니다. 이 방법은 다음과 같은 특징을 가집니다:

- **단일 투영**: 각 클래스에 대해 하나의 투영만 필요로 하며, 이는 INLP의 다중 반복 투영보다 효율적입니다.
- **안정성**: 클래스의 평균을 사용하여 투영 방향을 결정하므로, 클래스 분포가 불균형한 경우에도 효과적입니다.
- **적은 왜곡**: 제거된 방향의 수가 클래스의 수와 동일하므로, 전체 표현 공간에 미치는 부정적 영향이 적습니다.

#### LEACE

LEACE는 MP와 유사하게 레이블 기반의 하위 공간을 사용하여 선형 투영을 수행합니다. 이 방법은 다음과 같은 특징을 가집니다:

- **이론적 보장**: LEACE는 개념에 대한 모든 선형 정보를 제거하는 것을 이론적으로 보장합니다.
- **계산 복잡성**: MP보다 계산적으로 더 복잡하지만, 보다 정교한 제거를 제공합니다.




This paper proposes two alternative methods, Mean Projection (MP) and LEACE, to improve the technique of "Amnesic Probing," which analyzes the behavior of language models by removing specific linguistic attributes. These methods aim to overcome the limitations of the existing Iterative Nullspace Projection (INLP) method.

#### Amnesic Probing

Amnesic Probing is a method to determine whether a model uses specific linguistic information. The procedure involves the following steps:

1. **Attribute Identification and Removal**: Identify whether the model contains specific linguistic attributes and remove them.
2. **Model Behavior Evaluation**: Assess the change in the model's performance on the main task (e.g., next-word prediction) after attribute removal. If the removed information is crucial, the model's performance should decline.
3. **Information Control**: Compare the performance change with the effect of removing an equivalent amount of random information to verify the impact of the removed attribute.
4. **Selectivity Control**: Check if the model's performance can be restored by reintroducing the removed information.

#### Mean Projection (MP)

MP uses the mean of data points in each class to find projection directions. It has the following features:

- **Single Projection**: Requires only one projection per class, making it more efficient than INLP's multiple iterative projections.
- **Stability**: Uses class means to determine projection directions, effective even with unbalanced class distributions.
- **Minimal Distortion**: The number of removed directions equals the number of classes, reducing negative impacts on the overall representation space.

#### LEACE

LEACE, similar to MP, uses label-driven subspaces for linear projection. It has the following features:

- **Theoretical Guarantees**: LEACE theoretically ensures the removal of all linear information about the concept.
- **Computational Complexity**: More computationally intensive than MP but offers more precise removal.

These methods enhance the potential of Amnesic Probing by providing more targeted and precise removal of linguistic attributes, thereby allowing for more reliable behavioral explanations of language models.


<br/>
# Results



이 연구는 Amnesic Probing 기법을 사용하여 언어 모델에서 특정 언어적 속성을 제거하고 그 결과를 분석하는 실험을 수행했습니다. 이 실험에서는 세 가지 정보 제거 방법(INLP, MP, LEACE)을 비교하여 어떤 방법이 더 효과적으로 목표 정보를 제거하는지 평가했습니다.

#### 경쟁 모델 및 테스트 데이터
- **경쟁 모델**: BERT 모델의 두 가지 변형(마스크드 및 언마스크드)을 사용했습니다.
- **테스트 데이터**: Universal Dependencies (UD) 영어 데이터셋을 사용하여 실험을 수행했습니다. 이 데이터셋은 다양한 언어적 속성(예: syntactic dependency, f-pos, c-pos)을 포함하고 있습니다.

#### 메트릭
- **정확도(Accuracy)**: 모델이 다음 단어를 예측하는 정확도를 측정했습니다.
- **Kullback-Leibler Divergence (DKL)**: 두 확률 분포 간의 차이를 측정하여 정보 제거가 모델의 전반적인 토큰 분포에 미치는 영향을 평가했습니다.
- **코사인 유사도(Cosine Similarity)**: 정보 제거 전후의 임베딩 간 유사성을 측정하여 임베딩 공간의 변화를 평가했습니다.
- **행렬 랭크(Matrix Rank)**: 정보 제거 후 임베딩의 선형 독립성 변화를 평가했습니다.

#### 비교 결과
1. **정확도 및 DKL**:
   - INLP는 목표 정보를 제거할 때 다른 정보도 함께 제거하여 모델의 성능이 크게 저하되었습니다.
   - MP와 LEACE는 목표 정보를 보다 정확하게 제거하여 INLP보다 성능 저하가 적었습니다.
   - 특히, MP와 LEACE는 무작위로 정보를 제거한 경우보다 성능 저하가 더 컸으며, 이는 목표 정보가 실제로 모델 성능에 기여했음을 시사합니다.

2. **코사인 유사도 및 행렬 랭크**:
   - INLP는 많은 방향을 제거하여 임베딩 공간에 큰 변화를 초래했습니다.
   - MP와 LEACE는 임베딩 공간을 더 잘 보존하면서 목표 정보를 효과적으로 제거했습니다.
   - 코사인 유사도 측정 결과, MP와 LEACE는 원래 임베딩과의 유사성이 높아 임베딩 공간의 왜곡이 적었습니다.

3. **선택성 테스트**:
   - 목표 정보를 제거한 후 금 정보(gold labels)를 추가했을 때, MP와 LEACE는 원래 성능에 더 가깝게 복원되었습니다.
   - 이는 MP와 LEACE가 목표 정보를 보다 정확하게 제거했음을 나타냅니다.




This study conducted experiments using the Amnesic Probing technique to remove specific linguistic properties from language models and analyze the results. The experiments compared three information removal methods (INLP, MP, LEACE) to evaluate which method more effectively removes the target information.

#### Competing Models and Test Data
- **Competing Models**: Two variants of the BERT model (masked and unmasked) were used.
- **Test Data**: The experiments were conducted using the Universal Dependencies (UD) English dataset, which includes various linguistic properties (e.g., syntactic dependency, f-pos, c-pos).

#### Metrics
- **Accuracy**: Measured the model's accuracy in predicting the next word.
- **Kullback-Leibler Divergence (DKL)**: Assessed the impact of information removal on the overall token distribution of the model by measuring the difference between two probability distributions.
- **Cosine Similarity**: Measured the similarity between embeddings before and after information removal to evaluate changes in the embedding space.
- **Matrix Rank**: Evaluated changes in the linear independence of embeddings after information removal.

#### Comparison Results
1. **Accuracy and DKL**:
   - INLP caused significant performance degradation by removing other information along with the target information.
   - MP and LEACE removed the target information more precisely, resulting in less performance degradation compared to INLP.
   - Notably, MP and LEACE showed greater performance drops than random information removal, indicating that the target information indeed contributed to model performance.

2. **Cosine Similarity and Matrix Rank**:
   - INLP removed many directions, causing substantial changes in the embedding space.
   - MP and LEACE preserved the embedding space better while effectively removing the target information.
   - Cosine similarity measurements showed that MP and LEACE had higher similarity with the original embeddings, indicating less distortion of the embedding space.

3. **Selectivity Test**:
   - When gold labels were added after removing the target information, MP and LEACE restored performance closer to the original.
   - This indicates that MP and LEACE removed the target information more accurately.


<br/>
# 예제

논문에서 사용된 실험의 예시를 설명하겠습니다. 이 논문은 언어 모델에서 특정 언어적 정보를 제거하고 그 결과를 분석하는 "Amnesic Probing" 기법을 다루고 있습니다. 실험은 주로 BERT 모델을 사용하여 진행되었으며, 다음과 같은 단계로 구성됩니다.

### 한글 설명

#### 1. 데이터 세트
- **트레이닝 데이터**: 39,832개의 문장과 1,113,133개의 토큰으로 구성되어 있습니다.
- **테스트 데이터**: 1,700개의 문장과 47,095개의 토큰으로 구성되어 있습니다.

#### 2. 실험 설정
- **모델**: BERT 모델을 사용하여 실험을 진행합니다. 두 가지 변형(마스크드와 언마스크드)으로 실험이 진행됩니다.
- **언어적 속성**: 실험에서는 세 가지 언어적 속성(의존성, f-pos, c-pos)을 제거하는 방법을 비교합니다.

#### 3. 실험 단계
- **단계 1: 속성 식별 및 제거**
  - 원래 모델에서 선형 분류기를 사용하여 목표 속성을 학습할 수 있는지 확인합니다.
  - INLP, MP, LEACE 방법을 사용하여 목표 속성을 제거한 후, 속성을 다시 학습할 수 있는지 확인합니다.

- **단계 2: 모델 행동에 대한 영향**
  - 목표 속성을 제거한 후, 언어 모델링 작업에서의 성능 변화를 평가합니다.
  - 랜덤하게 선택된 방향으로 투영을 적용하여 목표 속성 제거의 영향을 비교합니다.

- **단계 3: 성능 복원**
  - 제거된 목표 속성을 명시적으로 벡터 공간에 추가하여 성능이 복원되는지 확인합니다.

#### 4. 결과
- **정확도 및 KL 발산**: 목표 속성 제거 후의 정확도와 KL 발산을 측정하여 모델의 성능 변화를 평가합니다.
- **코사인 유사도**: 원래 데이터와 수정된 데이터 간의 코사인 유사도를 측정하여 임베딩 공간의 변화를 평가합니다.




#### 1. Dataset
- **Training Data**: Consists of 39,832 sentences and 1,113,133 tokens.
- **Test Data**: Consists of 1,700 sentences and 47,095 tokens.

#### 2. Experimental Setup
- **Model**: The experiments are conducted using the BERT model, with both masked and unmasked variants.
- **Linguistic Properties**: The experiments compare methods for removing three linguistic properties: dependency, f-pos, and c-pos.

#### 3. Experimental Steps
- **Step 1: Identify and Remove Properties**
  - Verify if a linear classifier can learn the target property from the original model.
  - Apply INLP, MP, and LEACE methods to remove the target property and check if it can still be learned.

- **Step 2: Impact on Model Behavior**
  - Evaluate the change in performance on the language modeling task after removing the target property.
  - Compare the impact of removing the target property with random projections.

- **Step 3: Restore Performance**
  - Check if the performance can be restored by explicitly adding the removed target property back into the vector space.

#### 4. Results
- **Accuracy and KL Divergence**: Measure the change in accuracy and KL divergence after removing the target property to evaluate the model's performance change.
- **Cosine Similarity**: Measure the cosine similarity between the original and modified data to evaluate changes in the embedding space.

<br/>
# 요약

이 논문은 Amnesic Probing의 성능을 향상시키기 위해 Mean Projection (MP)와 LEACE라는 두 가지 대안적 정보 제거 방법을 제안합니다. 실험 결과, MP와 LEACE는 기존의 Iterative Nullspace Projection (INLP)보다 더 정확하게 목표 정보를 제거하며, 모델의 성능 저하가 목표 정보의 제거에 기인함을 더 잘 설명할 수 있음을 보여줍니다. 예를 들어, MP와 LEACE는 INLP에 비해 모델의 임베딩 공간을 덜 왜곡하면서도 목표 정보를 효과적으로 제거합니다.



This paper proposes two alternative information removal methods, Mean Projection (MP) and LEACE, to improve the performance of Amnesic Probing. Experimental results demonstrate that MP and LEACE remove target information more precisely than the existing Iterative Nullspace Projection (INLP), providing better explanations for performance degradation due to target information removal. For instance, MP and LEACE effectively remove target information while causing less distortion to the model's embedding space compared to INLP.

<br/>
# 기타



1. **Figure 1 & Figure 2**: 이 다이어그램들은 목표 정보를 식별하고 제거하는 과정을 시각적으로 설명합니다. Figure 1은 2차원 공간에서 목표 방향을 식별하는 방법을 보여주고, Figure 2는 목표 방향 제거 후 임베딩 공간이 어떻게 변화하는지를 보여줍니다. 이 시각화는 선형 투영을 통해 목표 정보를 제거하는 과정을 직관적으로 이해하는 데 도움을 줍니다.

2. **Table 1**: 이 테이블은 비마스크드 BERT에서 목표 정보를 제거한 후의 프로빙 결과를 보여줍니다. INLP, MP, LEACE 모두 목표 정보 제거 후 정확도가 크게 떨어졌으며, MP와 LEACE가 INLP보다 더 정밀하게 정보를 제거함을 보여줍니다.

3. **Table 2**: 이 테이블은 언어 모델링 정확도와 Kullback-Leibler Divergence(DKL)를 비교합니다. INLP는 MP와 LEACE에 비해 더 큰 정확도 감소와 DKL 증가를 보였으며, 이는 INLP가 목표 정보 외의 다른 정보도 제거했음을 시사합니다. MP와 LEACE는 목표 정보 제거가 더 정밀하게 이루어졌음을 보여줍니다.

4. **Table 3**: 이 테이블은 c-pos 카테고리별 왜곡의 영향을 분석합니다. 모든 투영 방법이 주로 기능어(접속사, 관사, 대명사)에 영향을 미쳤으며, MP는 INLP에 비해 명사에 대한 영향이 적고, 동사와 전치사에 대한 정확도에는 영향을 미치지 않거나 오히려 개선되었습니다.

5. **Table 4**: 이 테이블은 선택적 제어 테스트 결과를 보여줍니다. INLP, MP, LEACE 모두 금 라벨을 추가했을 때 성능이 개선되었으며, 이는 목표 정보가 원래 결과에 기여했음을 시사합니다. MP와 LEACE는 INLP보다 원래 성능에 더 가깝게 회복되었습니다.

6. **Appendix A**: INLP와 MP의 차이점을 설명합니다. MP는 단일 투영을 사용하여 목표 정보를 제거하며, 이는 INLP의 반복적인 투영보다 효율적이고 임베딩 공간에 미치는 영향이 적습니다.

7. **Appendix B**: Amnesic Probing 실험 설정의 시각적 표현을 제공합니다. 이 다이어그램은 실험의 각 단계를 개념적으로 보여줍니다.

8. **Appendix C**: 마스크드 BERT에 대한 결과를 제공합니다. 비마스크드 BERT와 유사한 경향을 보이며, MP와 LEACE가 INLP보다 더 정밀하게 정보를 제거함을 보여줍니다.

9. **Appendix D**: 데이터 및 모델 세부 정보를 제공합니다. 실험에 사용된 데이터셋과 모델의 세부 사항을 설명합니다.

10. **Appendix E**: 모델에 대한 전반적인 영향을 설명합니다. 행렬 랭크와 코사인 유사성을 사용하여 INLP, MP, LEACE가 모델에 미치는 영향을 분석합니다. MP와 LEACE가 INLP보다 임베딩 공간을 덜 왜곡함을 보여줍니다.

11. **Appendix F**: 실험에 사용된 컴퓨팅 리소스를 설명합니다. 실험이 수행된 하드웨어 사양을 제공합니다.

---




1. **Figure 1 & Figure 2**: These diagrams visually explain the process of identifying and removing target information. Figure 1 shows how to identify a target direction in a 2D space, and Figure 2 illustrates how the embedding space changes after removing the target direction. This visualization helps intuitively understand the process of removing target information through linear projection.

2. **Table 1**: This table shows the probing results after removing target information from unmasked BERT. All methods (INLP, MP, LEACE) show a significant drop in accuracy after removing target information, with MP and LEACE demonstrating more precise information removal than INLP.

3. **Table 2**: This table compares language modeling accuracy and Kullback-Leibler Divergence (DKL). INLP shows a larger drop in accuracy and increase in DKL compared to MP and LEACE, indicating that INLP removes more than just the target information. MP and LEACE show more precise removal of target information.

4. **Table 3**: This table analyzes the impact of distortion per c-pos category. All projection methods mainly impact function words (conjunctions, determiners, pronouns), with MP having less impact on nouns compared to INLP and no impact or improvement on verbs and adpositions.

5. **Table 4**: This table shows the results of the selectivity control test. All methods (INLP, MP, LEACE) improve performance when gold labels are added, indicating that the target information contributed to the original results. MP and LEACE recover performance closer to the original than INLP.

6. **Appendix A**: Explains the differences between INLP and MP. MP uses a single projection to remove target information, which is more efficient and has less impact on the embedding space than INLP's iterative projections.

7. **Appendix B**: Provides a visual representation of the Amnesic Probing experimental setup. This diagram conceptually shows each step of the experiment.

8. **Appendix C**: Provides results for masked BERT. Similar trends to unmasked BERT are observed, with MP and LEACE showing more precise information removal than INLP.

9. **Appendix D**: Provides details on the data and models used. It describes the datasets and model specifics used in the experiments.

10. **Appendix E**: Describes the overall impact on the model. It analyzes the impact of INLP, MP, and LEACE on the model using matrix rank and cosine similarity, showing that MP and LEACE distort the embedding space less than INLP.

11. **Appendix F**: Describes the computing resources used for the experiments. It provides the hardware specifications on which the experiments were conducted.

<br/>
# refer format:



**BibTeX 형식:**
```bibtex
@inproceedings{Dobrzeniecka2025,
  author    = {Alicja Dobrzeniecka and Antske Fokkens and Pia Sommerauer},
  title     = {Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {12981--12993},
  year      = {2025},
  month     = {July 27--August 1},
  publisher = {Association for Computational Linguistics},
  url       = {https://github.com/efemeryds/amnesic-probing-with-single-projection},
}
```

**시카고 스타일:**
Dobrzeniecka, Alicja, Antske Fokkens, and Pia Sommerauer. 2025. "Improving Causal Interventions in Amnesic Probing with Mean Projection or LEACE." In *Findings of the Association for Computational Linguistics: ACL 2025*, 12981-12993. The Netherlands: Association for Computational Linguistics.
