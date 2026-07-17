---
layout: post
title:  "[2026]SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones"
date:   2026-07-17 02:47:31 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: SLIM은 개별 데이터 소유자가 제한된 블랙박스 접근 하에서 데이터 사용을 검증할 수 있도록 설계된 로우-커버리지 워터마킹 프레임워크입니다.

즉, 훈련 데이터 보호를 위한 데이터 워터마킹 프레임워크.. 데이터에 적은 조작을 가해 워터마크 식별 가능하도록 하는 듯  
(LLM은 입력 시퀀스를 처리할 때, 의미적으로 유사한 접두사를 근접한 잠재 공간 임베딩으로 매핑합니다. SLIM은 이러한 특성을 이용하여 '잠재 공간 혼란 영역(Latent-Space Confusion Zone)'을 유도를 통해)  


짧은 요약(Abstract) :


이 논문의 초록에서는 대규모 언어 모델(LLM) 개발에서 훈련 데이터의 중요성과 데이터 워터마킹의 필요성을 강조하고 있습니다. 데이터 소유자들은 대규모 훈련 데이터의 극히 일부만 기여하기 때문에, 저자들은 "SLIM"이라는 새로운 프레임워크를 제안합니다. SLIM은 개별 사용자 데이터의 출처를 검증할 수 있도록 하며, 블랙박스 접근 방식에서도 작동합니다. 이 프레임워크는 LLM의 고유한 특성을 활용하여 잠재 공간 혼란 영역을 유도하고, 이를 통해 생성의 불안정성을 감지할 수 있습니다. 실험 결과, SLIM은 매우 낮은 커버리지에서도 강력한 검증 성능을 보이며, 스텔스성과 모델 유용성을 유지하면서 확장성이 뛰어난 솔루션을 제공합니다.




The abstract of this paper emphasizes the importance of training data in the development of large language models (LLMs) and the necessity of data watermarking. Since individual data owners typically contribute only a tiny fraction of massive training corpora, the authors propose a new framework called "SLIM." SLIM enables per-user data provenance verification under strict black-box access. This framework leverages intrinsic properties of LLMs to induce a Latent-Space Confusion Zone, allowing for the detection of generation instability. Experimental results demonstrate that SLIM achieves strong verification performance even with ultra-low coverage, while preserving stealthiness and model utility, offering a robust solution for protecting training data in modern LLM pipelines.


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



SLIM(Stealthy Low-coverage Instability Watermarking)은 대규모 언어 모델(LLM)의 훈련 데이터 보호를 위한 데이터 워터마킹 프레임워크입니다. SLIM은 저커버리지 환경에서도 효과적으로 작동할 수 있도록 설계되었으며, 개별 데이터 소유자가 자신의 데이터 사용 여부를 검증할 수 있도록 지원합니다. 이 프레임워크는 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **모델 아키텍처**: SLIM은 대규모 언어 모델의 내재적 특성을 활용하여 작동합니다. LLM은 입력 시퀀스를 처리할 때, 의미적으로 유사한 접두사를 근접한 잠재 공간 임베딩으로 매핑합니다. SLIM은 이러한 특성을 이용하여 '잠재 공간 혼란 영역(Latent-Space Confusion Zone)'을 유도합니다.

2. **워터마킹 단계**: SLIM의 워터마킹 단계에서는 각 타겟 시퀀스를 접두사와 연속으로 분리합니다. 그런 다음, 접두사를 재구성하여 여러 개의 변형된 접두사를 생성하고, 각 변형된 접두사에 대해 주제적으로 다른 연속을 생성합니다. 이 과정에서 생성된 여러 개의 워터마크 시퀀스는 훈련 데이터셋에 삽입됩니다.

3. **검증 단계**: SLIM은 검증 단계에서 통계적 가설 검정을 통해 생성된 연속의 불안정성을 감지합니다. 모델에 접두사를 입력하고 여러 번의 독립적인 실행을 통해 생성된 연속을 수집합니다. 이 연속의 쌍별 유사성을 계산하여, 워터마크가 포함된 모델의 경우 불안정한 생성이 나타나는지를 확인합니다.

4. **저커버리지 설계**: SLIM은 개별 데이터 소유자가 전체 데이터셋의 극히 일부만을 제공하는 현실적인 상황을 반영하여 설계되었습니다. 따라서 SLIM은 단 하나의 시퀀스만 수정하더라도 워터마크를 효과적으로 검증할 수 있도록 합니다.

5. **스텔스성 및 무해성**: SLIM은 워터마킹이 자연스러운 언어 생성에 방해가 되지 않도록 설계되었습니다. 또한, SLIM은 기존의 워터마킹 방법들과 비교하여 높은 스텔스성을 유지하며, 데이터 검증 파이프라인에서 쉽게 탐지되지 않도록 합니다.

SLIM은 이러한 특성 덕분에 대규모 언어 모델의 훈련 데이터 보호를 위한 강력한 솔루션을 제공합니다.

---




SLIM (Stealthy Low-coverage Instability Watermarking) is a data watermarking framework designed to protect training data in large language models (LLMs). It is specifically engineered to operate effectively under low-coverage conditions, allowing individual data owners to verify the usage of their data. The framework consists of the following key components:

1. **Model Architecture**: SLIM operates by leveraging the intrinsic properties of large language models. LLMs process input sequences by mapping semantically similar prefixes to nearby latent space embeddings. SLIM exploits this characteristic to induce a 'Latent-Space Confusion Zone.'

2. **Watermarking Phase**: In the watermarking phase, SLIM separates each target sequence into a prefix and a continuation. It then generates multiple rephrased variants of the prefix and produces topic-distinct continuations for each rephrased prefix. The resulting watermark sequences are inserted into the training dataset.

3. **Verification Phase**: During the verification phase, SLIM detects the instability of generated continuations through statistical hypothesis testing. The model is queried with the prefix, and multiple independent runs are conducted to collect the generated continuations. The pairwise similarity of these continuations is computed to check for instability, which indicates the presence of the watermark.

4. **Low-Coverage Design**: SLIM is designed to reflect realistic scenarios where individual data owners contribute only a tiny fraction of the entire dataset. Therefore, it can effectively verify the watermark even when only a single sequence is modified.

5. **Stealthiness and Harmlessness**: SLIM is engineered to ensure that watermarking does not interfere with natural language generation. Additionally, it maintains high stealthiness compared to existing watermarking methods, making it difficult to detect in data verification pipelines.

Thanks to these characteristics, SLIM provides a robust solution for protecting training data in modern large language model pipelines.


<br/>
# Results



SLIM 프레임워크의 성능을 평가하기 위해 여러 실험을 수행하였으며, 그 결과는 다음과 같습니다.

1. **경쟁 모델**: SLIM은 다양한 대형 언어 모델(LLM)에서 평가되었습니다. 특히, Pythia-160M, Llama-3.2-1B, Gemma-3-4B 모델을 사용하여 실험을 진행했습니다. 이들 모델은 서로 다른 아키텍처와 파라미터 수를 가지고 있어 SLIM의 범용성을 평가하는 데 적합합니다.

2. **테스트 데이터**: 실험에 사용된 데이터셋은 arXiv 논문 초록으로 구성된 gfissore/arxiv-abstracts-2021 데이터셋의 처음 500,000개 시퀀스입니다. 이 데이터셋은 약 1억 개의 토큰을 포함하고 있으며, SLIM의 워터마킹 효과를 평가하기 위해 각 실험에서 단일 타겟 시퀀스만 수정하여 개인 데이터 소유자의 접근을 시뮬레이션했습니다.

3. **메트릭**: SLIM의 성능을 평가하기 위해 여러 메트릭이 사용되었습니다. 주요 메트릭은 다음과 같습니다:
   - **t-통계량 변화(∆t)**: 워터마킹된 시퀀스의 수(K)가 증가함에 따라 t-통계량의 변화를 측정하여 워터마킹의 효과를 평가했습니다. 비워터마킹 샘플과 비교하여 워터마킹 샘플의 t-통계량이 유의미하게 변화하는 것을 관찰했습니다.
   - **모델 유틸리티**: SLIM이 모델의 성능에 미치는 영향을 평가하기 위해 ARC, MMLU, BBQ와 같은 벤치마크를 사용했습니다. 워터마킹 전후의 성능 차이는 0.02 이하로 나타나, SLIM이 모델 유틸리티에 미치는 영향이 미미함을 보여주었습니다.
   - **스텔스성**: SLIM의 스텔스성을 평가하기 위해 n-그램 필터링, 압축 비율, 임베딩 코사인 유사도와 같은 다양한 탐지 메트릭을 사용했습니다. SLIM은 모든 탐지 메트릭에서 우수한 성능을 보이며, 이전의 워터마킹 방법들보다 더 높은 스텔스성을 유지했습니다.

4. **비교**: SLIM은 기존의 워터마킹 방법들과 비교하여 여러 면에서 우수한 성능을 보였습니다. 예를 들어, WATERFALL, STAMP, TRACE와 같은 기존 방법들은 스텔스성이나 검증 가능성에서 한계를 보였으나, SLIM은 저커버리지 환경에서도 효과적으로 작동하며, 검증 가능성과 스텔스성을 모두 만족시켰습니다.

이러한 결과들은 SLIM이 현대 LLM 파이프라인에서 훈련 데이터 보호를 위한 강력한 솔루션임을 입증합니다.

---




To evaluate the performance of the SLIM framework, several experiments were conducted, and the results are as follows:

1. **Competing Models**: SLIM was evaluated on various large language models (LLMs). Specifically, experiments were conducted using the Pythia-160M, Llama-3.2-1B, and Gemma-3-4B models. These models have different architectures and parameter counts, making them suitable for assessing the generalizability of SLIM.

2. **Test Data**: The dataset used for the experiments consisted of the first 500,000 sequences from the gfissore/arxiv-abstracts-2021 dataset, which comprises arXiv paper abstracts. This dataset contains approximately 100 million tokens, and in each experiment, only a single target sequence was modified to simulate the access of individual data owners.

3. **Metrics**: Several metrics were used to evaluate the performance of SLIM. The key metrics include:
   - **Change in t-statistic (∆t)**: The change in the t-statistic was measured as the number of watermarked sequences (K) increased, assessing the effectiveness of the watermarking. A significant change in the t-statistic for watermarked samples compared to non-watermarked samples was observed.
   - **Model Utility**: To evaluate the impact of SLIM on model performance, benchmarks such as ARC, MMLU, and BBQ were used. The performance difference before and after watermarking remained below 0.02, indicating that SLIM has a negligible impact on model utility.
   - **Stealthiness**: Various detection metrics, including n-gram filtering, compression ratio, and embedding cosine similarity, were used to assess the stealthiness of SLIM. SLIM demonstrated excellent performance across all detection metrics, maintaining higher stealthiness compared to previous watermarking methods.

4. **Comparison**: SLIM outperformed existing watermarking methods in several aspects. For instance, prior methods like WATERFALL, STAMP, and TRACE showed limitations in stealthiness or verification feasibility, while SLIM effectively operates under low coverage conditions, satisfying both verifiability and stealthiness.

These results demonstrate that SLIM is a robust solution for protecting training data in modern LLM pipelines.


<br/>
# 예제



SLIM 프레임워크는 데이터 워터마킹을 통해 개별 데이터 소유자가 자신의 데이터 사용 여부를 검증할 수 있도록 설계되었습니다. 이 과정은 두 가지 주요 단계로 나뉩니다: 워터마킹 단계와 검증 단계입니다.

1. **워터마킹 단계**:
   - **입력 데이터**: 예를 들어, 특정 연구 논문의 초록이 포함된 데이터 시퀀스가 있다고 가정합니다. 이 시퀀스는 "우리는 초고속 전자 나노 결정학에 대한 연구를 보고합니다..."로 시작합니다.
   - **작업**: 이 시퀀스는 두 부분으로 나뉘어집니다: 접두사(P)와 연속(C). 예를 들어, 접두사는 "우리는 초고속 전자 나노 결정학에 대한 연구를 보고합니다..."이고, 연속은 "이 연구는 나노 입자의 크기 선택에 관한 것입니다."입니다.
   - **변형 생성**: SLIM은 접두사를 여러 가지 방식으로 재구성하여 K개의 서로 다른 변형을 생성합니다. 예를 들어, "우리는 초고속 전자 나노 결정학에 대한 연구를 보고합니다..."를 "우리는 초고속 전자 나노 결정학에 대한 조사를 수행했습니다..."로 바꿀 수 있습니다.
   - **워터마크 삽입**: 각 변형된 접두사에 대해 새로운 연속을 생성하여 K개의 워터마크 시퀀스를 만듭니다. 이 시퀀스는 원래 데이터셋에 삽입됩니다.

2. **검증 단계**:
   - **입력 데이터**: 훈련된 모델에 대해 의심스러운 접두사(P)를 입력합니다. 예를 들어, "우리는 초고속 전자 나노 결정학에 대한 연구를 보고합니다..."를 입력합니다.
   - **작업**: 모델은 이 접두사에 대해 Q번의 독립적인 실행을 수행하여 생성된 연속(C′)을 수집합니다. 이 연속은 모델이 훈련된 데이터에 포함되었는지 여부를 판단하는 데 사용됩니다.
   - **유사도 계산**: 생성된 연속의 쌍 간 유사도를 계산하여 통계적 검증을 수행합니다. 예를 들어, BERTScore를 사용하여 생성된 연속 간의 유사도를 측정합니다.
   - **결과**: 검증 결과는 t-통계량으로 표현되며, 이 값이 사전 정의된 임계값을 초과하면 해당 데이터가 모델 훈련에 사용되었음을 나타냅니다.

이러한 방식으로 SLIM은 개별 데이터 소유자가 자신의 데이터 사용 여부를 검증할 수 있도록 하며, 이는 데이터 소유권 보호 및 지적 재산권 보호에 기여합니다.

---




The SLIM framework is designed to enable individual data owners to verify whether their data has been used. This process is divided into two main stages: the watermarking phase and the verification phase.

1. **Watermarking Phase**:
   - **Input Data**: For example, consider a data sequence that contains the abstract of a specific research paper. This sequence starts with "We report the studies of ultrafast electron nanocrystallography on size-selected Au nanoparticles...".
   - **Task**: This sequence is divided into two parts: a prefix (P) and a continuation (C). For instance, the prefix could be "We report the studies of ultrafast electron nanocrystallography on size-selected Au nanoparticles..." and the continuation could be "This research pertains to the size selection of nanoparticles.".
   - **Variant Generation**: SLIM generates K different variants of the prefix by rephrasing it. For example, "We report the studies of ultrafast electron nanocrystallography on size-selected Au nanoparticles..." could be rephrased to "We have conducted an investigation into ultrafast electron nanocrystallography...".
   - **Watermark Insertion**: For each rephrased prefix, a new continuation is generated, resulting in K watermark sequences that are inserted into the original dataset.

2. **Verification Phase**:
   - **Input Data**: A suspicious prefix (P) is input into the trained model. For example, "We report the studies of ultrafast electron nanocrystallography on size-selected Au nanoparticles...".
   - **Task**: The model performs Q independent runs with this prefix to collect generated continuations (C′). This data is used to determine whether the model was trained on the input data.
   - **Similarity Calculation**: The pairwise similarity between the generated continuations is calculated to perform statistical verification. For instance, BERTScore is used to measure the similarity between the generated continuations.
   - **Result**: The verification result is expressed as a t-statistic, and if this value exceeds a predefined threshold, it indicates that the data was used in training the model.

In this way, SLIM allows individual data owners to verify the usage of their data, contributing to the protection of data ownership and intellectual property rights.

<br/>
# 요약
SLIM은 개별 데이터 소유자가 제한된 블랙박스 접근 하에서 데이터 사용을 검증할 수 있도록 설계된 저커버리지 워터마킹 프레임워크입니다. 이 방법은 잠재 공간 혼란 영역을 유도하여 생성 불안정을 발생시키고, 이를 통해 검증을 위한 통계적 가설 검정을 수행합니다. 실험 결과, SLIM은 강력한 추적 가능성을 유지하면서 모델 유틸리티를 보존하고, 일반적인 탐지 파이프라인에서도 스텔스성을 유지하는 것으로 나타났습니다.

---

SLIM is a low-coverage watermarking framework designed to enable individual data owners to verify data usage under strict black-box access. This method induces localized generation instability by creating a Latent-Space Confusion Zone, allowing for statistical hypothesis testing for verification. Experimental results show that SLIM maintains strong traceability while preserving model utility and remains stealthy under common detection pipelines.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: SLIM의 워터마킹 및 검증 프로세스를 보여줍니다. 이 다이어그램은 SLIM이 어떻게 Latent-Space Confusion Zone을 생성하고, 이를 통해 모델의 생성 불안정성을 유도하는지를 시각적으로 설명합니다.
   - **Figure 2**: Latent-Space Confusion Zone의 형성을 설명합니다. 이 피규어는 원래 접두사와 재구성된 접두사가 어떻게 근접한 잠재 표현으로 매핑되는지를 보여줍니다.
   - **Figure 3**: 검증 t-통계량의 변화가 워터마크된 시퀀스 수(K)에 따라 어떻게 변화하는지를 나타냅니다. K가 증가함에 따라 t-통계량의 변화가 뚜렷하게 나타나며, 이는 SLIM이 낮은 커버리지에서도 효과적으로 작동함을 보여줍니다.
   - **Figure 4**: H0 하에서 t-값의 분포와 워터마크된 샘플의 위치를 보여줍니다. K=64일 때, 워터마크된 샘플의 t-값이 H0의 분포에서 유의미하게 벗어나 있음을 나타냅니다.
   - **Figure 5**: 훈련 데이터셋 크기에 따른 t-통계량의 변화를 보여줍니다. 데이터셋 크기가 증가함에 따라 신호가 희석되는 경향이 있음을 나타냅니다.
   - **Figure 6**: 모델 아키텍처에 따른 t-통계량의 변화를 보여줍니다. 모델 크기가 증가함에 따라 신호가 감지 가능해지지만, 큰 모델에서는 신호가 부분적으로 감소하는 경향이 있습니다.
   - **Figure 7**: 여러 독립적인 워터마크를 동시에 삽입했을 때 t-통계량의 변화를 보여줍니다. K가 증가해도 개별 워터마크의 감지 가능성이 유지됨을 나타냅니다.

2. **테이블**
   - **Table 1**: SLIM과 기존의 데이터 워터마킹 방법들을 비교합니다. SLIM은 검증 가능성, 은폐성, 커버리지 측면에서 모두 우수한 성능을 보입니다.
   - **Table 2**: SLIM 워터마킹을 적용한 모델의 다운스트림 작업 성능을 평가합니다. 워터마킹 전후의 성능 차이가 미미함을 보여줍니다.
   - **Table 3**: SLIM의 은폐성을 평가하기 위한 다양한 탐지 메트릭에 대한 비교 결과를 보여줍니다. SLIM은 모든 탐지 메트릭에서 우수한 성능을 보입니다.

3. **어펜딕스**
   - **Appendix A**: SLIM 워터마킹 알고리즘의 의사코드입니다. 워터마킹 과정의 세부적인 단계를 설명합니다.
   - **Appendix B**: 원본 시퀀스와 워터마크 시퀀스의 예시를 제공합니다. 접두사와 연속 부분이 어떻게 구분되는지를 보여줍니다.
   - **Appendix C**: 접두사 재구성을 위한 프롬프트를 설명합니다. 의미를 유지하면서 다양한 버전을 생성하는 방법을 제시합니다.
   - **Appendix D**: 연속 생성에 대한 프롬프트를 설명합니다. 접두사와 연결될 수 있는 완전히 다른 내용을 생성하는 방법을 제시합니다.




1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the watermarking and verification process of SLIM. This diagram visually explains how SLIM generates a Latent-Space Confusion Zone and induces generation instability in the model.
   - **Figure 2**: Describes the formation of the Latent-Space Confusion Zone. This figure shows how the original prefix and the reconstructed prefix are mapped to nearby latent representations.
   - **Figure 3**: Displays how the shift in the verification t-statistic varies with the number of watermarked sequences (K). As K increases, the change in the t-statistic becomes pronounced, demonstrating SLIM's effectiveness even under low coverage.
   - **Figure 4**: Shows the distribution of t-values under H0 and the positions of watermarked samples. At K=64, the t-values of watermarked samples significantly deviate from the distribution of H0.
   - **Figure 5**: Illustrates the change in t-statistic as a function of training dataset size. It indicates a dilution effect as the dataset size increases.
   - **Figure 6**: Reports the change in t-statistic based on model architecture. As model size increases, the signal becomes detectable, but larger models exhibit partial attenuation of the signal.
   - **Figure 7**: Examines the change in t-statistic when multiple independent watermarks are injected simultaneously. It shows that the detectability of individual watermarks is maintained even as K increases.

2. **Tables**
   - **Table 1**: Compares SLIM with existing data watermarking methods. SLIM demonstrates superior performance in terms of verification feasibility, stealthiness, and coverage.
   - **Table 2**: Evaluates the downstream task performance of models with SLIM watermarking. The performance difference before and after watermarking is minimal.
   - **Table 3**: Presents a comparison of various detection metrics for evaluating the stealthiness of SLIM. SLIM performs excellently across all detection metrics.

3. **Appendices**
   - **Appendix A**: Pseudocode for the SLIM watermarking algorithm. It details the steps involved in the watermarking process.
   - **Appendix B**: Provides examples of original and watermarked sequences, showing how the prefix and continuation parts are distinguished.
   - **Appendix C**: Describes the prompt for prefix rephrasing. It presents a method for generating different versions while maintaining meaning.
   - **Appendix D**: Explains the prompt for continuation generation. It outlines how to generate completely different content that can connect to the prefix.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@inproceedings{wu2026slim,
  author    = {Hengyu Wu and Yang Cao},
  title     = {SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2026},
  pages     = {18458--18473},
  year      = {2026},
  month     = {July},
  publisher = {Association for Computational Linguistics},


}
```

### 시카고 스타일 인용
Hengyu Wu and Yang Cao. "SLIM: Stealthy Low-Coverage Black-Box Watermarking via Latent-Space Confusion Zones." In *Findings of the Association for Computational Linguistics: ACL 2026*, 18458–18473. Association for Computational Linguistics, July 2026.   
