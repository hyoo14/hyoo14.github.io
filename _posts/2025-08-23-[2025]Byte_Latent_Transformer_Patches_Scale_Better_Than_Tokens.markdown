---
layout: post
title:  "[2025]Byte Latent Transformer: Patches Scale Better Than Tokens"
date:   2025-08-23 20:19:51 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

Byte Latent Transformer(BLT)는 바이트를 동적으로 패치로 그룹화하여 토큰화 기반 모델과 유사한 성능을 달성하면서 추론 효율성을 개선하는 새로운 아키텍처입니다.


짧은 요약(Abstract) :



이 논문에서는 Byte Latent Transformer (BLT)라는 새로운 바이트 수준의 대형 언어 모델(LLM) 아키텍처를 소개합니다. BLT는 처음으로 토큰화 기반 LLM의 성능을 대규모로 맞추면서도 추론 효율성과 견고성을 크게 개선합니다. BLT는 바이트를 동적으로 크기가 조정되는 패치로 인코딩하며, 이는 주된 계산 단위로 사용됩니다. 패치는 다음 바이트의 엔트로피에 따라 분할되어, 데이터 복잡성이 증가할 때 더 많은 계산과 모델 용량을 할당합니다. 우리는 최대 80억 개의 매개변수와 4조 개의 학습 바이트에 이르는 바이트 수준 모델의 FLOP 제어 확장 연구를 처음으로 제시하며, 고정된 어휘 없이 원시 바이트로 학습된 모델의 확장 가능성을 입증합니다. 데이터가 예측 가능할 때 긴 패치를 동적으로 선택함으로써 학습 및 추론 효율성이 향상되며, 추론 비용이 고정된 경우 BLT는 패치 크기와 모델 크기를 동시에 증가시켜 토큰화 기반 모델보다 훨씬 나은 확장성을 보여줍니다.




We introduce the Byte Latent Transformer (BLT), a new byte-level LLM architecture that, for the first time, matches tokenization-based LLM performance at scale with significant improvements in inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the primary units of computation. Patches are segmented based on the entropy of the next byte, allocating more compute and model capacity where increased data complexity demands it. We present the first FLOP controlled scaling study of byte-level models – up to 8B parameters and 4T training bytes – demonstrating the feasibility of scaling models trained on raw bytes without a fixed vocabulary. Both training and inference efficiency improve due to dynamically selecting long patches when data is predictable, along with qualitative improvements on reasoning and long tail generalization. For fixed inference costs, BLT shows significantly better scaling than tokenization-based models, by simultaneously growing both patch and model size.


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



**바이트 잠재 변환기 (Byte Latent Transformer, BLT)**

바이트 잠재 변환기(BLT)는 바이트 수준의 대형 언어 모델(LLM) 아키텍처로, 기존의 토큰화 기반 모델과 비교하여 성능을 유지하면서도 추론 효율성과 강건성을 크게 개선한 모델입니다. BLT는 바이트를 동적으로 크기가 조정되는 패치로 인코딩하여, 이를 주된 계산 단위로 사용합니다. 이러한 패치는 다음 바이트의 엔트로피에 따라 분할되며, 데이터의 복잡성이 증가할 때 더 많은 계산과 모델 용량을 할당합니다.

BLT는 고정된 어휘 없이 원시 바이트로부터 모델을 확장할 수 있는 가능성을 보여주며, 최대 80억 개의 매개변수와 4조 개의 훈련 바이트를 사용하여 바이트 수준 모델의 FLOP(부동소수점 연산) 제어 확장 연구를 처음으로 제시합니다. 예측 가능한 데이터에서는 긴 패치를 동적으로 선택함으로써 훈련 및 추론 효율성이 향상되며, 추론 비용이 고정된 경우 BLT는 패치 크기와 모델 크기를 동시에 증가시켜 토큰화 기반 모델보다 훨씬 더 나은 확장성을 보여줍니다.

BLT는 세 가지 주요 모듈로 구성됩니다:
1. **로컬 인코더**: 입력 바이트를 패치 표현으로 인코딩하는 경량의 변환기 기반 모델입니다.
2. **잠재 변환기**: 패치 표현을 기반으로 다음 패치를 예측하는 대형 자회귀 변환기 모델입니다.
3. **로컬 디코더**: 다음 패치의 바이트를 디코딩하는 경량의 변환기 기반 모델입니다.

BLT는 바이트 n-그램 임베딩과 교차 주의 메커니즘을 통합하여 바이트와 패치 간의 정보 흐름을 극대화하고 바이트 수준의 정보를 유지합니다. BLT는 고정된 어휘 없이 임의의 바이트 그룹을 잠재 패치 표현으로 매핑하는 경량의 학습된 인코더 및 디코더 모듈을 사용합니다.

BLT는 다음과 같은 주요 기여를 합니다:
1. FLOP 효율성을 개선하기 위해 동적으로 계산을 할당하는 바이트 잠재 LLM 아키텍처를 소개합니다.
2. 최대 80억 규모에서 Llama 3와의 FLOP 제어 훈련 성능을 달성하며, 최대 50%의 FLOP 효율성 향상의 잠재력을 제공합니다.
3. LLM을 확장할 수 있는 새로운 차원을 열어, 고정된 추론 예산을 유지하면서 모델과 패치 크기를 함께 증가시킬 수 있습니다.
4. 입력 노이즈에 대한 강건성을 개선하고, 토큰 기반 LLM이 놓치는 하위 단어 측면에 대한 인식을 보여줍니다.




**Byte Latent Transformer (BLT)**

The Byte Latent Transformer (BLT) is a byte-level large language model (LLM) architecture that matches the performance of tokenization-based models while significantly improving inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, which serve as the primary units of computation. These patches are segmented based on the entropy of the next byte, allocating more compute and model capacity where increased data complexity demands it.

BLT demonstrates the feasibility of scaling models trained on raw bytes without a fixed vocabulary, presenting the first FLOP-controlled scaling study of byte-level models with up to 8 billion parameters and 4 trillion training bytes. Both training and inference efficiency improve due to dynamically selecting long patches when data is predictable, and for fixed inference costs, BLT shows significantly better scaling than tokenization-based models by simultaneously growing both patch and model size.

BLT consists of three main modules:
1. **Local Encoder**: A lightweight transformer-based model that encodes input bytes into patch representations.
2. **Latent Transformer**: A large autoregressive transformer model that predicts the next patch based on patch representations.
3. **Local Decoder**: A lightweight transformer-based model that decodes the next patch of bytes.

BLT incorporates byte n-gram embeddings and a cross-attention mechanism to maximize information flow between bytes and patches and preserve access to byte-level information. It uses lightweight learned encoder and decoder modules to map arbitrary groups of bytes to latent patch representations without a fixed vocabulary.

BLT contributes the following key advancements:
1. Introduces a byte latent LLM architecture that dynamically allocates compute for improved FLOP efficiency.
2. Achieves training FLOP-controlled parity with Llama 3 up to 8B scale, with potential FLOP efficiency gains of up to 50%.
3. Unlocks a new dimension for scaling LLMs, allowing model and patch size to jointly increase while maintaining a fixed-inference budget.
4. Demonstrates improved robustness to input noise and awareness of sub-word aspects missed by token-based LLMs.


<br/>
# Results



이 논문에서는 Byte Latent Transformer (BLT)라는 새로운 바이트 수준의 대형 언어 모델(LLM) 아키텍처를 소개합니다. BLT는 기존의 토큰화 기반 모델과 비교하여 여러 가지 장점을 가지고 있으며, 특히 추론 효율성과 강건성에서 큰 개선을 보입니다. BLT는 바이트를 동적으로 크기가 조정되는 패치로 인코딩하여, 데이터의 복잡성에 따라 더 많은 계산과 모델 용량을 할당합니다. 이 논문에서는 BLT의 성능을 다양한 실험을 통해 평가하였으며, 그 결과를 다음과 같이 요약할 수 있습니다.

1. **경쟁 모델과의 비교**: BLT는 Llama 3와 같은 토큰화 기반 모델과 비교하여, 최대 8B 파라미터와 4T 바이트의 훈련 데이터를 사용하여 성능을 맞추거나 초과합니다. 특히, BLT는 추론 시 최대 50%의 FLOP(부동소수점 연산) 절감을 달성합니다.

2. **테스트 데이터 및 메트릭**: BLT는 다양한 데이터셋과 벤치마크에서 평가되었습니다. 여기에는 HellaSwag, PIQA, MMLU, MBPP, HumanEval과 같은 일반적인 자연어 처리(NLP) 벤치마크가 포함됩니다. 또한, FLORES-101을 사용하여 다국어 번역 성능도 평가되었습니다. 성능 평가는 주로 BPB(바이트당 비트)와 BLEU 점수로 측정되었습니다.

3. **비교 결과**: BLT는 Llama 3와 비교하여 여러 벤치마크에서 더 나은 성능을 보였습니다. 특히, BLT는 노이즈가 있는 입력에 대한 강건성에서 우수한 성능을 보였으며, 이는 바이트 수준의 정보 모델링 덕분입니다. 또한, BLT는 저자원 기계 번역 작업에서도 Llama 3보다 더 나은 성능을 보였습니다.

4. **추가 실험**: BLT는 패치 크기와 모델 크기를 동시에 증가시킬 수 있는 새로운 스케일링 축을 제공합니다. 이는 고정된 어휘를 사용하는 토큰 기반 모델의 효율성 한계를 극복할 수 있는 가능성을 보여줍니다.




### Summary of Results

This paper introduces a new byte-level large language model (LLM) architecture called the Byte Latent Transformer (BLT). BLT offers several advantages over traditional tokenization-based models, particularly in terms of inference efficiency and robustness. BLT encodes bytes into dynamically sized patches, allocating more compute and model capacity based on data complexity. The performance of BLT is evaluated through various experiments, and the results can be summarized as follows:

1. **Comparison with Competing Models**: BLT matches or exceeds the performance of tokenization-based models like Llama 3, using up to 8B parameters and 4T training bytes. Notably, BLT achieves up to 50% reduction in FLOPs (floating-point operations) during inference.

2. **Test Data and Metrics**: BLT was evaluated on a variety of datasets and benchmarks, including common NLP benchmarks like HellaSwag, PIQA, MMLU, MBPP, and HumanEval. Additionally, multilingual translation performance was assessed using FLORES-101. Performance was primarily measured using bits-per-byte (BPB) and BLEU scores.

3. **Comparison Results**: BLT outperformed Llama 3 on several benchmarks. In particular, BLT demonstrated superior robustness to noisy inputs, thanks to its byte-level information modeling. BLT also showed better performance in low-resource machine translation tasks compared to Llama 3.

4. **Additional Experiments**: BLT provides a new scaling axis that allows simultaneous increases in patch and model size. This suggests the potential to overcome the efficiency trade-offs of fixed-vocabulary token-based models.


<br/>
# 예제
논문에서 소개된 Byte Latent Transformer (BLT) 모델은 다양한 자연어 처리(NLP) 작업에서 성능을 평가받았습니다. 이 모델은 특히 바이트 수준의 입력을 처리하는 데 중점을 두고 있으며, 이는 전통적인 토큰화 기반 모델과의 차별점입니다. BLT의 성능을 평가하기 위해 사용된 몇 가지 주요 작업과 데이터셋을 설명하겠습니다.

### 예시 1: HellaSwag
- **트레이닝 데이터**: HellaSwag 데이터셋은 다양한 상황에서의 문장 완성을 요구하는 문제로 구성되어 있습니다. 각 예시는 문맥을 제공하는 문장과 그 문맥에 맞는 여러 개의 문장 후보로 이루어져 있습니다.
- **테스트 데이터**: 테스트 데이터는 트레이닝 데이터와 유사한 형식으로, 모델이 문맥에 가장 적합한 문장을 선택해야 합니다.
- **구체적인 테스크**: 모델은 주어진 문맥에 가장 적합한 문장을 선택하는 것입니다. 예를 들어, "그녀는 방에 들어가서"라는 문장이 주어졌을 때, "불을 켰다"와 같은 문장을 선택해야 합니다.

### 예시 2: Phonology - Grapheme-to-Phoneme (G2P)
- **트레이닝 데이터**: 이 작업은 문자(그래프) 시퀀스를 발음(음소)으로 변환하는 것을 목표로 합니다. 트레이닝 데이터는 각 문자 시퀀스와 그에 대응하는 발음 시퀀스로 구성됩니다.
- **테스트 데이터**: 테스트 데이터는 새로운 문자 시퀀스를 포함하며, 모델은 이를 정확한 발음으로 변환해야 합니다.
- **구체적인 테스크**: 예를 들어, "cat"이라는 입력이 주어졌을 때, 모델은 이를 /kæt/로 변환해야 합니다.

### 예시 3: CUTE Benchmark
- **트레이닝 데이터**: CUTE 벤치마크는 문자 이해, 철자법, 시퀀스 조작과 관련된 다양한 작업을 포함합니다. 트레이닝 데이터는 이러한 작업에 대한 다양한 예시를 포함합니다.
- **테스트 데이터**: 테스트 데이터는 모델이 문자 수준에서의 이해와 조작 능력을 평가할 수 있도록 설계된 새로운 예시를 포함합니다.
- **구체적인 테스크**: 예를 들어, "apple"이라는 단어가 주어졌을 때, 이를 "elppa"로 뒤집는 작업을 수행해야 할 수 있습니다.

### 예시 4: Low Resource Machine Translation
- **트레이닝 데이터**: FLORES-101 데이터셋은 다양한 언어 쌍에 대한 번역 작업을 포함합니다. 특히 자원이 적은 언어 쌍에 대한 번역 예시가 포함됩니다.
- **테스트 데이터**: 테스트 데이터는 새로운 문장 쌍을 포함하며, 모델은 이를 정확하게 번역해야 합니다.
- **구체적인 테스크**: 예를 들어, 한국어 문장 "안녕하세요"가 주어졌을 때, 이를 영어로 "Hello"로 번역해야 합니다.

---




The Byte Latent Transformer (BLT) model introduced in the paper is evaluated on various natural language processing (NLP) tasks. This model focuses on processing byte-level inputs, which distinguishes it from traditional tokenization-based models. Here are some key tasks and datasets used to evaluate the performance of BLT.

### Example 1: HellaSwag
- **Training Data**: The HellaSwag dataset consists of tasks that require sentence completion in various contexts. Each example includes a context-providing sentence and multiple candidate sentences that fit the context.
- **Test Data**: The test data is similar in format to the training data, where the model must select the sentence that best fits the context.
- **Specific Task**: The model needs to select the sentence that best fits the given context. For example, given the sentence "She entered the room and," the model should choose a sentence like "turned on the light."

### Example 2: Phonology - Grapheme-to-Phoneme (G2P)
- **Training Data**: This task aims to convert sequences of graphemes (characters) into their phonetic transcriptions (phonemes). The training data consists of each character sequence and its corresponding pronunciation sequence.
- **Test Data**: The test data includes new character sequences, and the model must convert them into the correct pronunciation.
- **Specific Task**: For example, given the input "cat," the model should convert it to /kæt/.

### Example 3: CUTE Benchmark
- **Training Data**: The CUTE benchmark includes various tasks related to character understanding, spelling, and sequence manipulation. The training data includes diverse examples for these tasks.
- **Test Data**: The test data includes new examples designed to evaluate the model's ability to understand and manipulate characters.
- **Specific Task**: For example, given the word "apple," the task might be to reverse it to "elppa."

### Example 4: Low Resource Machine Translation
- **Training Data**: The FLORES-101 dataset includes translation tasks for various language pairs, especially those with low resources. It includes examples of translations for these language pairs.
- **Test Data**: The test data includes new sentence pairs, and the model must translate them accurately.
- **Specific Task**: For example, given the Korean sentence "안녕하세요," the model should translate it to English as "Hello."

<br/>
# 요약

Byte Latent Transformer(BLT)는 바이트를 동적으로 패치로 그룹화하여 토큰화 기반 모델과 유사한 성능을 달성하면서 추론 효율성을 개선하는 새로운 아키텍처입니다. BLT는 최대 8B 파라미터와 4T 바이트의 데이터를 사용하여 고정된 어휘 없이 바이트 수준에서 모델을 훈련할 수 있으며, 패치 크기와 모델 크기를 동시에 확장하여 고정된 추론 비용 내에서 더 나은 성능을 보여줍니다. 실험 결과, BLT는 입력 노이즈에 대한 강건성과 문자 수준의 이해에서 개선된 성능을 보였으며, 특히 저자원 기계 번역 작업에서 우수한 성능을 나타냈습니다.


 
The Byte Latent Transformer (BLT) is a novel architecture that groups bytes into dynamic patches, achieving performance comparable to tokenization-based models while improving inference efficiency. BLT can train models at the byte level with up to 8B parameters and 4T bytes of data without a fixed vocabulary, showing better performance within a fixed inference budget by simultaneously scaling patch and model sizes. Experimental results demonstrate BLT's improved robustness to input noise and enhanced character-level understanding, particularly excelling in low-resource machine translation tasks.

<br/>
# 기타


1. **Figure 1: Scaling trends for fixed inference FLOP models**
   - 이 다이어그램은 고정된 추론 FLOP 예산 내에서 BLT 모델과 BPE 기반 모델의 스케일링 트렌드를 비교합니다. BLT 모델은 패치 크기와 모델 크기를 동시에 증가시킬 수 있어, BPE 기반 모델보다 더 나은 스케일링 트렌드를 보여줍니다. 이는 BLT가 더 적은 FLOP로 더 큰 패치 크기를 처리할 수 있음을 시사합니다.

2. **Table 1: Comparison of FLOP-matched BLT and BPE 8B models on downstream tasks**
   - 이 테이블은 FLOP가 일치하는 BLT와 BPE 기반 모델을 다양한 다운스트림 작업에서 비교합니다. BLT 모델은 Llama 3 모델보다 더 나은 성능을 보이며, 특히 패치 크기를 조정하여 FLOP 효율성을 높일 수 있습니다.

3. **Table 2: 8B BLT and BPE Llama 3 on tasks assessing robustness to noise and character-level understanding**
   - 이 테이블은 노이즈에 대한 강건성과 문자 수준 이해를 평가하는 작업에서 BLT와 BPE Llama 3 모델을 비교합니다. BLT 모델은 노이즈가 있는 데이터에 대해 더 나은 성능을 보이며, 문자 수준의 이해에서도 우수한 성능을 나타냅니다.

4. **Table 3: Performance on translation tasks from FLORES-101**
   - 이 테이블은 FLORES-101 번역 작업에서 BLT와 BPE Llama 3 모델의 성능을 비교합니다. BLT 모델은 저자원 번역 작업에서 특히 더 나은 성능을 보이며, 전반적으로 BPE 모델보다 높은 BLEU 점수를 기록합니다.

5. **Table 4: Ablations on the use of Cross Attention for a 1B BLT model**
   - 이 테이블은 1B BLT 모델에서 크로스 어텐션의 사용에 대한 다양한 설정을 비교합니다. 크로스 어텐션을 적절히 활용하면 모델의 성능이 향상될 수 있음을 보여줍니다.




The supplementary materials (diagrams, figures, tables, appendices, etc.) in the paper evaluate the performance and efficiency of the Byte Latent Transformer (BLT) model from various perspectives. The results and insights from each material are summarized as follows:

1. **Figure 1: Scaling trends for fixed inference FLOP models**
   - This diagram compares the scaling trends of BLT models and BPE-based models within a fixed inference FLOP budget. BLT models show better scaling trends than BPE-based models, as they can simultaneously increase patch size and model size, indicating that BLT can handle larger patch sizes with fewer FLOPs.

2. **Table 1: Comparison of FLOP-matched BLT and BPE 8B models on downstream tasks**
   - This table compares FLOP-matched BLT and BPE-based models on various downstream tasks. BLT models outperform the Llama 3 model and can enhance FLOP efficiency by adjusting patch sizes.

3. **Table 2: 8B BLT and BPE Llama 3 on tasks assessing robustness to noise and character-level understanding**
   - This table compares BLT and BPE Llama 3 models on tasks evaluating robustness to noise and character-level understanding. BLT models perform better on noisy data and exhibit superior character-level understanding.

4. **Table 3: Performance on translation tasks from FLORES-101**
   - This table compares the performance of BLT and BPE Llama 3 models on FLORES-101 translation tasks. BLT models show better performance, especially in low-resource translation tasks, achieving higher BLEU scores overall compared to BPE models.

5. **Table 4: Ablations on the use of Cross Attention for a 1B BLT model**
   - This table compares various configurations of cross-attention usage in a 1B BLT model. It demonstrates that appropriate use of cross-attention can enhance model performance.

<br/>
# refer format:



**BibTeX:**
```bibtex
@inproceedings{pagnoni2025byte,
  title={Byte Latent Transformer: Patches Scale Better Than Tokens},
  author={Pagnoni, Artidoro and Pasunuru, Ram and Rodriguez, Pedro and Nguyen, John and Muller, Benjamin and Li, Margaret and Zhou, Chunting and Yu, Lili and Weston, Jason and Zettlemoyer, Luke and Ghosh, Gargi and Lewis, Mike and Holtzman, Ari and Iyer, Srinivasan},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={9238--9258},
  year={2025},
  organization={Association for Computational Linguistics}
}
```

**Chicago Style:**
Artidoro Pagnoni, Ram Pasunuru, Pedro Rodriguez, John Nguyen, Benjamin Muller, Margaret Li, Chunting Zhou, Lili Yu, Jason Weston, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Ari Holtzman, and Srinivasan Iyer. "Byte Latent Transformer: Patches Scale Better Than Tokens." In *Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 9238-9258. Association for Computational Linguistics, 2025.
