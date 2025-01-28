---
layout: post
title:  "[2024]DNA language model GROVER learns sequence context in the human genome"  
date:   2025-01-28 12:08:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    





GROVER(Gene Rules Obtained Via Extracted Representations)은 인간 유전자 서열의 맥락을 학습하기 위해 개발된 DNA 언어 모델로, 기초적인 생물학적 작업에서 뛰어난 성능을 보여줍니다. DNA 서열은 자연 언어와 유사한 규칙을 따르지만, 단어의 개념이 없다는 점에서 차이가 있습니다. 이를 해결하기 위해, 인간 유전체에 바이트-페어 인코딩(byte-pair encoding, BPE)을 적용하여 최적의 어휘를 생성하고, '다음 k-머(k-mer)' 예측 작업으로 어휘를 선정했습니다. GROVER은 서열의 빈도, 내용 및 길이를 포함한 정보를 학습하며, 반복 서열이나 염색질 주석과 같은 생물학적 구조를 순전히 서열 맥락을 통해 학습할 수 있습니다. 이 모델은 정밀 조정 작업에서도 기존 모델들을 능가하며, 유전자 엘리먼트 식별과 단백질-DNA 결합에 대한 생물학적 질문에서 우수한 성능을 발휘합니다. 이는 생명의 코드에 대한 문법책을 작성하는 데 활용될 수 있는 잠재력을 보여줍니다.

---



GROVER (Gene Rules Obtained Via Extracted Representations) is a DNA language model developed to learn sequence context in the human genome, demonstrating high performance in foundational biological tasks. DNA sequences follow rules akin to natural language but differ in lacking a concept of words. To address this, byte-pair encoding (BPE) was applied to the human genome to create an optimal vocabulary, selected via the "next k-mer" prediction task. GROVER learns information such as sequence frequency, content, and length, and can discern biological structures like repeats and chromatin annotations purely through sequence context. The model surpasses existing ones in fine-tuning tasks, including genome element identification and protein–DNA binding, highlighting its potential to write a grammar book for the code of life.



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




이 연구에서 사용된 GROVER 모델은 인간 유전체에 대한 특화된 DNA 언어 모델로, BERT 아키텍처를 기반으로 개발되었습니다. 주요 특징과 장점은 다음과 같습니다:

1. **새로운 구조와 어휘 생성 방식**  
   GROVER은 바이트-페어 인코딩(Byte-Pair Encoding, BPE)을 활용하여 인간 유전체를 토큰화하였습니다. 기존의 k-머(k-mer) 기반 모델들과 달리, BPE는 빈도 기반으로 효율적인 어휘 구성을 가능하게 하며, 불균형한 서열 빈도 문제를 줄여줍니다. 이를 통해 더 복잡한 유전체 정보를 포착할 수 있습니다.

2. **트레이닝 데이터**  
   모델은 인간 유전체 GRCh37(hg19) 조립본 데이터를 사용했습니다. 학습 데이터는 염기(A, C, G, T)로만 구성된 약 500만 개의 샘플로 이루어졌으며, 각 샘플은 최대 510개의 토큰으로 구성됩니다. 80%는 훈련 세트로, 나머지 20%는 테스트 세트로 사용되었습니다.

3. **학습 과정**  
   마스크드 토큰 예측(masked token prediction)을 학습 목표로 설정하여 모델을 훈련했습니다. 입력 서열에서 2.2%의 토큰을 무작위로 선택한 후, 이 중 80%는 마스크 토큰으로 대체하고 나머지는 다른 랜덤 토큰으로 대체하였습니다. 학습은 A100 GPU 클러스터에서 수행되었고, Adam 옵티마이저를 사용하여 학습률 4×10⁻⁴로 진행되었습니다.

4. **특징 및 장점**  
   - **어휘 최적화**: BPE를 통해 최적화된 어휘를 생성하고, '다음 k-머 예측' 작업으로 어휘를 평가하여 가장 적합한 600개의 BPE 사이클을 선택했습니다.
   - **맥락 학습 능력**: GROVER은 단순히 토큰 빈도를 학습하는 것을 넘어, 유전체 서열의 맥락과 생물학적 구조를 포착합니다. 예를 들어, 반복 염기서열이나 염색질 주석과 같은 생물학적 정보를 서열 자체로부터 학습할 수 있습니다.
   - **성능 우수성**: PROM300(프로모터 예측)과 같은 생물학적 작업에서 기존 모델보다 높은 정확도를 보이며, 특히 단백질-DNA 결합 예측과 같은 작업에서도 우수한 성능을 보였습니다.

---
 

The GROVER model introduced in this study is a DNA language model tailored to the human genome, built on the BERT architecture. Its key features and advantages include:

1. **Novel Structure and Vocabulary Creation**  
   GROVER leverages Byte-Pair Encoding (BPE) for tokenizing the human genome. Unlike traditional k-mer-based models, BPE allows efficient vocabulary construction based on sequence frequency, reducing the impact of imbalanced sequence frequencies. This enables the model to capture more complex genomic information.

2. **Training Data**  
   The model was trained on the GRCh37 (hg19) human genome assembly, focusing only on sequences containing the bases A, C, G, and T. The training dataset consisted of over 5 million samples, each up to 510 tokens long, with 80% used for training and 20% for testing.

3. **Training Process**  
   The model was trained using a masked token prediction objective. In the input sequences, 2.2% of tokens were randomly selected, of which 80% were replaced with a special `[MASK]` token, and the rest were replaced with random tokens. Training was conducted on A100 GPU clusters using an Adam optimizer with a learning rate of 4×10⁻⁴.

4. **Features and Advantages**  
   - **Optimized Vocabulary**: BPE was used to create an optimized vocabulary, selecting the best-performing 600-cycle BPE vocabulary based on the "next k-mer prediction" task.
   - **Contextual Learning**: GROVER goes beyond simple token frequency learning to capture the context of genomic sequences and biological structures. For example, it can learn patterns related to repeats or chromatin annotations directly from sequence data.
   - **Superior Performance**: The model outperformed existing models in biological tasks such as promoter prediction (PROM300) and protein–DNA binding prediction, showcasing its robustness in genome analysis tasks.


   
 
<br/>
# Results  



GROVER 모델은 다양한 경쟁 모델과의 비교를 통해 탁월한 성능을 입증했습니다. 주요 결과는 다음과 같습니다:

1. **경쟁 모델**  
   - **DNABERT-2**: 다중 종(genomes of multiple species)을 사용하는 모델로, 유전체 서열의 맥락보다는 서열 빈도 학습에 더 초점을 맞춤.
   - **HyenaDNA**: 암묵적 컨볼루션 구조를 사용하는 DNA 모델.
   - **NT (Nucleotide Transformer)**: k-머 기반의 토큰화를 사용하는 모델.
   - **고정 길이 k-머 모델**: 4-머, 5-머, 6-머를 사용하는 단순 모델.

2. **테스트 데이터**  
   GROVER의 성능은 PROM300(프로모터 예측), PromScan(프로모터 스캐닝), 그리고 CTCF(단백질-DNA 결합 예측) 작업을 포함하여 다양한 생물학적 작업에 대해 평가되었습니다. PROM300은 전사 개시 사이트(TSS) 주변의 프로모터를 실제 프로모터와 셔플된 서열로 분류하는 작업이며, CTCF 예측은 HepG2 세포의 ChIP-seq 데이터를 기반으로 특정 DNA 서열에서의 단백질 결합을 예측하는 작업입니다.

3. **평가 메트릭**  
   모델 성능은 주로 MCC(Matthew’s Correlation Coefficient)와 정확도로 측정되었습니다.

4. **결과 및 성능 차이**  
   - **PROM300 작업**:
     - GROVER: MCC 99.6%
     - 4-머 모델: MCC 79%
     - TF-IDF 모델(5-머): MCC 67%
   - **PromScan 작업**:
     - GROVER: MCC 63%
     - NT 모델: MCC 52%
     - TF-IDF 모델(3-머): MCC 39%
   - **CTCF 단백질-결합 예측**:
     - GROVER: MCC 60%
     - DNABERT-2: MCC 59%
     - TF-IDF 모델(4-머): MCC 26%

5. **성능 차이 분석**  
   GROVER은 특히 '다음 k-머 예측'과 같은 문맥 학습에서 우수한 정확도를 보이며, 생물학적 정보(예: 서열 맥락, 빈도, DNA 물리화학적 특성)를 학습하는 데 뛰어난 능력을 입증했습니다. 또한, TF-IDF와 같은 빈도 기반 모델에 비해 더 정밀한 생물학적 작업 수행 능력을 보여줍니다.

---


The GROVER model demonstrated superior performance compared to several competitive models. Key findings include:

1. **Competitive Models**  
   - **DNABERT-2**: Utilizes genomes from multiple species and focuses more on sequence frequency than context learning.
   - **HyenaDNA**: Employs implicit convolution structures for DNA modeling.
   - **NT (Nucleotide Transformer)**: A k-mer-based model using tokenization strategies.
   - **Fixed-length k-mer models**: Models based on 4-mer, 5-mer, and 6-mer tokenization.

2. **Test Data**  
   GROVER was evaluated on various biological tasks, including PROM300 (promoter prediction), PromScan (promoter scanning), and CTCF (protein–DNA binding prediction). PROM300 classifies promoters around transcription start sites (TSS) as either real promoters or shuffled sequences, while CTCF prediction uses ChIP-seq data from HepG2 cells to predict protein binding at specific DNA sequences.

3. **Evaluation Metrics**  
   Performance was primarily measured using MCC (Matthew’s Correlation Coefficient) and accuracy.

4. **Results and Performance Differences**  
   - **PROM300 Task**:
     - GROVER: MCC 99.6%
     - 4-mer Model: MCC 79%
     - TF-IDF Model (5-mer): MCC 67%
   - **PromScan Task**:
     - GROVER: MCC 63%
     - NT Model: MCC 52%
     - TF-IDF Model (3-mer): MCC 39%
   - **CTCF Protein-Binding Prediction**:
     - GROVER: MCC 60%
     - DNABERT-2: MCC 59%
     - TF-IDF Model (4-mer): MCC 26%

5. **Analysis of Performance Differences**  
   GROVER excels in tasks requiring contextual learning, such as "next k-mer prediction," demonstrating its ability to learn biological information (e.g., sequence context, frequency, and DNA physicochemical properties). It outperforms frequency-based models like TF-IDF and delivers significantly higher precision in biological tasks.



<br/>
# 예제  




#### 1. **트레이닝 데이터 예제**
   **형식:**
   - **입력(Input):** 유전체 서열(예: A, C, G, T)에서 510개의 염기 서열로 토큰화된 데이터.
   - **출력(Output):** 마스크된 토큰의 원래 값(다음 k-머 예측).

   **구체적인 예제:**
   - **입력:** `AATGC [MASK] TCGGA`
   - **출력:** `[MASK] = CAT`
     - 여기서 `[MASK]`는 마스크된 토큰이며, 모델은 이를 정확히 `CAT`으로 예측해야 합니다.
   - **훈련 방법:** 모델은 전체 서열 중 무작위로 선택된 2.2%의 토큰을 마스킹하고 나머지 문맥 정보를 사용하여 마스크된 토큰을 예측하도록 학습합니다.

---

#### 2. **테스트 데이터 예제 및 성능 비교**
   **형식:**
   - **PROM300 작업:** 전사 개시 사이트(TSS) 근처의 서열을 실제 프로모터 또는 셔플된 프로모터로 분류.

   **구체적인 예제:**
   - **입력 서열:** `AAGTCCATGGCTGTCCTGCA...` (TSS -250 ~ +50 염기 서열)
   - **GROVER의 출력:** `Actual promoter` (실제 프로모터로 정확히 분류)
   - **경쟁 모델 출력:**
     - **DNABERT-2:** `Shuffled promoter` (셔플된 프로모터로 잘못 분류)
     - **TF-IDF 모델:** `Shuffled promoter` (빈도 기반 서열에 과적합)

   **설명:** GROVER은 서열의 문맥 정보를 활용하여 실제 프로모터를 정확히 식별한 반면, 경쟁 모델은 빈도 기반 패턴에 의존하여 잘못된 결과를 도출했습니다. PROM300 작업에서 GROVER의 MCC는 99.6%로, DNABERT-2의 79%를 크게 능가했습니다.

---



#### 1. **Training Data Example**
   **Format:**
   - **Input:** Genomic sequence tokenized into 510 bases (e.g., A, C, G, T).
   - **Output:** Original value of the masked token (next k-mer prediction).

   **Specific Example:**
   - **Input:** `AATGC [MASK] TCGGA`
   - **Output:** `[MASK] = CAT`
     - Here, `[MASK]` represents a masked token, and the model predicts it as `CAT`.
   - **Training Method:** The model randomly masks 2.2% of tokens in a sequence and uses the remaining context to predict the masked tokens.

---

#### 2. **Test Data Example and Performance Comparison**
   **Format:**
   - **PROM300 Task:** Classify sequences near transcription start sites (TSS) as either actual promoters or shuffled promoters.

   **Specific Example:**
   - **Input Sequence:** `AAGTCCATGGCTGTCCTGCA...` (TSS -250 to +50 base sequence)
   - **GROVER's Output:** `Actual promoter` (Correctly classified as an actual promoter)
   - **Competitive Models' Outputs:**
     - **DNABERT-2:** `Shuffled promoter` (Incorrectly classified as a shuffled promoter)
     - **TF-IDF Model:** `Shuffled promoter` (Overfit to frequency-based patterns)

   **Explanation:** GROVER successfully identified the actual promoter by leveraging sequence context, whereas competitive models failed due to their reliance on frequency-based patterns. GROVER achieved an MCC of 99.6% in the PROM300 task, significantly outperforming DNABERT-2 (MCC 79%).


<br/>  
# 요약   





GROVER 모델은 인간 유전체를 분석하기 위해 BERT 구조와 바이트-페어 인코딩(BPE)을 사용하여 서열을 토큰화하고, 마스크드 토큰 예측을 통해 학습되었습니다. PROM300(프로모터 예측)과 같은 생물학적 작업에서 평가된 결과, GROVER은 MCC 99.6%로, 경쟁 모델 DNABERT-2의 79%를 크게 능가했습니다. 트레이닝 데이터는 510개 염기 서열로 구성된 입력과 마스크된 토큰의 정답으로 구성되며, 문맥 정보를 기반으로 마스크된 값을 정확히 예측하도록 훈련되었습니다. 예를 들어, `AATGC [MASK] TCGGA` 입력에서 `[MASK]`를 `CAT`으로 예측하며, 테스트 데이터에서는 실제 프로모터를 정확히 분류했습니다. GROVER은 문맥 학습 능력을 통해 경쟁 모델 대비 높은 정확도와 성능을 입증했습니다.

---



The GROVER model, based on the BERT architecture, uses byte-pair encoding (BPE) to tokenize genomic sequences and is trained through masked token prediction. In biological tasks like PROM300 (promoter prediction), GROVER achieved an MCC of 99.6%, significantly outperforming DNABERT-2 (MCC 79%). Training data consists of 510-base sequences as input and the correct value of masked tokens, with the model accurately predicting the masked value using context. For example, given `AATGC [MASK] TCGGA` as input, the model predicts `[MASK]` as `CAT`. In testing, GROVER successfully classified actual promoters, demonstrating superior contextual learning capabilities and outperforming competing models.

<br/>  
# 기타  






1. **Fig. 1: BPE와 모델 아키텍처**  
   이 그림은 GROVER 모델이 사용하는 바이트-페어 인코딩(BPE) 과정을 보여줍니다. 염기(A, C, G, T)에서 시작해 가장 빈도가 높은 쌍을 반복적으로 결합하여 최적의 어휘를 생성합니다. 아키텍처는 BERT 구조를 기반으로 하며, 12개의 트랜스포머 레이어로 구성되어 있습니다.

2. **Fig. 2: 어휘 최적화**  
   BPE의 600 사이클이 가장 적합하다는 것을 보여주는 그래프입니다. `next-k-mer` 예측에서 정확도를 기반으로 어휘를 선택했으며, 다른 k-mer 기반 모델들과 비교했을 때 GROVER은 높은 정확도를 기록했습니다.

3. **Fig. 3: GROVER 어휘의 빈도 균형**  
   GROVER 어휘는 k-mer 빈도와 길이에 따라 다르게 학습 성능을 보였습니다. 예를 들어, 길이가 긴 k-mer는 빈도가 낮지만, 더 많은 생물학적 정보를 포함할 가능성이 큽니다.

4. **Fig. 5: 서열 문맥 학습**  
   이 그림은 GROVER이 문맥 정보를 얼마나 학습했는지 보여줍니다. 코사인 유사도를 통해 동일한 토큰이 다른 문맥에서 나타날 때의 표현 차이를 분석했으며, GROVER은 매우 낮은 유사성을 보여 문맥 학습 능력을 입증했습니다.

5. **Fig. 6: 생물학적 작업에서 성능 비교**  
   PROM300, CTCF 등 작업에서 GROVER의 성능을 경쟁 모델(DNABERT-2, HyenaDNA 등)과 비교한 결과입니다. GROVER은 PROM300에서 99.6%의 MCC로 가장 높은 성능을 기록했습니다.

---



1. **Fig. 1: BPE and Model Architecture**  
   This figure illustrates the byte-pair encoding (BPE) process used in the GROVER model. Starting with nucleotides (A, C, G, T), it iteratively combines the most frequent pairs to generate an optimal vocabulary. The architecture is based on BERT with 12 transformer layers.

2. **Fig. 2: Vocabulary Optimization**  
   The graph shows that 600 cycles of BPE were optimal. Vocabulary selection was based on accuracy in `next-k-mer` prediction. Compared to other k-mer-based models, GROVER demonstrated higher accuracy.

3. **Fig. 3: Frequency-Balanced Vocabulary**  
   The GROVER vocabulary shows differential learning performance based on k-mer frequency and length. Longer k-mers, while less frequent, tend to carry more biological information.

4. **Fig. 5: Sequence Context Learning**  
   This figure demonstrates GROVER's ability to learn sequence context. By measuring cosine similarity across different contexts of the same token, GROVER showed low self-similarity, indicating strong contextual learning.

5. **Fig. 6: Performance in Biological Tasks**  
   This figure compares GROVER's performance on tasks such as PROM300 and CTCF against competitive models (e.g., DNABERT-2, HyenaDNA). GROVER achieved the highest MCC of 99.6% in PROM300, showcasing its superior performance.


<br/>
# refer format:     


@article{sanabria2024grover,
  author       = {Melissa Sanabria and Jonas Hirsch and Pierre M. Joubert and Anna R. Poetsch},
  title        = {DNA language model GROVER learns sequence context in the human genome},
  journal      = {Nature Machine Intelligence},
  volume       = {6},
  pages        = {911--923},
  year         = {2024},
  month        = {August},
  doi          = {10.1038/s42256-024-00872-0}
}



Melissa Sanabria, Jonas Hirsch, Pierre M. Joubert, and Anna R. Poetsch. "DNA Language Model GROVER Learns Sequence Context in the Human Genome." Nature Machine Intelligence 6 (August 2024): 911-923. https://doi.org/10.1038/s42256-024-00872-0.


