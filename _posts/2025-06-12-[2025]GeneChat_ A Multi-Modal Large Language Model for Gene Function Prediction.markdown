---
layout: post
title:  "[2025]GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction"  
date:   2025-06-12 01:16:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

다들 한번쯤 생각해보지 않았을까.. 관련 분야 사람이라면    
DNA LM과 LLM의 합?    
dna는 dna lm으로 임베딩화    
그리고 llm인 vicuna13b와 결합  
추가적인 어댑터는 두 lm간의 차원 맞춰주기용  



짧은 요약(Abstract) :    





DNA 서열로부터 유전자 기능을 정확하게 예측하는 것은 유전체학에서 여전히 어려운 과제입니다. 기존 방식은 대부분 유전자를 고정된 범주의 라벨로 분류하는 방식이라 표현력이 제한적입니다. 이에 저자들은 **GeneChat**이라는 멀티모달 대형 언어 모델(LLM)을 제안합니다. 이 모델은 DNA 염기서열과 텍스트 프롬프트를 입력받아 유전자 기능에 대한 자연어 설명을 생성합니다. GeneChat은 (1) DNABERT-2 기반 유전자 인코더, (2) 언어모델 입력 공간과 정렬해주는 어댑터, (3) LLaMA-2 기반 Vicuna-13B 언어모델로 구성되어 있으며, 5만 개 이상의 NCBI 유전자 데이터를 기반으로 학습되었습니다. BLEU 및 METEOR 점수 기준으로 GPT-4o보다 더 정확하고 문맥을 잘 반영한 설명을 생성했으며, 자유 형식 자연어 기반의 유전자 기능 예측이라는 새로운 가능성을 보여줍니다.

---


Accurately predicting gene function from DNA sequences remains a fundamental challenge in genomics, particularly given the limited experimental annotation available for most genes. Existing computational approaches often formulate function prediction as a classification task over predefined categories, limiting their flexibility and expressiveness. We introduce **GeneChat**, a multi-modal large language model designed to generate free-form, natural language descriptions of gene functions directly from nucleotide sequences and textual prompts. GeneChat integrates three components: a DNABERT-2-based gene encoder optimized for long-range genomic context, an adaptor that aligns gene representations with the input space of a large language model, and Vicuna-13B, a fine-tuned LLaMA-2 variant used to produce coherent functional narratives. Trained on over 50,000 genes from the NCBI database, GeneChat outperforms GPT-4o on BLEU and METEOR metrics, demonstrating superior ability to generate accurate, context-aware, and semantically rich descriptions. This work highlights the potential of foundation models for advancing interpretable and scalable gene function prediction in a free-form, language-driven paradigm.





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




###  모델 아키텍처 개요

GeneChat은 다음 세 가지 핵심 구성요소로 이루어진 멀티모달 LLM입니다:

1. **유전자 인코더**:

   * DNABERT-2 사용
   * 입력된 DNA 염기서열을 학습된 표현 벡터로 변환
   * 긴 범위 유전체 문맥(long-range dependencies)을 모델링할 수 있음
   * k-mer 기반 동적 토크나이징 사용

2. **어댑터 (Adaptor)**:

   * DNABERT-2가 출력한 벡터를 LLM의 임베딩 공간에 맞게 선형 변환
   * 변환 수식:

     $$
     h_g = h(x_g) \cdot W
     $$

     여기서 $h(x_g) \in \mathbb{R}^{(l/k)\times256}$, $W \in \mathbb{R}^{256 \times 5120}$

3. **대형 언어 모델 (LLM)**:

   * Vicuna-13B (LLaMA-2 기반 모델)
   * 대화형 스타일로 유전자 기능을 자유 형식(natural language)으로 출력
   * 입력 형식:

     ```
     Human: <Gene> SEQUENCE </Gene> Prompt
     Assistant: Generated answer
     ```

---

###  트레이닝 데이터

* 출처: NCBI 유전자 데이터베이스
* 총 50,248개 유전자 샘플 포함
* 각 항목에 대해 DNA 염기서열 + 기능 설명 문장이 제공됨
* 데이터셋은 95% 학습용, 5% 평가용으로 분할

---

###  학습 수식

* 사용된 학습 목표는 **Causal Language Modeling (CLM)**
* 출력 문장(유전자 기능 설명)을 순차적으로 예측
* 학습 손실 수식:

  $$
  p(x_a | x_g, x_{aux}) = \prod_{i=0}^{l} p_\theta (x_a^{(i)} | x_g, x_{aux}, x_a^{<i})
  $$

  여기서:

  * $x_g$: 유전자 염기서열
  * $x_{aux}$: 텍스트 프롬프트 (e.g., “Predict the function of this gene”)
  * $x_a$: 출력 문장
  * $\theta$: 모델 파라미터

---

###  Model Architecture

GeneChat consists of three main modules:

1. **Gene Encoder**:

   * Based on **DNABERT-2**, pretrained for long-range genomic modeling
   * Extracts contextual representations from input DNA sequences
   * Uses dynamic k-mer tokenization for flexible granularity

2. **Adaptor Module**:

   * Projects gene embeddings into the LLM’s latent space
   * Linear transformation equation:

     $$
     h_g = h(x_g) \cdot W
     $$

     where $h(x_g) \in \mathbb{R}^{(l/k)\times256}$, $W \in \mathbb{R}^{256 \times 5120}$

3. **Language Model (Vicuna-13B)**:

   * Fine-tuned version of LLaMA-2
   * Generates natural language descriptions of gene function
   * Input format:

     ```
     Human: <Gene> SEQUENCE </Gene> Prompt
     Assistant: Answer
     ```

---

###  Training Data

* Source: National Center for Biotechnology Information (NCBI)
* 50,248 gene entries, each with:

  * Nucleotide sequence
  * Textual description of gene function
  * Metadata (e.g., organism, chromosome)
* Data split: 95% training / 5% testing

---

###  Training Objective

* Training used a **causal language modeling** loss:

  $$
  p(x_a | x_g, x_{aux}) = \prod_{i=0}^{l} p_\theta (x_a^{(i)} | x_g, x_{aux}, x_a^{<i})
  $$

  where:

  * $x_g$: input gene sequence
  * $x_{aux}$: auxiliary prompt
  * $x_a$: target output (functional description)
  * $\theta$: model parameters




   
 
<br/>
# Results  




###  비교 모델

* **GeneChat** (제안 모델)
* **GPT-4o** (OpenAI의 최신 범용 LLM)

두 모델을 비교하여 유전자 기능에 대한 자유 형식(natural language) 예측 성능을 평가했습니다.

---

###  테스트 데이터

* **출처**: NCBI (National Center for Biotechnology Information)
* **데이터 수**: 총 50,248개 유전자 중 5% (약 2,500개) 테스트용으로 사용
* 각 유전자 샘플에 대해:

  * DNA 서열
  * 해당 유전자의 기능 설명 (자연어로 된 레퍼런스 정답)

---

###  테스크

* 입력: 유전자 염기서열 + 프롬프트
  예: `"Please predict the function of this gene"`
* 출력: 해당 유전자의 기능을 설명하는 **자연어 문단** 생성
* 비교 대상인 GPT-4o는 두 가지 조건에서 테스트됨:

  1. 유전자 이름만 제공
  2. DNA 서열 전체 제공 (이 경우 의미 있는 출력 생성에 실패함)

---

###  평가 지표 (Metrics)

* **BLEU** (BLEU-1 to BLEU-4)

  * n-gram 중복률 기반의 기계번역 평가 지표
  * BLEU-1: 단어 수준 / BLEU-4: 문장 전체의 정확도 반영
* **METEOR**

  * 정답과의 정렬 정도, 유의어 매칭, 어순 등 고려
  * 인간 판단과의 상관성이 높음

---

### 주요 결과

| 모델           | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| ------------ | ------ | ------ | ------ | ------ | ------ |
| **GeneChat** | 0.1937 | 0.1384 | 0.1065 | 0.0816 | 0.2725 |
| GPT-4o       | 0.1444 | 0.0563 | 0.0208 | 0.0088 | 0.2422 |

* **GeneChat이 모든 지표에서 GPT-4o를 압도적으로 능가**
* 특히 **BLEU-4 기준으로 약 9배 성능 차이**
* GPT-4o는 DNA 서열 입력 시 일관성 없는 결과를 생성함
* GeneChat은 실제 정답과 유사한 생물학적 설명 생성 가능

---

###  Baseline Models

* **GeneChat** (proposed model)
* **GPT-4o** (state-of-the-art general-purpose LLM from OpenAI)

The evaluation compares the models’ ability to **generate free-form natural language descriptions** of gene function.

---

###  Test Dataset

* **Source**: National Center for Biotechnology Information (NCBI)
* **Size**: 5% of 50,248 genes (\~2,500 genes used for testing)
* Each entry includes:

  * Nucleotide sequence (input)
  * Natural language function description (reference output)

---

###  Task Description

* **Input**: DNA sequence + textual prompt
  Example: `"Please predict the function of this gene"`
* **Output**: Natural language paragraph describing gene function
* GPT-4o was tested with:

  1. Gene name only
  2. Full DNA sequence (failed to generate coherent responses)

---

###  Evaluation Metrics

* **BLEU (1-gram to 4-gram)**:

  * Measures n-gram overlap between prediction and ground truth
* **METEOR**:

  * Accounts for precision, recall, synonym matching, and word order
  * Better aligned with human judgment

---

###  Key Results

| Model        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR |
| ------------ | ------ | ------ | ------ | ------ | ------ |
| **GeneChat** | 0.1937 | 0.1384 | 0.1065 | 0.0816 | 0.2725 |
| GPT-4o       | 0.1444 | 0.0563 | 0.0208 | 0.0088 | 0.2422 |

* **GeneChat significantly outperforms GPT-4o** across all metrics.
* The **BLEU-4 score** of GeneChat is nearly **9× higher**, highlighting better contextual coherence.
* GPT-4o fails to handle raw DNA sequences effectively.
* GeneChat generates predictions that align closely with ground truth annotations, offering biologically meaningful insights.




<br/>
# 예제  




논문에서는 **Figure 3**을 통해 GeneChat이 실제 유전자 입력에 대해 생성한 설명을 **정답(annotation)** 과 비교하여 제시하고 있습니다.

###  예시 1: 전사 조절 유전자 (Transcription Factor 관련)

####  정답 (Ground Truth)

> RNA polymerase II 특이적인 DNA 결합 전사 활성화 기능을 수행함. 전사 억제 및 DNA 수리 조절에 관여하며, 세포핵에 위치함.

####  GeneChat 출력

> RNA polymerase II 조절 DNA 서열에 특이적으로 결합하는 기능을 수행함. 전사 억제 및 전사 활성화에 모두 관여하며, 세포핵에 위치함.

** 해설**:
GeneChat은 기능적 맥락(전사 억제/활성화) 및 세포 내 위치(nucleus)를 정확하게 예측했으며, 설명 방식도 인간이 이해하기 쉬운 문장 형태로 제공했습니다.

---

###  예시 2: 조절 염색질 영역 (Regulatory DNA Region)

####  정답 (Ground Truth)

> 이 서열은 ENCODE 프로젝트의 크로마틴 상태 분석을 기반으로 전사 조절 영역으로 예측됨. Sharpr-MPRA 기술로 HepG2 간암세포에서 강한 enhancer로 검증됨.

####  GeneChat 출력

> 이 서열은 ENCODE 프로젝트 분석을 기반으로 전사 조절 영역으로 예측됨. Sharpr-MPRA 실험에서 HepG2와 K562 세포 모두에서 enhancer로 기능함을 보임.

** 해설**:
GeneChat은 정답에 없던 **K562 세포**를 추가로 언급하며 일반화된 설명을 제공. 정보량은 많지만, 일부는 훈련 데이터 기반의 일반화일 수 있음.

---


In **Figure 3**, the paper presents two concrete examples comparing GeneChat's predictions against ground truth annotations.

---

###  Example 1: Transcriptional Regulatory Gene

####  Ground Truth

> Enables RNA polymerase II-specific DNA binding and transcription activator activity. Involved in transcription regulation and DNA repair. Located in the nucleus.

####  GeneChat Prediction

> Enables RNA polymerase II cis-regulatory region sequence-specific DNA binding. Involved in both negative and positive regulation of transcription. Located in the nucleus.

** Commentary**:
GeneChat correctly captures key aspects of gene function, including regulatory role and subcellular localization. Its output mirrors expert-level biological summaries.

---

### Example 2: Chromatin Regulatory Region

####  Ground Truth

> Predicted to be a transcriptional regulatory region based on ENCODE chromatin state analysis. Validated as a strong enhancer in HepG2 liver carcinoma cells via Sharpr-MPRA.

####  GeneChat Prediction

> Predicted as a transcriptional regulatory region based on ENCODE analysis. Validated as an enhancer in both HepG2 and K562 cells using Sharpr-MPRA assays.

** Commentary**:
GeneChat expands the description by including K562 cells, which may reflect generalization based on similar examples in the training data. The prediction remains biologically plausible and detailed.





<br/>  
# 요약   




GeneChat은 DNABERT-2 유전자 인코더와 Vicuna-13B 언어 모델을 어댑터로 연결한 멀티모달 구조로, DNA 서열과 텍스트 프롬프트를 입력받아 유전자 기능을 자연어로 예측한다. NCBI 50,000개 유전자 데이터를 기반으로 학습되었으며, BLEU와 METEOR 지표 모두에서 GPT-4o보다 우수한 성능을 보였다. 실제 예시에서도 전사 인자나 조절 영역에 대해 의미 있고 세부적인 설명을 생성하며 생물학적으로 타당한 출력을 보여준다.

---

GeneChat is a multi-modal architecture that connects a DNABERT-2 gene encoder and a Vicuna-13B language model via an adaptor to generate natural language descriptions of gene function from DNA sequences and prompts. Trained on over 50,000 genes from the NCBI database, it outperforms GPT-4o across BLEU and METEOR metrics. In real examples, GeneChat produces biologically meaningful and detailed outputs, accurately describing transcriptional regulators and enhancer regions.



<br/>  
# 기타  




###  **Figure 1: GeneChat 아키텍처 및 데이터 분포**

* **Figure 1a: GeneChat 구조도**

  * DNABERT-2로 DNA 서열을 임베딩한 후, 어댑터를 통해 Vicuna-13B에 전달
  * 입력 프롬프트와 함께 LLM에 주입하여 자연어 기능 설명 생성
  * 핵심 인사이트: 다양한 생물 종에 적용 가능하며, 유전자 기능을 대화식으로 서술 가능

* **Figure 1b: 학습 및 테스트 데이터의 생물종 분포**

  * 인간(Homo sapiens), 생쥐(Mus musculus), 초파리, 효모 등 포함
  * 테스트 세트도 균형 있게 분포되어 있어 일반화 가능성 확보

---

###  **Figure 2: GeneChat vs. GPT-4o 성능 비교**

* **BLEU 및 METEOR 점수 그래프**

  * GeneChat이 GPT-4o 대비 모든 지표에서 **크게 우수**
  * BLEU-4에서 특히 큰 차이 발생 → 더 긴 문장과 문맥적 정확성에서 강점

**인사이트**:

> 범용 LLM인 GPT-4o는 DNA 서열 처리에 제한이 있으며, GeneChat은 생물학적으로 정제된 표현 생성을 통해 의미 있는 예측을 제공한다는 점이 시각적으로 명확하게 드러남.

---

###  **Figure 3: 실제 예측 예시 (Ground Truth vs. GeneChat Output)**

* 유전자 기능 예측 사례 2개 제시

  1. 전사 인자 기능 설명
  2. 조절 염색질 영역에 대한 설명
* GeneChat의 출력은 정답과 매우 유사하거나 **일부 확장된 정보 포함**

**인사이트**:

> 단순 분류를 넘어서 실제 연구자가 이해할 수 있는 **해석 가능하고 풍부한 자연어 설명**을 제공한다는 점에서 모델의 실용성을 강조함.

---



###  **Figure 1: GeneChat Architecture and Dataset Composition**

* **Figure 1a: Architecture Diagram**

  * DNA sequence is processed by DNABERT-2 → mapped via an adaptor → passed into Vicuna-13B LLM
  * Prompt and gene representation are combined to generate a functional description
  * **Insight**: Flexible architecture that enables natural-language interaction across diverse gene types

* **Figure 1b: Organism Distribution in Dataset**

  * Training/test sets include Homo sapiens, Mus musculus, Drosophila, yeast, etc.
  * Balanced distribution promotes cross-species generalization

---

###  **Figure 2: Performance Comparison (GeneChat vs. GPT-4o)**

* Visual comparison of BLEU and METEOR scores
* GeneChat **consistently outperforms** GPT-4o across all metrics
* Especially large performance gap in BLEU-4 → reflects better fluency and contextual accuracy

**Insight**:

> GPT-4o struggles with raw DNA input, while GeneChat excels at generating biologically aligned, semantically rich functional descriptions.

---

###  **Figure 3: Example Outputs (Ground Truth vs. GeneChat)**

* Two real examples:

  1. Transcription factor-related gene
  2. Chromatin regulatory element
* GeneChat predictions closely match or **extend** the ground truth with additional plausible details

**Insight**:

> Demonstrates GeneChat's strength in producing **interpretable, narrative-style predictions** that resemble expert biological summaries.




<br/>
# refer format:     


@article{dhanasekar2025genechat,
  title     = {GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction},
  author    = {Dhanasekar, Shashi and Saranathan, Akash and Xie, Pengtao},
  journal   = {bioRxiv},
  year      = {2025},
  doi       = {10.1101/2025.06.05.658031},
  url       = {https://www.biorxiv.org/content/10.1101/2025.06.05.658031},
  note      = {Preprint. Not peer-reviewed.}
}



Shashi Dhanasekar, Akash Saranathan, and Pengtao Xie. “GeneChat: A Multi-Modal Large Language Model for Gene Function Prediction.” bioRxiv, June 6, 2025. https://doi.org/10.1101/2025.06.05.658031.  

