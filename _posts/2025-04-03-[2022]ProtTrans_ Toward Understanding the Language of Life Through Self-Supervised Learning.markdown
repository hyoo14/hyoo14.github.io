---
layout: post
title:  "[2022]ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning"  
date:   2025-04-03 01:56:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

프로틴 파운데이션모델 제안(MLM방식 BERT같은->ProtBERT 제안 )

짧은 요약(Abstract) :    





---



이 논문은 자연어 처리(NLP)에서 사용되는 언어 모델(Language Models, LMs)을 단백질 서열에 적용하여 생명 현상의 언어를 이해하려는 시도를 다룹니다. 연구진은 약 3,930억 개의 아미노산으로 이루어진 UniRef 및 BFD 데이터셋을 사용해 총 6개의 언어 모델(2개의 auto-regressive 모델과 4개의 auto-encoder 모델)을 대규모 컴퓨팅 자원을 활용해 훈련시켰습니다. 이렇게 학습된 단백질 언어 모델(pLM)은 라벨이 없는 데이터만으로도 생물물리학적 특징을 어느 정도 포착해낼 수 있었고, 별도의 정렬(MSA)이나 진화 정보 없이도 고성능 예측을 수행했습니다. 특히 ProtT5 모델은 기존 최고 수준의 성능을 능가하며 단백질 2차 구조, 세포 내 위치, 막단백질 여부 등을 예측할 수 있었으며, 이는 pLM이 생명의 언어 문법 일부를 학습했다는 것을 시사합니다.

---



Computational biology and bioinformatics provide vast data gold-mines from protein sequences, ideal for Language Models (LMs) taken from Natural Language Processing (NLP). These LMs reach for new prediction frontiers at low inference costs. Here, we trained two auto-regressive models (Transformer-XL, XLNet) and four auto-encoder models (BERT, Albert, Electra, T5) on data from UniRef and BFD containing up to 393 billion amino acids. The protein LMs (pLMs) were trained on the Summit supercomputer using 5616 GPUs and TPU Pod up-to 1024 cores. Dimensionality reduction revealed that the raw pLM-embeddings from unlabeled data captured some biophysical features of protein sequences. We validated the advantage of using the embeddings as exclusive input for several subsequent tasks: (1) a per-residue (per-token) prediction of protein secondary structure (3-state accuracy Q3=81%-87%); (2) per-protein (pooling) predictions of protein sub-cellular location (ten-state accuracy: Q10=81%) and membrane versus water-soluble (2-state accuracy Q2=91%). For secondary structure, the most informative embeddings (ProtT5) for the first time outperformed the state-of-the-art without multiple sequence alignments (MSAs) or evolutionary information thereby bypassing expensive database searches. Taken together, the results implied that pLMs learned some of the grammar of the language of life. All our models are available through https://github.com/agemagician/ProtTrans.

---



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




다음은 **메서드(Method)** 섹션에서 다룬 내용을 정리한 것입니다.  
**백본 모델, 아키텍처 구성, 트레이닝 데이터**를 중심으로 한 **한글 요약 설명**과 **영문 버전**을 함께 제공합니다.

---



**1. 백본(Backbone) 및 아키텍처:**  
논문에서는 자연어처리(NLP) 분야에서 성공적으로 사용된 언어 모델들을 단백질 서열 학습에 적용했습니다. 사용된 모델들은 다음과 같습니다.

- **Auto-regressive 모델:**  
  - Transformer-XL  
  - XLNet  

- **Auto-encoder 모델:**  
  - BERT  
  - Albert  
  - Electra  
  - T5  

이 중 **ProtBERT, ProtAlbert, ProtElectra**는 bidirectional 아키텍처를 갖춘 auto-encoder이고, **ProtTXL, ProtXLNet**은 unidirectional auto-regressive 모델입니다. **ProtT5**는 encoder-decoder 구조를 가진 유일한 모델입니다.

각 모델은 원래 NLP용으로 설계된 구조를 대부분 유지했으며, 층 수 등의 일부 하이퍼파라미터는 protein sequence 학습에 맞게 조정했습니다.

---

**2. 트레이닝 데이터:**

모델 훈련에는 다음과 같은 대규모 단백질 서열 데이터베이스가 사용되었습니다:

- **UniRef50:** 유사성이 50% 이하인 단백질 군집
- **UniRef100:** 중복 제거 없이 모든 단백질 서열 포함
- **BFD (Big Fantastic Database):** 21억 개 이상의 단백질로 구성된 초대형 데이터베이스

총 **3930억 개의 아미노산 토큰**이 포함된 데이터로, 자연어처리에서 사용되는 대규모 코퍼스보다 훨씬 큽니다. 각 아미노산은 토큰으로 간주되며, 단백질 하나는 문장으로 해석됩니다.

---

**3. 트레이닝 전략:**

- 대부분의 모델은 **self-supervised learning**으로 학습되었습니다.  
- 마스킹된 아미노산을 예측하거나, 시퀀스를 생성하는 방식으로 훈련됩니다.
- **Transformer-XL, ProtT5** 등은 매우 긴 시퀀스도 처리할 수 있도록 설계되어 긴 단백질을 다루는 데 적합합니다.
- 학습은 **Summit 슈퍼컴퓨터(5616 GPU)** 및 **Google TPU Pod(최대 1024 core)**에서 수행되었습니다.

---


**1. Backbone and Architecture:**  
The authors adapted state-of-the-art language model architectures from NLP to protein sequence modeling. They used:

- **Auto-regressive models:**  
  - Transformer-XL  
  - XLNet  

- **Auto-encoder models:**  
  - BERT  
  - Albert  
  - Electra  
  - T5  

Among them, ProtBERT, ProtAlbert, and ProtElectra are bidirectional encoders; ProtTXL and ProtXLNet are unidirectional autoregressive models. ProtT5 is the only encoder-decoder model in the study.  
The original NLP configurations were mostly retained, with some hyperparameter adjustments (e.g., number of layers) to suit biological sequences.

---

**2. Training Data:**  
The models were trained on large-scale protein sequence corpora:

- **UniRef50:** clustered sequences with ≤50% pairwise identity  
- **UniRef100:** all unique sequences without clustering  
- **BFD (Big Fantastic Database):** over 2 billion protein sequences from metagenomic projects  

In total, up to **393 billion amino acids** were used, far surpassing the size of even the largest NLP corpora.  
Each amino acid was treated as a token, and each protein sequence as a sentence.

---

**3. Training Strategy:**  

- All models were trained in a **self-supervised** fashion.  
- Pre-training involved masked token prediction (like BERT) or autoregressive modeling (like XLNet).  
- Some models (e.g., Transformer-XL, ProtT5) can process very long sequences, allowing for modeling entire proteins.  
- Training was conducted on **Summit supercomputer with 5616 GPUs** and **Google TPU Pods with up to 1024 cores**.

---



   
 
<br/>
# Results  




---



**1. 수행한 예측 테스크:**

연구팀은 학습된 단백질 언어 모델(pLM)의 임베딩만을 입력으로 사용하여, 아래 두 수준의 생물학적 예측 작업을 수행했습니다:

- **(1) 단백질 서열의 각 아미노산에 대한 예측 (Per-residue)**  
  → 단백질 2차 구조 예측 (3-state: Q3)

- **(2) 단백질 전체에 대한 예측 (Per-protein)**  
  → 세포 내 위치(subcellular localization, 10-class: Q10)  
  → 막 단백질 여부(membrane vs. soluble, 2-class: Q2)

---

**2. 테스트 데이터셋:**

- **NetSurfP-2.0**: 학습 데이터  
- **CB513, TS115, CASP12**: 기존 벤치마크 테스트셋  
- **NEW364**: 새로운 비중복(Non-redundant) 테스트셋 (2019년 이후 발표된 PDB 기반)

각 데이터셋은 단백질 서열과 실험적으로 얻은 2차 구조 정보를 포함합니다.

---

**3. 비교한 경쟁 모델:**

- **기존 전통적 방법**:  
  - NetSurfP-2.0 (진화정보 기반 MSA 사용)  
  - DeepLoc (위치 예측 SOTA)  
  - LocTree2, JPred4, RaptorX, Spider3 등

- **기계학습 기반 모델**:  
  - DeepSeqVec (ELMo 스타일 pLM)  
  - DeepProtVec (word2vec 스타일)  
  - ESM-1b (Facebook의 대형 pLM)

---

**4. 성능 평가 지표 (Metrics):**

- **Q3**: 3-state 2차 구조 정확도 (per-residue)  
- **Q10**: 10-class 세포 내 위치 분류 정확도 (per-protein)  
- **Q2**: 막 단백질 여부 이진 분류 정확도 (per-protein)

---

**5. 주요 결과 요약:**

- **ProtT5-XL-U50 모델**은 MSA(다중 서열 정렬) 없이도 기존 SOTA인 **NetSurfP-2.0**과 유사한 수준의 정확도를 달성하거나 일부에서 능가함.
- **Q3(2차 구조)**: ProtT5-XL-U50이 81%~87%의 정확도 달성.
- **Q10(위치 예측)**: 약 81%
- **Q2(막 단백질 분류)**: 약 91%
- 특히 **진화 정보가 부족한 단백질 (Neff=1)**에 대해서는 ProtT5 모델이 MSA 기반 모델보다 우수한 성능을 보임.
- **추론 속도** 또한 MSA 기반 접근법 대비 최대 **28배 빠름**.

---


**1. Prediction Tasks:**

The models were evaluated on two levels:

- **Per-residue (amino acid-level) prediction:**  
  - Protein secondary structure (3-state classification, Q3)

- **Per-protein (sequence-level) prediction:**  
  - Subcellular localization (10 classes, Q10)  
  - Membrane vs. soluble classification (2 classes, Q2)

---

**2. Test Datasets:**

- **NetSurfP-2.0** (training set)
- **CB513, TS115, CASP12** (benchmark test sets)
- **NEW364**: A newly constructed non-redundant test set containing proteins added after 2019 with low sequence identity to training data.

---

**3. Competing Models:**

- **Traditional MSA-based methods:**  
  - NetSurfP-2.0 (SOTA for secondary structure prediction)  
  - DeepLoc (SOTA for localization)  
  - Other tools: JPred4, RaptorX, Spider3, LocTree2

- **ML-based baselines:**  
  - DeepSeqVec (ELMo-based pLM)  
  - DeepProtVec (word2vec-style embedding)  
  - ESM-1b (Transformer-based protein LM)

---

**4. Evaluation Metrics:**

- **Q3:** Accuracy for 3-state secondary structure prediction  
- **Q10:** Accuracy for 10-class subcellular localization  
- **Q2:** Accuracy for 2-class membrane vs. soluble prediction

---

**5. Key Findings:**

- **ProtT5-XL-U50**, trained without MSA or evolutionary info, achieved **comparable or superior** performance to the state-of-the-art method **NetSurfP-2.0**.
- **Q3** reached 81%–87%, **Q10** about 81%, and **Q2** about 91%.
- Notably, for proteins with **little or no evolutionary information** (e.g., Neff = 1), ProtT5 models **outperformed** MSA-dependent models.
- **Inference speed** was also significantly better: up to **28× faster** than traditional MSA-based pipelines.

---




<br/>
# 예제  




---



**1. 트레이닝 데이터 예시 (Training Data Example):**

- **입력 형식(Input):**  
  각 단백질 서열은 아미노산 1글자 코드로 이루어진 문자열입니다.  
  예를 들어:

  ```
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQNLARTQ
  ```

- **전처리:**  
  - 각 아미노산은 하나의 토큰으로 간주됨.  
  - 시퀀스 간 구분은 공백 줄로 처리됨.  
  - `[X]`는 알 수 없는 아미노산으로 매핑됨.  
  - NLP처럼 BERT-style 마스킹(`[MASK]`)도 사용됨.

---

**2. 테스트 데이터 예시 (Test Data Example):**

- **예시 단백질 시퀀스:**

  ```
  MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDR
  ```

- **부가 정보 (ground truth):**  
  이 시퀀스에 대해 실제로는 2차 구조 정보(예: H, E, C)가 있음.  
  예를 들어:

  ```
  HHHHHHCCCCCEEEECCCCCCCCCCCCCCCCCEEEECCCCCCC
  ```

  - H: alpha-helix  
  - E: beta-strand  
  - C: coil

---

**3. 예측 테스크 인풋/아웃풋 예시 (Input/Output of Tasks):**

| Task Type              | Input Example                                                                                      | Output Example                                                   |
|------------------------|---------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| Per-residue 구조 예측   | `MKTAYIAKQRQISFVK...` (단백질 서열)                                                               | `CCCHHHHHHHHCCCCEEE...` (2차 구조: C, H, E)                      |
| Per-protein 위치 예측   | `MGLSDGEWQLVLNVWGKVEADIPGH...` (단백질 서열 전체)                                                  | `Cytoplasm` 또는 `Plasma Membrane` (10개 중 하나)               |
| Per-protein 막 여부 예측 | `MKQLEDKVEELLSKNYHLENEVARLKKLV`                                                                  | `Membrane` 또는 `Soluble` (2개 중 하나)                          |

---



**1. Training Data Example:**

- **Input Format:**  
  Each protein is a string of single-letter amino acid codes.  
  For example:

  ```
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQANLQNLARTQ
  ```

- **Preprocessing:**  
  - Each amino acid is treated as a token.  
  - Sequences are separated by blank lines.  
  - Unusual or unknown amino acids (e.g., `B`, `Z`, `X`) are mapped to `X`.  
  - BERT-style masked tokens (e.g., `[MASK]`) are used for training.

---

**2. Test Data Example:**

- **Example protein sequence:**

  ```
  MGLSDGEWQLVLNVWGKVEADIPGHGQEVLIRLFKGHPETLEKFDR
  ```

- **Ground truth label:**  
  The true secondary structure is provided as a string of labels like:

  ```
  HHHHHHCCCCCEEEECCCCCCCCCCCCCCCCCEEEECCCCCCC
  ```

  - H: alpha-helix  
  - E: beta-sheet  
  - C: coil

---

**3. Task Input/Output Examples:**

| Task Type               | Input Example                                                                                     | Output Example                                                    |
|-------------------------|--------------------------------------------------------------------------------------------------|-------------------------------------------------------------------|
| Per-residue prediction  | `MKTAYIAKQRQISFVK...` (protein sequence)                                                         | `CCCHHHHHHHHCCCCEEE...` (secondary structure: C, H, E)            |
| Per-protein localization| `MGLSDGEWQLVLNVWGKVEADIPGH...` (full protein sequence)                                           | `Cytoplasm` or `Plasma Membrane` (1 of 10 classes)               |
| Per-protein membrane    | `MKQLEDKVEELLSKNYHLENEVARLKKLV`                                                                  | `Membrane` or `Soluble` (binary classification)                  |

---





<br/>  
# 요약   




이 논문은 BERT, T5 등 최신 NLP 백본을 활용하여 대규모 단백질 서열(3930억 아미노산)을 기반으로 self-supervised 방식으로 ProtTrans 모델들을 학습했다.  
이 모델들은 MSA 없이도 단백질 2차 구조, 세포 내 위치, 막 단백질 여부 등을 정확히 예측하며, 기존 SOTA보다 빠르고 경쟁력 있는 성능을 보였다.  
예측은 단백질 서열을 입력으로 받아 아미노산 단위의 구조 또는 단백질 단위의 기능 정보를 출력하는 방식으로 진행된다.

---



This paper presents ProtTrans, a family of models trained on 393 billion amino acids using state-of-the-art NLP backbones like BERT and T5 with self-supervised learning.  
The models achieve competitive or superior performance compared to SOTA methods for tasks such as secondary structure prediction, subcellular localization, and membrane classification—all without using MSAs.  
Predictions are made by inputting protein sequences and outputting either residue-level structural labels or sequence-level functional classifications.


<br/>  
# 기타  


좋아요! 아래는 해당 논문에 포함된 **기타 구성 요소들** — 예를 들어 **피규어, 테이블, 어펜딕스 등** —에 대한 요약 설명입니다.  
각 구성 요소가 논문에서 어떤 정보를 전달하는지 **한글 설명**과 **영문 버전**으로 함께 정리했어요.

---


**1. 피규어 (Figures):**

- **Fig. 1:** 전체 실험 설계 개요를 보여줍니다.  
  → 어떤 모델들이 사용되었고, 어떤 종류의 데이터와 테스크에 적용되었는지 시각적으로 설명합니다.

- **Fig. 2:** t-SNE를 활용해 학습된 임베딩이 구조적/기능적 특성에 따라 어떻게 클러스터링되는지를 보여줍니다.  
  → 예: 위치별 단백질 임베딩이 공간적으로 분리됨.

- **Fig. 3:** 단백질 서열 길이에 따른 모델 성능 변화를 시각화.  
  → 긴 서열에서도 ProtT5가 안정적인 성능을 유지함을 보여줌.

---

**2. 테이블 (Tables):**

- **Table 1:** 각 모델 별 파라미터 수, 학습된 토큰 수, 사용한 데이터셋 등을 비교한 요약표입니다.

- **Table 2:** 2차 구조 예측 정확도(Q3)를 기존 방법들과 비교한 성능 테이블입니다.  
  → ProtT5가 MSA 없이도 기존 모델과 대등하거나 더 나은 성능을 보임.

- **Table 3:** 세포 내 위치 예측 정확도(Q10), 막 단백질 여부(Q2) 비교 표.  
  → 기존 DeepLoc, LocTree2 등과 비교하여 ProtTrans 모델의 효율성과 정확도 강조.

---

**3. 부록 (Appendix):**

- 모델 학습에 사용된 **하드웨어 사양(Summit, TPU Pods)**, **학습 시간**, **하이퍼파라미터 설정**,  
  **추론 속도 비교**, **데이터 정제 방법** 등을 상세히 다룹니다.

- 또한 각 테스크에 사용된 **데이터셋 출처**, **분할 기준**, **모델의 파인튜닝 전략**도 설명되어 있어  
  재현 연구에 유용한 정보를 포함하고 있습니다.

---


**1. Figures:**

- **Figure 1:** Provides an overview of the full experimental pipeline, detailing which models were trained and how they were evaluated across tasks.

- **Figure 2:** Shows a t-SNE projection of the learned embeddings, highlighting that proteins with similar functions or locations cluster together.

- **Figure 3:** Visualizes model performance by protein sequence length, indicating that ProtT5 maintains stable accuracy even for long sequences.

---

**2. Tables:**

- **Table 1:** Summarizes each model's architecture, number of parameters, training data size, and total tokens processed.

- **Table 2:** Presents Q3 scores (secondary structure accuracy) across models, showing that ProtT5 achieves comparable or better performance than MSA-based models.

- **Table 3:** Compares Q10 (localization) and Q2 (membrane vs. soluble) accuracy across several baselines like DeepLoc and LocTree2, highlighting ProtTrans efficiency.

---

**3. Appendix:**

- Includes technical details such as **hardware used (Summit, TPUs)**, **training time**, **hyperparameters**,  
  **inference speed**, and **data preprocessing steps**.

- Also provides dataset sources, splitting strategy, and fine-tuning setups, making the paper easier to replicate and extend.

---



<br/>
# refer format:     



@article{elnaggar2022prottrans,
  title={ProtTrans: Toward Understanding the Language of Life Through Self-Supervised Learning},
  author={Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Wang, Yu and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={44},
  number={10},
  pages={7112--7127},
  year={2022},
  publisher={IEEE},
  doi={10.1109/TPAMI.2021.3095381}
}





Elnaggar, A., Heinzinger, M., Dallago, C., Rehawi, G., Wang, Y., Jones, L., Gibbs, T., Feher, T., Angerer, C., Steinegger, M., Bhowmik, D., & Rost, B. (2022). ProtTrans: Toward understanding the language of life through self-supervised learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 44(10), 7112–7127. https://doi.org/10.1109/TPAMI.2021.3095381  
