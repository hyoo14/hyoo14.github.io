---
layout: post
title:  "[2025]Controllable Protein Sequence Generation with LLM Preference Optimization"  
date:   2025-03-02 14:10:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

파인튜닝한 프로틴 LLM?  좀 특이한 테스크로 파인튜닝(new multi-listwise preference optimization strategy)  

짧은 요약(Abstract) :    



단백질 설계는 특정 기능을 갖춘 단백질을 생성하여 생의학적 문제를 해결하는 중요한 방법이다. 사전 학습된 단백질 대형 언어 모델(LLM)은 단백질 서열 생성에서 유망한 결과를 보였으나, 특정 속성을 제어하는 기존 방법들은 기능성과 구조적 안정성이 부족한 문제가 있었다. 이를 해결하기 위해, 본 연구에서는 **CtrlProt**이라는 새로운 단백질 설계 방법을 제안한다. 이 방법은 **멀티 리스트 방식의 선호 최적화 기법**을 적용하여 단백질 LLM을 미세 조정(finetuning)함으로써, 생성 품질을 향상시키고 다중 속성 제어를 가능하게 한다. 실험 결과, **CtrlProt**은 기능성과 구조적 안정성 요구사항을 효과적으로 충족하며, 단일 속성과 다중 속성 단백질 서열 생성에서 최첨단 성능을 달성함을 확인하였다.

---



**Controllable Protein Sequence Generation with LLM Preference Optimization**

Designing proteins with specific attributes offers an important solution to address biomedical challenges. Pre-trained protein large language models (LLMs) have shown promising results on protein sequence generation. However, to control sequence generation for specific attributes, existing work still exhibits poor functionality and structural stability. In this paper, we propose a novel controllable protein design method called **CtrlProt**. We finetune a protein LLM with a **new multi-listwise preference optimization strategy** to improve generation quality and support multi-attribute controllable generation. Experiments demonstrate that **CtrlProt** can meet functionality and structural stability requirements effectively, achieving state-of-the-art performance in both single-attribute and multi-attribute protein sequence generation.



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






**1. 모델 개요**  
본 연구에서는 **CtrlProt**이라는 새로운 단백질 서열 생성 모델을 제안한다. 이 모델은 사전 학습된 단백질 대형 언어 모델(Protein Large Language Model, LLM)을 미세 조정(finetuning)하여 **특정 속성을 제어할 수 있는 단백질을 생성**하는 것을 목표로 한다. 특히, **멀티 리스트 방식의 선호 최적화(multi-listwise preference optimization)**를 적용하여, 기능성과 구조적 안정성을 동시에 고려한 학습 전략을 도입하였다.

---

**2. 모델 아키텍처 및 학습 과정**  

**(1) 사전 학습된 단백질 언어 모델 (Protein LLM) 미세 조정**  
- 특정 단백질 속성(예: 금속 이온 결합, 인산화, 세포 내 위치 등)을 가진 단백질 서열을 학습하여 속성별 단백질 서열 생성을 가능하게 함.  
- **Prefix-tuning 기법**을 사용하여 속성별 프리픽스(prefix)를 적용하고, 이를 바탕으로 단백질 서열을 생성함.  
- 학습 과정에서 **음의 로그 가능도(Negative Log-Likelihood, NLL) 손실 함수**를 사용하여 모델을 최적화함.  

**(2) 선호 최적화 (Preference Optimization) 적용**  
- **기능성과 구조적 안정성을 평가하는 두 가지 지표**를 설계하여 최적화에 활용:
  1. **구조적 안정성(Structural Stability):**  
     - 단백질의 접힘 안정성을 평가하기 위해 **Rosetta Energy Score**를 사용하며, 더 낮은 에너지 값일수록 안정적인 구조임.  
  2. **기능적 유사성(Functional Similarity):**  
     - 사전 학습된 단백질 구조 인코더(ProteinMPNN)를 이용하여 서열의 기능적 유사성을 평가함.  

- **다중 리스트 방식의 선호 최적화(Multi-Listwise Preference Optimization)**  
  - 평가 지표(기능성과 구조적 안정성)를 기반으로 샘플을 순위화하여 학습 데이터를 구성.  
  - **더 높은 품질의 서열을 선호하는 방식으로 학습을 진행**, 모델이 낮은 품질의 단백질 서열을 생성하지 않도록 함.  
  - **정규화 항(Regularization Term)**을 추가하여, 모델이 고품질 서열과 저품질 서열의 차이를 명확하게 인식하도록 유도.  

---

**3. 학습 데이터 (Training Data)**  

**(1) 단백질 데이터셋 구축**  
- **UniProtKB 데이터베이스**에서 특정 **Gene Ontology (GO) 속성을 가진 단백질 서열**을 추출하여 학습 데이터로 사용.  
- **AlphaFold 단백질 구조 데이터베이스**에서 해당 서열의 3D 구조 정보를 수집하여 기능성과 구조적 안정성 평가에 활용.  
- **6가지 속성의 단백질 서열**을 학습에 사용:
  1. **Molecular Function (MFO)**: 금속 이온 결합(Metal Ion Binding), RNA 결합(RNA Binding)  
  2. **Biological Process (BPO)**: 인산화(Phosphorylation), 번역(Translation)  
  3. **Cellular Component (CCO)**: 세포질(Cytoplasm), 핵(Nucleus)  

**(2) 데이터 구성**  
- 속성별 **10,000개 단백질 서열**을 학습에 사용.  
- **평가 데이터셋**: 100,000개 단백질 서열을 별도로 구축하여 모델의 성능을 검증.  
- **구조적 유사성 평가를 위한 데이터베이스** 생성: Foldseek을 사용하여 평가 데이터를 정리.  

---

**4. 모델 학습 및 평가 설정**  

- **Prefix-Tuning 설정**  
  - 모델: **ProtGPT2**  
  - 배치 크기: 16  
  - 학습률: 1e-4  
  - 프리픽스 토큰 수: 100  

- **선호 최적화(Preference Optimization) 설정**  
  - 속성당 5,000개 데이터 쌍 사용  
  - 학습률: 5e-5  
  - 하이퍼파라미터:  
    - β = 0.1  
    - α = 0.05  

- **구조 예측 및 평가**  
  - **ProteinMPNN**을 구조 인코더로 사용  
  - **ESMFold**를 이용하여 예측된 단백질 구조 평가  
  - **Rosetta Energy Score**를 계산하여 구조적 안정성 확인  

---



**1. Overview of the Model**  
This study introduces **CtrlProt**, a novel protein sequence generation model designed to produce proteins with controllable attributes. The model fine-tunes a **pre-trained protein large language model (LLM)** and applies **multi-listwise preference optimization** to improve both functionality and structural stability of the generated sequences.

---

**2. Model Architecture and Training Process**  

**(1) Fine-tuning a Pre-trained Protein LLM**  
- The model learns from protein sequences associated with specific attributes (e.g., metal ion binding, phosphorylation, cellular localization).  
- Uses **prefix-tuning** to apply attribute-specific prefixes, enabling targeted protein sequence generation.  
- Optimized using **negative log-likelihood (NLL) loss** to enhance sequence relevance.  

**(2) Applying Preference Optimization**  
- Two key evaluation metrics were designed to assess generated protein sequences:
  1. **Structural Stability:**  
     - **Rosetta Energy Score** is used to evaluate folding stability; lower scores indicate more stable structures.  
  2. **Functional Similarity:**  
     - **ProteinMPNN encoder** extracts structural representations to measure functional relevance.  

- **Multi-listwise Preference Optimization**  
  - Ranks samples based on their evaluation scores (functionality and structural stability).  
  - Ensures that **higher-quality sequences are prioritized** while discouraging low-quality sequences.  
  - Introduces **a regularization term** to distinguish between high- and low-quality samples effectively.  

---

**3. Training Data**  

**(1) Protein Dataset Construction**  
- **UniProtKB database** was used to extract protein sequences annotated with **Gene Ontology (GO) terms**.  
- **AlphaFold protein structure database** was used to retrieve corresponding 3D structural data for evaluation.  
- **Six attributes were selected** for training:
  1. **Molecular Function (MFO):** Metal ion binding, RNA binding  
  2. **Biological Process (BPO):** Phosphorylation, Translation  
  3. **Cellular Component (CCO):** Cytoplasm, Nucleus  

**(2) Dataset Composition**  
- **10,000 protein sequences per attribute** were used for training.  
- **100,000 sequences** were collected separately for evaluation.  
- **Evaluation database for structural similarity** was built using Foldseek.  

---

**4. Training and Evaluation Settings**  

- **Prefix-Tuning Settings**  
  - Model: **ProtGPT2**  
  - Batch size: 16  
  - Learning rate: 1e-4  
  - Prefix tokens: 100  

- **Preference Optimization Settings**  
  - 5,000 data pairs per attribute  
  - Learning rate: 5e-5  
  - Hyperparameters:  
    - β = 0.1  
    - α = 0.05  

- **Structure Prediction and Evaluation**  
  - **ProteinMPNN** used as the structural encoder.  
  - **ESMFold** for predicting protein structures.  
  - **Rosetta Energy Score** calculated for structural stability assessment.  

---

This method ensures that **CtrlProt** generates protein sequences with improved functional relevance and structural stability, achieving state-of-the-art performance in both **single-attribute and multi-attribute** protein design.

   
 
<br/>
# Results  



**1. 경쟁 모델 (Baseline Models)**  
**CtrlProt**의 성능을 평가하기 위해 다양한 기존 단백질 서열 생성 모델과 비교하였다. 비교 대상은 다음과 같다:  

1. **ESM-1b** [(Rives et al., 2021)]  
   - **엔코더 기반** 단백질 언어 모델로, 단백질 서열의 의미적 정보를 학습.  
2. **ESM-2** [(Lin et al., 2023)]  
   - **ESM-1b의 확장 모델**로, 더 높은 정확도로 단백질 구조 및 기능을 예측.  
3. **EvoDiff** [(Alamdari et al., 2023)]  
   - **확산 모델(Diffusion Model)** 기반 단백질 서열 생성 모델.  
4. **PrefixProt** [(Luo et al., 2023)]  
   - **Prefix-tuning**을 활용하여 ProtGPT2를 미세 조정한 모델.  
5. **ProGen2** [(Nijkamp et al., 2023)]  
   - **대규모 단백질 서열을 사전 학습**한 디코더 기반 모델.  
6. **ProLLaMA** [(Lv et al., 2024)]  
   - **자연어 정렬(LM alignment) 접근법**을 사용하여 단백질 서열을 생성하는 최신 모델.  

---

**2. 테스트 데이터 (Test Dataset)**  
- **UniProtKB** 데이터베이스에서 **6가지 속성**을 포함하는 **100,000개 단백질 서열**을 추출하여 평가에 사용.  
- 속성별 **10,000개 단백질 서열**을 학습 데이터로 활용.  
- **AlphaFold 단백질 구조 데이터베이스**를 활용하여 3D 단백질 구조 예측을 수행.  
- **Foldseek**을 사용하여 생성된 단백질 서열과 실제 단백질 서열 간의 유사성을 분석.  

---

**3. 평가 메트릭 (Evaluation Metrics)**  
단백질 서열의 품질을 평가하기 위해 4가지 주요 메트릭을 사용하였다:  

1. **CLS-score**  
   - **ESM-2 모델을 활용한 분류기(Classifier) 점수**로, 단백질이 특정 속성을 가지는지 평가.  
   - 높은 CLS-score일수록 속성 관련성이 높음을 의미.  

2. **TM-score (Template Modeling Score)**  
   - **Foldseek을 사용하여 구조적 유사성을 평가**하는 점수.  
   - 0~1 사이의 값으로, 1에 가까울수록 높은 구조적 유사성을 의미.  

3. **RMSD (Root Mean Square Deviation)**  
   - **단백질 구조 간 차이를 측정하는 지표**, 값이 낮을수록 더 유사한 구조를 의미.  

4. **pLDDT (Predicted Local Distance Difference Test)**  
   - **AlphaFold의 구조 예측 신뢰도 점수**, 값이 높을수록 신뢰도가 높음.  

---

**4. 실험 결과 (Results)**  

- **단일 속성 단백질 생성 (Single-Attribute Generation) 결과**  
  - **CtrlProt**은 경쟁 모델보다 **더 높은 TM-score와 CLS-score, 낮은 RMSD, 높은 pLDDT**를 달성.  
  - 특히, **pLDDT**에서 큰 향상을 보여 **CtrlProt이 구조적으로 안정적인 단백질을 생성**함을 입증.  
  - **ProLLaMA** 모델은 긴 서열 생성에서 성능 저하를 보였으며, **EvoDiff**는 비교적 높은 RMSD를 기록함.  

- **다중 속성 단백질 생성 (Multi-Attribute Generation) 결과**  
  - 단일 속성을 학습한 후 **멀티 속성 프리픽스(Prefix)를 결합하여** 다중 속성을 가진 단백질을 생성.  
  - 기존의 **단순 프리픽스 병합(Average, Concat) 방식보다 높은 성능**을 보였으며, **구조적 안정성과 기능성 모두에서 향상**됨.  

- **다양성 평가 (Diversity Analysis)**  
  - **CtrlProt은 훈련 데이터에 과적합(overfitting)되지 않고 새로운 서열을 생성**함을 입증.  
  - **n-gram 분석**을 통해 기존 데이터와의 중복도를 측정한 결과, **자연 단백질(Natural Proteins)과 유사한 수준의 다양성**을 유지하면서도 학습된 패턴을 따르는 것을 확인.  

---



**1. Baseline Models**  
To evaluate the performance of **CtrlProt**, it was compared against multiple existing protein sequence generation models:  

1. **ESM-1b** [(Rives et al., 2021)]  
   - **Encoder-based** protein language model trained for understanding sequence semantics.  
2. **ESM-2** [(Lin et al., 2023)]  
   - **An extended version of ESM-1b** with improved accuracy in structure and function prediction.  
3. **EvoDiff** [(Alamdari et al., 2023)]  
   - A **diffusion-based** model for protein sequence generation.  
4. **PrefixProt** [(Luo et al., 2023)]  
   - A **prefix-tuning** approach applied to ProtGPT2 for fine-tuning.  
5. **ProGen2** [(Nijkamp et al., 2023)]  
   - **Decoder-based model** pre-trained on large-scale protein sequence data.  
6. **ProLLaMA** [(Lv et al., 2024)]  
   - A recent model leveraging **language model alignment** to generate protein sequences.  

---

**2. Test Data**  
- **100,000 protein sequences** with **six specific attributes** were extracted from the **UniProtKB** database for evaluation.  
- **10,000 protein sequences per attribute** were used for training.  
- **AlphaFold protein structure database** was used for 3D structure prediction.  
- **Foldseek** was employed to measure structural similarity between generated and natural protein sequences.  

---

**3. Evaluation Metrics**  
Four key evaluation metrics were used to assess the quality of generated protein sequences:  

1. **CLS-score**  
   - Classification score obtained using **ESM-2**, evaluating whether the generated sequence exhibits the desired attributes.  

2. **TM-score (Template Modeling Score)**  
   - Measures **structural similarity** between generated and reference proteins.  
   - Ranges from 0 to 1, with higher values indicating higher similarity.  

3. **RMSD (Root Mean Square Deviation)**  
   - Measures **structural deviation**, where **lower values** indicate higher similarity.  

4. **pLDDT (Predicted Local Distance Difference Test)**  
   - Confidence score from **AlphaFold**, where **higher values** indicate more reliable structure predictions.  

---

**4. Experimental Results**  

- **Single-Attribute Protein Generation Results**  
  - **CtrlProt** achieved **higher TM-score and CLS-score, lower RMSD, and higher pLDDT** than baseline models.  
  - Particularly, **pLDDT** improvements indicate that **CtrlProt generates structurally stable proteins**.  
  - **ProLLaMA** struggled with generating long sequences, while **EvoDiff** had a higher RMSD.  

- **Multi-Attribute Protein Generation Results**  
  - **CtrlProt** used a **prefix concatenation** approach to generate proteins with multiple attributes.  
  - It **outperformed naive prefix merging methods (Average, Concat)** and achieved **better structural stability and functionality**.  

- **Diversity Analysis**  
  - **CtrlProt did not overfit to training data and generated novel sequences.**  
  - **n-gram analysis** confirmed that it maintained **diversity similar to natural proteins**, while still adhering to learned patterns.  

---

These results demonstrate that **CtrlProt** effectively generates **high-quality, controllable protein sequences** with **state-of-the-art performance in both single-attribute and multi-attribute protein design**.


<br/>
# 예제  

**1. 트레이닝 데이터 예제 (Training Data Example)**  
CtrlProt 모델 학습을 위해 **UniProtKB 데이터베이스**에서 특정 **Gene Ontology (GO) 속성**을 포함하는 단백질 서열을 추출하였다. 예제 데이터는 다음과 같다:

```text
>Protein_001 | Function: Metal Ion Binding | Source: UniProtKB
MKTFFIICLVVVKDSDGKAGGAASVVGFAAKTAEEGTLHGEQFGGAFPYGDILRHVN
```

```text
>Protein_002 | Function: RNA Binding | Source: UniProtKB
MSTTVKRHLIYEDTKEIRLRAKYGQGELLGFDKDGRYASIFHYLKKDVSGDVVKPV
```

각 단백질 서열에는 특정 **생물학적 기능(Biological Function)**이 포함되어 있으며, **프리픽스 튜닝(Prefix-Tuning)**을 사용하여 모델을 학습함.

---

**2. 실험 데이터 예제 (Experimental Data Example)**  
학습된 모델의 성능을 평가하기 위해 **AlphaFold 단백질 구조 데이터베이스**에서 예측된 3D 구조를 포함하는 데이터를 사용하였다.  
예제 데이터는 다음과 같다:

```text
>Experimental_001 | Function: Metal Ion Binding | Predicted Structure: AlphaFold
MKTYVQICLLIYTVASGAGSGKGAGGTIVGGFAAKTAEEGKLHGEKFGGAFPYGDILRHVN
```

이러한 데이터를 사용하여 **Foldseek을 활용한 구조적 유사성 평가**를 수행하였다.

---

**3. 태스크 입력/출력 예제 (Task Input/Output Example)**  

**입력 (Input):**  
CtrlProt은 특정 속성을 가진 단백질 서열을 생성하는 태스크를 수행한다.  

```text
Task: Generate a protein sequence with "Phosphorylation" function.
```

**출력 (Output):**  
모델이 생성한 단백질 서열 예제:

```text
>Generated_Protein | Function: Phosphorylation
MSTAVKRHLIYEDTKEIRLRAKYGQGELLGFDKDGRYASIFHYLKKDVSGDVVKPVQTVG
```

이 서열은 **기능성과 구조적 안정성을 최적화한 결과**이며, TM-score, RMSD, pLDDT 등의 메트릭을 활용하여 평가됨.

---



**1. Training Data Example**  
The **CtrlProt** model was trained using protein sequences extracted from the **UniProtKB** database, containing specific **Gene Ontology (GO) attributes**. Example data:

```text
>Protein_001 | Function: Metal Ion Binding | Source: UniProtKB
MKTFFIICLVVVKDSDGKAGGAASVVGFAAKTAEEGTLHGEQFGGAFPYGDILRHVN
```

```text
>Protein_002 | Function: RNA Binding | Source: UniProtKB
MSTTVKRHLIYEDTKEIRLRAKYGQGELLGFDKDGRYASIFHYLKKDVSGDVVKPV
```

Each protein sequence is labeled with a specific **biological function**, and the model is trained using **prefix-tuning**.

---

**2. Experimental Data Example**  
To evaluate the trained model, **predicted 3D structures from the AlphaFold protein structure database** were used.  
Example data:

```text
>Experimental_001 | Function: Metal Ion Binding | Predicted Structure: AlphaFold
MKTYVQICLLIYTVASGAGSGKGAGGTIVGGFAAKTAEEGKLHGEKFGGAFPYGDILRHVN
```

These data were used for **structural similarity assessment using Foldseek**.

---

**3. Task Input/Output Example**  

**Input:**  
CtrlProt performs a task to generate a protein sequence with a specified attribute.

```text
Task: Generate a protein sequence with "Phosphorylation" function.
```

**Output:**  
Example of a generated protein sequence:

```text
>Generated_Protein | Function: Phosphorylation
MSTAVKRHLIYEDTKEIRLRAKYGQGELLGFDKDGRYASIFHYLKKDVSGDVVKPVQTVG
```

This sequence is **optimized for functionality and structural stability**, evaluated using **TM-score, RMSD, and pLDDT** metrics.


<br/>  
# 요약   




CtrlProt은 단백질 서열 생성을 위해 **멀티 리스트 방식의 선호 최적화(Multi-Listwise Preference Optimization)**를 적용하여 기능성과 구조적 안정성을 동시에 향상시킨다. 실험 결과, CtrlProt은 단일 및 다중 속성 단백질 생성에서 기존 모델보다 **더 높은 TM-score, CLS-score, pLDDT, 더 낮은 RMSD**를 달성하였다. 모델이 생성한 단백질 서열은 학습 데이터와 차별화된 **새로운 서열이지만 기능적으로 유사한 구조**를 가지며, 다양한 생물학적 속성을 반영할 수 있음을 입증하였다.  

---


CtrlProt enhances protein sequence generation by applying **multi-listwise preference optimization**, improving both **functionality and structural stability**. Experimental results show that CtrlProt outperforms baseline models in **single- and multi-attribute protein generation**, achieving **higher TM-score, CLS-score, pLDDT, and lower RMSD**. The generated protein sequences are **novel yet structurally and functionally relevant**, demonstrating the model’s ability to reflect various biological attributes.


<br/>  
# 기타  



**1. 방법론 개요 다이어그램 (Method Overview Diagram - CtrlProt 구조)**  
- **Figure 2: CtrlProt 아키텍처 개요**  
  - **Prefix-tuning을 통한 단백질 LLM 미세 조정(Fine-tuning)** 후, **기능성과 구조적 안정성 평가 지표**를 사용하여 생성된 서열을 평가.  
  - **멀티 리스트 방식의 선호 최적화(Multi-listwise Preference Optimization)**를 적용하여, **고품질 단백질 서열을 생성**하도록 학습.  
  - 다중 속성 단백질 생성을 위해 **여러 프리픽스를 결합(Prefix Concatenation) 후 최적화 수행**.  

---

**2. 실험 결과 테이블 (Experimental Results Tables)**  

- **Table 1: 단일 속성 단백질 생성 결과 (Single-Attribute Generation Results)**  
  - **메트릭:** CLS-score, TM-score, RMSD, pLDDT  
  - **CtrlProt이 기존 모델(EvoDiff, ProGen2, ProLLaMA 등)보다 높은 TM-score, CLS-score, 낮은 RMSD, 높은 pLDDT를 달성**.  
  - 특히 **pLDDT(구조 신뢰도)에서 큰 향상을 보이며, 구조적으로 안정적인 단백질을 생성함을 입증**.  

- **Table 2: 방법론 비교 및 제거 실험 (Ablation and Alternative Studies)**  
  - **구조적 안정성 평가 지표(γ) 또는 기능성 평가 지표(τ)를 제거하면 성능이 하락**, 두 지표가 모두 모델 성능에 중요함을 확인.  
  - 기존 DPO, ORPO, Lipo-λ보다 **멀티 리스트 방식의 선호 최적화가 더 높은 성능을 보임**.  

- **Table 3: 생성된 단백질 서열의 다양성 평가 (Diversity Analysis Results)**  
  - **CtrlProt은 훈련 데이터에 과적합(overfitting)되지 않고 새로운 서열을 생성**.  
  - n-gram 기반 유사도 분석 결과, **자연 단백질과 비슷한 수준의 다양성을 유지하며, 특정 패턴에 고정되지 않음을 증명**.  

---

**3. 생성된 단백질 서열의 구조 비교 (Case Study - Protein Structure Comparison)**  

- **Figure 3: 단일 속성 단백질 생성 사례 (Single-Attribute Protein Generation Case Study)**  
  - CtrlProt이 생성한 단백질 서열(파란색)과 자연 단백질(노란색)의 **3D 구조 비교**.  
  - 높은 **TM-score(구조적 유사성)**을 가지며, 생성된 단백질이 기존 자연 단백질과 기능적으로 유사함을 확인.  

- **Figure 5: 다중 속성 단백질 생성 사례 (Multi-Attribute Protein Generation Case Study)**  
  - 두 가지 속성을 가진 단백질을 생성하고, 실제 자연 단백질과의 구조 비교 수행.  
  - **Prefix Concatenation을 통한 다중 속성 최적화가 효과적임을 보여줌**.  

---



**1. Method Overview Diagram (CtrlProt Architecture)**  
- **Figure 2: CtrlProt Framework Overview**  
  - **Fine-tuning of the protein LLM using prefix-tuning** followed by evaluation based on **functionality and structural stability metrics**.  
  - **Multi-listwise preference optimization** is applied to ensure **high-quality protein sequence generation**.  
  - **For multi-attribute generation**, different prefixes are **concatenated and optimized** to enhance output quality.  

---

**2. Experimental Results Tables**  

- **Table 1: Single-Attribute Protein Generation Results**  
  - **Metrics:** CLS-score, TM-score, RMSD, pLDDT  
  - **CtrlProt outperforms existing models (EvoDiff, ProGen2, ProLLaMA) with higher TM-score, CLS-score, lower RMSD, and higher pLDDT**.  
  - **Significant improvements in pLDDT indicate that CtrlProt generates structurally stable proteins**.  

- **Table 2: Ablation and Alternative Studies**  
  - **Removing either structural stability (γ) or functionality (τ) metrics leads to performance degradation**, proving their necessity.  
  - **Multi-listwise preference optimization outperforms standard methods like DPO, ORPO, and Lipo-λ**.  

- **Table 3: Diversity Analysis Results**  
  - **CtrlProt does not overfit to training data and generates novel protein sequences**.  
  - n-gram similarity analysis shows that **it maintains diversity similar to natural proteins while avoiding mode collapse**.  

---

**3. Protein Structure Comparison (Case Studies)**  

- **Figure 3: Single-Attribute Protein Generation Case Study**  
  - **3D structure comparison between CtrlProt-generated proteins (blue) and natural proteins (yellow)**.  
  - **High TM-score (structural similarity)** confirms that generated proteins maintain similar functionality.  

- **Figure 5: Multi-Attribute Protein Generation Case Study**  
  - Generates proteins with **two attributes** and compares them to real proteins.  
  - **Prefix concatenation and optimization prove effective for multi-attribute protein generation**.  


<br/>
# refer format:     


@article{Liu2025CtrlProt,
  author    = {Xiangyu Liu and Yi Liu and Silei Chen and Wei Hu},
  title     = {Controllable Protein Sequence Generation with LLM Preference Optimization},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  url       = {https://github.com/nju-websoft/CtrlProt}
}




Liu, Xiangyu, Yi Liu, Silei Chen, and Wei Hu. "Controllable Protein Sequence Generation with LLM Preference Optimization." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Association for the Advancement of Artificial Intelligence (AAAI). Available at https://github.com/nju-websoft/CtrlProt.





