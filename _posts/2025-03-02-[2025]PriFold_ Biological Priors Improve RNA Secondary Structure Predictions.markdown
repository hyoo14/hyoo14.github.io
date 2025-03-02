---
layout: post
title:  "[2025]PriFold: Biological Priors Improve RNA Secondary Structure Predictions"  
date:   2025-03-02 13:43:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


특정 속성 바탕으로 어규멘테이션(RNA COVARIANCE 바탕?)+페이링 확률을 어텐션에 포함시킴  


짧은 요약(Abstract) :    




RNA의 2차 구조 예측은 RNA의 기능을 이해하고, RNA 기반 치료제를 설계하며, 세포 내 분자 상호작용을 연구하는 데 필수적이다. 기존 딥러닝 기반 방법들은 주로 지역적 구조 특성에 집중하여 RNA 서열의 전체적인 특성과 진화적 정보를 충분히 반영하지 못하는 한계가 있었다. 이를 해결하기 위해 우리는 PriFold라는 새로운 모델을 제안한다. 이 모델은 1) 페어링 확률을 활용한 주의(attention) 메커니즘 개선을 통해 전역적인 결합 특성을 반영하고, 2) RNA 공변이(covariation)를 기반으로 데이터 증강을 수행하여 진화적 정보를 활용하는 것이 특징이다. 구조적 특성을 반영한 사전 학습과 미세 조정을 통해 모델 성능을 최적화하였다. 다양한 실험을 통해 PriFold가 bpRNA, RNAStrAlign 및 ArchiveII 등 여러 벤치마크 데이터셋에서 최신(SOTA) 성능을 달성함을 확인하였다. 이 연구는 생물학적 사전 지식을 RNA 구조 예측에 통합하는 것이 높은 예측 성능을 달성하는 데 유용하며, RNA 생물학 및 생물정보학 분야에서 새로운 연구 방향을 제시할 가능성을 보여준다.

---


Predicting RNA secondary structures is crucial for understanding RNA function, designing RNA-based therapeutics, and studying molecular interactions within cells. Existing deep-learning-based methods for RNA secondary structure prediction have mainly focused on local structural properties, often overlooking the global characteristics and evolutionary features of RNA sequences. Guided by biological priors, we propose PriFold, incorporating two key innovations: 1) improving the attention mechanism with pairing probabilities to utilize global pairing characteristics, and 2) implementing data augmentation based on RNA covariation to leverage evolutionary information. Our structured enhanced pretraining and finetuning strategy significantly optimizes model performance. Extensive experiments demonstrate that PriFold achieves state-of-the-art (SOTA) results in RNA secondary structure prediction on benchmark datasets such as bpRNA, RNAStrAlign, and ArchiveII. These results not only validate our prediction approach but also highlight the potential of integrating biological priors, such as global characteristics and evolutionary information, into RNA structure prediction tasks, opening new avenues for research in RNA biology and bioinformatics.



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




#### **1. PriFold 모델 개요**  
PriFold는 RNA 2차 구조 예측을 개선하기 위해 생물학적 사전 지식(biological priors)을 활용하는 새로운 모델이다. 이 모델의 주요 특징은 다음과 같다.  
1) **페어링 확률 기반 주의(attention) 메커니즘 개선**: RNA 염기쌍 형성 확률을 활용하여 글로벌 결합 특성을 반영한다.  
2) **RNA 공변이(covariation) 기반 데이터 증강**: RNA 진화적 특성을 활용하여 모델 학습을 강화한다.  

#### **2. 모델 아키텍처 (Architecture)**
PriFold의 학습 과정은 **사전 학습 (pretraining)** 및 **미세 조정 (finetuning)**의 두 단계로 나뉜다.

- **사전 학습 (Pretraining)**  
  - MARS 데이터셋(1.1억 개의 RNA 서열)을 이용하여 마스킹 언어 모델(Masked Language Modeling, MLM) 방식으로 학습 진행.  
  - **Llama2 스타일 인코더**를 적용하여 총 12개의 Transformer 블록(각 블록은 RMS 정규화, 다중 헤드 주의(Multi-Head Attention), 피드포워드 신경망 포함)로 구성됨.  

- **미세 조정 (Finetuning)**  
  - RNA 2차 구조가 주어진 데이터셋을 활용하여 추가 학습 수행.  
  - **RNA 공변이 기반 데이터 증강 기법 적용**  
    - RNA 변이를 시뮬레이션하여 구조적 특성을 보존한 상태에서 학습 데이터 확장.  
  - **Axial Attention 메커니즘 적용**  
    - 행(row) 및 열(column) 방향으로 분리된 주의(attention)를 적용하여 계산량 감소.  
  - **위치 가중 행렬(Positional Factor Matrix) 추가**  
    - RNA 염기쌍 빈도를 반영한 가중치를 적용하여 예측 정확도를 향상.  

#### **3. 트레이닝 데이터 (Training Data)**
PriFold의 학습 및 평가를 위해 다음과 같은 벤치마크 데이터셋을 사용하였다.  

1) **RNAStrAlign** (37,149개 RNA 구조)  
   - 주 학습 데이터셋으로 활용됨.  
2) **ArchiveII** (3,975개 RNA 구조)  
   - 일반화 성능 평가를 위해 사용됨.  
3) **bpRNA** (13,419개 RNA 서열)  
   - 대규모 벤치마크 실험을 위해 사용됨.  

실험 결과, PriFold는 기존 RNA 2차 구조 예측 모델들보다 높은 정확도를 기록하며 최첨단(SOTA) 성능을 달성하였다.  

---



#### **1. PriFold Model Overview**  
PriFold is a novel model that enhances RNA secondary structure prediction by integrating biological priors. Its key innovations include:  
1) **Attention mechanism enhanced with pairing probabilities**, incorporating global pairing characteristics.  
2) **RNA covariation-based data augmentation**, leveraging evolutionary information for better training.  

#### **2. Model Architecture**  
The training process of PriFold consists of two stages: **Pretraining** and **Finetuning**.

- **Pretraining Stage**  
  - Trained using the **MARS dataset** (1.1 billion RNA sequences) with **Masked Language Modeling (MLM)** objective.  
  - Utilizes a **Llama2-style encoder** with 12 Transformer blocks, each containing RMS normalization, multi-head attention (MHA), and feedforward neural networks.  

- **Finetuning Stage**  
  - Uses datasets with annotated RNA secondary structures for further training.  
  - **RNA Covariation-Based Data Augmentation**  
    - Simulates RNA mutations while preserving structural integrity.  
  - **Axial Attention Mechanism**  
    - Applies separate attention along rows and columns to reduce computational complexity.  
  - **Positional Factor Matrix**  
    - Incorporates base-pairing frequency information to improve prediction accuracy.  

#### **3. Training Data**  
PriFold is trained and evaluated on the following benchmark datasets:  

1) **RNAStrAlign** (37,149 RNA structures)  
   - Used as the primary training dataset.  
2) **ArchiveII** (3,975 RNA structures)  
   - Used to evaluate generalization performance.  
3) **bpRNA** (13,419 RNA sequences)  
   - Used for large-scale benchmarking.  

The experimental results demonstrate that PriFold outperforms existing RNA secondary structure prediction methods, achieving state-of-the-art (SOTA) performance.

   
 
<br/>
# Results  




#### **1. 경쟁 모델 (Comparison Models)**
PriFold의 성능을 기존 RNA 2차 구조 예측 모델들과 비교하기 위해 다양한 경쟁 모델을 선정하였다. 비교 대상은 다음과 같다.  

- **전통적 방법 (Traditional Methods)**  
  - **MFold**, **RNAstructure**, **RNAfold**, **LinearFold** 등  
  - 에너지 최소화 기반의 동적 프로그래밍(DP) 기법을 사용하여 RNA 2차 구조를 예측하는 방식.  

- **딥러닝 기반 방법 (Deep Learning-Based Methods)**  
  - **SPOT-RNA**, **E2Efold**, **UFold**, **RFold**  
  - ResNet, LSTM, Transformer 등의 신경망을 활용한 최신 딥러닝 기법 적용.  

- **사전학습 모델 (Pretrained Language Models)**  
  - **RNAErnie**, **RiNALMo**, **RNA-FM**, **ERNIE-RNA**  
  - 대규모 RNA 서열 데이터로 사전 학습된 언어 모델을 활용하여 RNA 구조를 예측하는 방식.  

#### **2. 테스트 데이터 (Test Datasets)**
PriFold의 성능을 평가하기 위해 다음 세 개의 대표적인 RNA 2차 구조 벤치마크 데이터셋을 사용하였다.

1) **RNAStrAlign** (37,149개 RNA 구조)  
   - 표준 RNA 2차 구조 예측 성능을 평가하기 위한 핵심 데이터셋.  

2) **ArchiveII** (3,975개 RNA 구조)  
   - 일반화 성능 검증을 위해 사용.  

3) **bpRNA** (13,419개 RNA 서열)  
   - 대규모 데이터에서의 성능을 평가하는 데 활용됨.  

#### **3. 평가 메트릭 (Evaluation Metrics)**
RNA 2차 구조 예측 성능을 평가하기 위해 **Precision(정밀도)**, **Recall(재현율)**, **F1-score**를 사용하였다.  

- **Precision (정밀도)**: 예측된 염기쌍 중 실제로 올바른 염기쌍의 비율  
- **Recall (재현율)**: 실제 RNA 2차 구조에서 예측된 염기쌍의 비율  
- **F1-score**: Precision과 Recall의 조화 평균, 전체적인 예측 성능을 평가하는 핵심 지표  

#### **4. 결과 (Results)**
- **RNAStrAlign 데이터셋에서 PriFold는 기존 SOTA 모델을 능가하는 성능을 기록하였다.**  
  - **F1-score: 0.988**, 기존 최고 성능이었던 RFold(0.977)를 초과함.  

- **Pseudoknot(거짓 매듭) 포함 RNA 서열 예측에서도 우수한 성능을 보임.**  
  - **F1-score: 0.944**, RFold(0.918)보다 높은 점수 기록.  

- **일반화 성능 평가 (ArchiveII 데이터셋)에서도 기존 모델 대비 높은 성능을 기록함.**  
  - **F1-score: 0.952**, 기존 최고 성능이었던 RFold(0.921)를 초과.  

- **대규모 벤치마크 데이터셋(bpRNA)에서도 최고 성능 달성.**  
  - **F1-score: 0.770**, 기존 모델인 RNA-FM(0.704) 및 RiNALMo(0.747)보다 높은 성능을 기록.  

#### **5. 결과 요약**
PriFold는 전통적인 RNA 구조 예측 기법뿐만 아니라 최신 딥러닝 및 사전학습된 언어 모델 기반의 접근 방식보다 높은 성능을 보였다. 특히, RNA 공변이 기반 데이터 증강 및 위치 가중 행렬을 적용한 점이 모델 성능 향상에 기여한 것으로 분석된다.  

---



#### **1. Comparison Models**
PriFold’s performance was compared against various existing RNA secondary structure prediction models, categorized as follows:

- **Traditional Methods**  
  - **MFold**, **RNAstructure**, **RNAfold**, **LinearFold**  
  - These methods use dynamic programming (DP) techniques to predict RNA secondary structures based on energy minimization.  

- **Deep Learning-Based Methods**  
  - **SPOT-RNA**, **E2Efold**, **UFold**, **RFold**  
  - These models incorporate neural networks such as ResNet, LSTM, and Transformer-based architectures.  

- **Pretrained Language Models**  
  - **RNAErnie**, **RiNALMo**, **RNA-FM**, **ERNIE-RNA**  
  - These models leverage large-scale RNA sequence datasets for pretraining to enhance RNA structure prediction.  

#### **2. Test Datasets**
PriFold was evaluated on three major RNA secondary structure benchmark datasets:

1) **RNAStrAlign** (37,149 RNA structures)  
   - Used as the primary dataset for evaluating standard RNA secondary structure prediction performance.  

2) **ArchiveII** (3,975 RNA structures)  
   - Used to assess generalization performance.  

3) **bpRNA** (13,419 RNA sequences)  
   - Used for large-scale benchmarking.  

#### **3. Evaluation Metrics**
The following metrics were used to assess RNA secondary structure prediction performance:

- **Precision**: The proportion of correctly predicted base pairs among all predicted base pairs.  
- **Recall**: The proportion of correctly predicted base pairs among all true base pairs in the actual RNA structure.  
- **F1-score**: The harmonic mean of precision and recall, serving as a key performance indicator.  

#### **4. Results**
- **PriFold outperformed existing SOTA models on the RNAStrAlign dataset.**  
  - **F1-score: 0.988**, surpassing the previous best RFold (0.977).  

- **Superior performance in predicting RNA sequences with pseudoknots.**  
  - **F1-score: 0.944**, higher than RFold (0.918).  

- **Achieved strong generalization performance on the ArchiveII dataset.**  
  - **F1-score: 0.952**, surpassing the previous best RFold (0.921).  

- **Achieved top performance in large-scale benchmarking (bpRNA dataset).**  
  - **F1-score: 0.770**, exceeding RNA-FM (0.704) and RiNALMo (0.747).  

#### **5. Summary of Results**
PriFold demonstrated superior performance compared to both traditional RNA structure prediction methods and state-of-the-art deep learning and pretrained language models. The integration of **RNA covariation-based data augmentation** and the **positional factor matrix** significantly contributed to the model’s performance improvements.



<br/>
# 예제  


#### **1. 트레이닝 데이터 예제 (Training Data Example)**
PriFold는 **MARS 데이터셋**을 이용하여 사전 학습되며, **RNAStrAlign** 및 **bpRNA** 데이터셋을 활용해 미세 조정을 진행하였다. 각 데이터셋은 RNA 서열과 해당 2차 구조 정보를 포함한다.

**예제 데이터 (RNAStrAlign 데이터셋)**
```plaintext
>Sequence_1
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
((((((....))))))......(((....)))
```
- **첫 번째 줄**: RNA 서열의 ID  
- **두 번째 줄**: RNA 서열 (A, U, G, C)  
- **세 번째 줄**: RNA 2차 구조 (괄호 표기법)  

---

#### **2. 실험 데이터 예제 (Experimental Data Example)**
실험 데이터는 사전 학습된 PriFold 모델이 예측하는 RNA 2차 구조와 실제 정답(ground truth)을 비교하여 평가된다.

**실제 RNA 2차 구조 예제 (Ground Truth)**
```plaintext
>Test_Sequence_42
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
((((((....))))))......(((....)))
```

**PriFold 예측 결과**
```plaintext
>Predicted_Sequence_42
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
((((((....))))))......((......))
```
- **Ground Truth와 비교하여 차이를 분석하여 F1-score 계산**  

---

#### **3. 테스트 입력/출력 예제 (Test Input/Output Example)**  
PriFold 모델은 RNA 서열을 입력으로 받아 2차 구조를 예측한다.

**테스트 입력 (Test Input)**
```plaintext
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
```

**모델 출력 (Predicted Output)**
```plaintext
((((((....))))))......(((....)))
```

- **Precision, Recall, F1-score로 평가**  

---



#### **1. Training Data Example**
PriFold is pretrained on the **MARS dataset** and fine-tuned using **RNAStrAlign** and **bpRNA** datasets. Each dataset contains RNA sequences along with their secondary structure annotations.

**Example Data (RNAStrAlign dataset)**
```plaintext
>Sequence_1
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
((((((....))))))......(((....)))
```
- **First line**: RNA sequence ID  
- **Second line**: RNA sequence (A, U, G, C)  
- **Third line**: RNA secondary structure (bracket notation)  

---

#### **2. Experimental Data Example**
The experimental data is used to evaluate the PriFold model’s predictions against ground truth RNA secondary structures.

**Ground Truth Example**
```plaintext
>Test_Sequence_42
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
((((((....))))))......(((....)))
```

**PriFold Predicted Output**
```plaintext
>Predicted_Sequence_42
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
((((((....))))))......((......))
```
- **The difference between the ground truth and predicted structure is analyzed to compute the F1-score.**  

---

#### **3. Test Input/Output Example**  
PriFold takes an RNA sequence as input and predicts its secondary structure.

**Test Input**
```plaintext
AUGCGAUCGUAGCUAGCUGAUCGUAGCUAGC
```

**Predicted Output**
```plaintext
((((((....))))))......(((....)))
```

- **Evaluation is conducted using Precision, Recall, and F1-score.**




<br/>  
# 요약   




PriFold는 RNA 2차 구조 예측을 향상시키기 위해 **페어링 확률 기반 주의 메커니즘**과 **RNA 공변이 기반 데이터 증강**을 적용한 딥러닝 모델이다. 실험 결과, PriFold는 RNAStrAlign, ArchiveII, bpRNA 데이터셋에서 기존 최신 모델보다 높은 F1-score(최대 0.988)를 기록하며 SOTA 성능을 달성했다. 입력으로 RNA 서열을 받아 예측된 2차 구조를 출력하며, 예측 성능은 정밀도(Precision), 재현율(Recall), F1-score로 평가된다.  

---


PriFold is a deep learning model that improves RNA secondary structure prediction by incorporating **pairing probability-based attention mechanisms** and **RNA covariation-based data augmentation**. Experimental results show that PriFold outperforms previous state-of-the-art models, achieving an F1-score of up to 0.988 on RNAStrAlign, ArchiveII, and bpRNA datasets. The model takes RNA sequences as input and outputs predicted secondary structures, with performance evaluated using Precision, Recall, and F1-score.


<br/>  
# 기타  



#### **1. 모델 아키텍처 다이어그램 (Figure: PriFold Model Architecture)**  
- **구성 요소**: PriFold의 전체 학습 과정은 **사전 학습 (Pretraining)**과 **미세 조정 (Finetuning)** 두 단계로 구성됨.  
- **사전 학습 (Pretraining)**: 대규모 RNA 서열 데이터(MARS)를 이용하여 Llama2 스타일 인코더를 학습.  
- **미세 조정 (Finetuning)**: RNA 공변이 기반 데이터 증강을 적용하고, Axial Attention을 포함한 모델로 구조 예측 수행.  
- **특징**: **페어링 확률(Position Factor Matrix) 적용**, **Covariation-Based Data Augmentation 사용**, **Axial Attention 메커니즘 도입**하여 전통적인 딥러닝 모델보다 더 높은 성능을 달성.  

#### **2. 성능 비교 테이블 (Tables: Performance Comparison)**  
- **RNAStrAlign 데이터셋에서 F1-score 비교**  
  - PriFold: **0.988** (기존 최고 모델 RFold: 0.977보다 우수)  
- **Pseudoknot 포함 RNA 구조 예측 성능 비교**  
  - PriFold: **0.944**, RFold(0.918)보다 향상됨.  
- **ArchiveII 데이터셋에서 일반화 성능 비교**  
  - PriFold: **0.952**, 기존 최고 성능 모델 RFold(0.921)를 초과.  
- **bpRNA 대규모 데이터셋에서의 성능 비교**  
  - PriFold: **0.770**, RNA-FM(0.704) 및 RiNALMo(0.747)보다 높은 성능.  

#### **3. 데이터 증강 방식 다이어그램 (Figure: Covariation-Based Data Augmentation)**  
- **설명**: RNA 공변이를 반영하여 데이터 증강을 수행하는 방식 시각화.  
- **예제**: 원본 RNA 구조에서 G-C 페어링을 A-U 또는 G-U 페어링으로 변환하여 모델이 다양한 변이를 학습하도록 유도.  
- **효과**: RNA 구조의 보존을 유지하면서 훈련 데이터 다양성을 증가시켜 성능 향상.  

#### **4. Axial Attention 설명 그림 (Figure: Axial Attention Mechanism)**  
- **설명**: 기존 2D 어텐션보다 **계산량이 줄어들면서도 RNA 구조 정보를 효과적으로 반영**하는 Axial Attention 적용 방식.  
- **특징**: 행(row) 및 열(column) 방향으로 주의(attention)를 분할하여 계산 효율성을 높이고, 위치 가중 행렬(Position Factor)을 반영하여 생물학적 의미를 보강함.  

---



#### **1. Model Architecture Diagram (Figure: PriFold Model Architecture)**  
- **Components**: The PriFold model training process consists of **Pretraining** and **Finetuning** phases.  
- **Pretraining**: Trained on large-scale RNA sequence data (MARS) using a **Llama2-style encoder**.  
- **Finetuning**: Uses **RNA covariation-based data augmentation** and **Axial Attention** for structure prediction.  
- **Key Features**:  
  - **Pairing probability-based attention (Positional Factor Matrix)**  
  - **Covariation-Based Data Augmentation**  
  - **Axial Attention Mechanism** for better performance over traditional deep learning models.  

#### **2. Performance Comparison Tables (Tables: Performance Comparison)**  
- **F1-score comparison on RNAStrAlign dataset**  
  - PriFold: **0.988**, outperforming the previous best RFold (0.977).  
- **RNA structure prediction with pseudoknots**  
  - PriFold: **0.944**, better than RFold (0.918).  
- **Generalization performance on ArchiveII dataset**  
  - PriFold: **0.952**, surpassing the previous best RFold (0.921).  
- **Large-scale performance on bpRNA dataset**  
  - PriFold: **0.770**, higher than RNA-FM (0.704) and RiNALMo (0.747).  

#### **3. Covariation-Based Data Augmentation Diagram (Figure: Covariation-Based Data Augmentation)**  
- **Description**: A visualization of how RNA covariation is used for data augmentation.  
- **Example**: Original RNA structure with a **G-C base pair** is transformed into an **A-U or G-U base pair**, allowing the model to learn diverse variations.  
- **Effect**: Preserves RNA structure while increasing training data diversity, leading to improved performance.  

#### **4. Axial Attention Mechanism Illustration (Figure: Axial Attention Mechanism)**  
- **Description**: Shows how Axial Attention **reduces computational cost while effectively capturing RNA structure information**.  
- **Features**:  
  - Decomposes **global 2D attention** into **separate row and column attention** for computational efficiency.  
  - Integrates a **positional factor matrix** to enhance biological relevance.


<br/>
# refer format:     



@article{yang2025prifold,
  author    = {Chenchen Yang and Hao Wu and Tao Shen and Kai Zou and Siqi Sun},
  title     = {PriFold: Biological Priors Improve RNA Secondary Structure Predictions},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)},
  url       = {https://github.com/BEAM-Labs/PriFold}
}




Yang, Chenchen, Hao Wu, Tao Shen, Kai Zou, and Siqi Sun. PriFold: Biological Priors Improve RNA Secondary Structure Predictions. Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Association for the Advancement of Artificial Intelligence (AAAI). https://github.com/BEAM-Labs/PriFold.   





