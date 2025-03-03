---
layout: post
title:  "[2025]FoldToken: Learning Protein Language via Vector Quantization and Beyond"  
date:   2025-03-03 17:59:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


단백질 시퀀스 + 단백질 구조(여기서는 좌표 기반으로 표현)  
이를 위한 각각 FoldTokenizer(좌표랑 단백질 다 숫자 매핑 줌(좌표당 하나, 아미노산당 하나), 그리고 컨캣함, 근데 이때 vector quantization 이라는 정수를 주는 기법 사용->오 이거 좀 신기)       
그리고 다시 gpt에 입력해서 좌표랑 단백질 시퀀스 복원하도록 훈련   
이 모델로 생성 테스트 수행.. 성능 좋았다! 이말.. 굿   


짧은 요약(Abstract) :    




이 연구는 단백질 서열과 구조를 동시에 표현하는 새로운 외국어 개념을 제시한다. 단백질 구조는 연속적인 3D 점들로 표현되며, 기존의 이산적인 서열 모델링 방식과 차이가 있어 다루기 어려웠다. 이에 따라, 연구진은 **FoldTokenizer**를 개발하여 단백질 서열-구조를 이산 기호로 변환하는 방법을 제안했다. 이 과정에서는 잔기(residue) 유형과 구조를 이산 공간으로 투영하며, 정보 손실을 최소화하는 재구성 손실을 통해 학습한다. 학습된 이산 기호를 **FoldToken**이라 명명하고, 이를 통해 단백질 서열-구조를 단일한 양식(modality)으로 변환하여 새로운 단백질 언어를 형성한다.

이 연구에서는 **FoldGPT**라는 GPT 스타일 모델을 구축하여 서열-구조를 함께 생성하는 첫 번째 시스템을 제시했다. 핵심적인 성과는 벡터 양자화(vector quantization) 모듈을 크게 개선한 **Soft Conditional Vector Quantization (SoftCVQ)** 기법을 개발한 것이다. 이를 통해 단백질의 백본 복원(backbone inpainting)에서 우수한 성능을 보였으며, 기존 연속적인 각도 기반 방식보다 더 나은 성과를 거두었다.

---


**Is there a foreign language describing protein sequences and structures simultaneously?** Protein structures, represented by continuous 3D points, have long posed a challenge due to the contrasting modeling paradigms of discrete sequences. We introduce **FoldTokenizer** to represent protein sequence-structure as discrete symbols. This approach involves projecting residue types and structures into a discrete space, guided by a reconstruction loss for information preservation. We name the learned discrete symbols as **FoldToken**, and the sequence of FoldTokens serves as a new protein language, transforming the protein sequence-structure into a unified modality. 

We apply the created protein language on the general backbone inpainting task, building the first **GPT-style model (FoldGPT)** for sequence-structure co-generation with promising results. Key to our success is the substantial enhancement of the vector quantization module, **Soft Conditional Vector Quantization (SoftCVQ)**.



 

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






이 연구에서는 단백질 서열과 구조를 통합하여 표현하는 새로운 기법을 제안하며, 이를 위해 **FoldTokenizer**와 **FoldGPT**라는 두 가지 핵심 모델을 도입하였다.  

#### **1. FoldTokenizer**  
**FoldTokenizer**는 단백질 서열과 구조를 이산적 기호로 변환하는 모델로, 크게 세 가지 구성 요소로 이루어져 있다.  
- **인코더(Encoder)**: 단백질 서열과 구조를 연속적(continuous) 임베딩 공간으로 변환하는 트랜스포머 기반 모델.  
- **양자화기(Quantizer)**: 연속적인 임베딩을 이산적인 코드로 변환하는 벡터 양자화(Vector Quantization, VQ) 기법을 적용. 여기서 제안된 **Soft Conditional Vector Quantization (SoftCVQ)** 기법은 기존 방식보다 더 높은 재구성(reconstruction) 성능을 제공한다.  
- **디코더(Decoder)**: 이산적인 코드에서 원래 단백질 서열과 구조를 복원하는 트랜스포머 모델.  

**FoldTokenizer**를 통해 단백질 서열과 구조를 새로운 단백질 언어(Protein Language)로 변환하고, 이를 기반으로 다양한 생성 및 예측 작업을 수행할 수 있도록 한다.  

#### **2. FoldGPT**  
**FoldGPT**는 GPT 스타일의 언어 모델을 기반으로 단백질 서열과 구조를 동시에 생성할 수 있도록 설계되었다.  
- **백본 인페인팅(Backbone Inpainting)**: 기존 서열 및 구조 데이터를 기반으로 결손된 단백질 부분을 예측하는 작업.  
- **시퀀스-구조 동시 생성(Co-generation)**: 단백질 서열과 3D 구조를 함께 예측하여 새로운 단백질을 생성.  
- **트랜스포머 기반 설계**: 기존의 그래프 신경망(GNN)이나 SE(3) 변환을 사용하지 않고, 단순한 트랜스포머 모델을 통해 서열과 구조를 통합적으로 모델링.  

#### **3. 트레이닝 데이터**  
- **사전학습(Pretraining)**: AlphaFold Uniprot v3 데이터베이스에서 추출된 1,323,729개의 단백질 구조 데이터를 이용하여 **FoldTokenizer**를 학습.  
- **백본 인페인팅 학습(Backbone Inpainting Training)**: CATH4.3 데이터셋을 이용하여 **FoldGPT** 모델을 학습. 훈련 데이터는 30,290개의 단백질을 포함하며, 검증(validation) 및 테스트(test) 데이터는 각각 638개, 957개 단백질로 구성됨.  
- **평가 및 실험**: 모델의 성능을 검증하기 위해, 시퀀스 복원 정확도(Recovery Rate), TMScore(단백질 구조 유사성 평가), 그리고 백본 인페인팅 성공률 등의 지표를 활용하여 성능을 평가.  

---



This study introduces a novel approach to jointly model protein sequences and structures using **FoldTokenizer** and **FoldGPT** as the two main components.  

#### **1. FoldTokenizer**  
**FoldTokenizer** converts protein sequences and structures into discrete symbols, comprising three key components:  
- **Encoder**: A transformer-based model that maps protein sequences and structures into a continuous latent space.  
- **Quantizer**: A vector quantization (VQ) module that converts the continuous embeddings into discrete tokens. The proposed **Soft Conditional Vector Quantization (SoftCVQ)** significantly improves reconstruction quality compared to previous methods.  
- **Decoder**: A transformer model that reconstructs the original protein sequence and structure from the discrete tokens.  

By learning this discrete **protein language**, FoldTokenizer enables sequence-structure co-modeling for generative and predictive tasks.  

#### **2. FoldGPT**  
**FoldGPT** is a GPT-style language model designed to co-generate protein sequences and structures.  
- **Backbone Inpainting**: Predicting missing protein fragments using existing sequence and structure context.  
- **Sequence-Structure Co-generation**: Simultaneous generation of protein sequences and their corresponding 3D structures.  
- **Transformer-Based Design**: Unlike traditional methods relying on graph neural networks (GNNs) or SE(3) models, FoldGPT uses a pure transformer architecture to model both sequences and structures.  

#### **3. Training Data**  
- **Pretraining**: **FoldTokenizer** is pretrained using **1,323,729 protein structures** from the **AlphaFold Uniprot v3** database.  
- **Backbone Inpainting Training**: **FoldGPT** is trained on the **CATH4.3 dataset**, consisting of **30,290 training proteins, 638 validation proteins, and 957 test proteins**.  
- **Evaluation & Experiments**: Model performance is assessed using recovery rate (sequence accuracy), TMScore (structural similarity), and backbone inpainting success rate.  

These advancements enable **FoldGPT** to outperform existing methods in protein structure recovery and co-generation tasks, establishing a new paradigm for protein modeling using discrete representations.


   




   
 
<br/>
# Results  




이 연구에서는 **FoldTokenizer**와 **FoldGPT**가 기존 모델보다 뛰어난 성능을 보였음을 다양한 실험을 통해 입증하였다. 연구진은 경쟁 모델과의 비교, 테스트 데이터, 그리고 평가 메트릭을 활용하여 성능을 검증하였다.  

#### **1. 경쟁 모델 (Baseline Models)**  
본 연구에서 제안된 **FoldGPT**는 단백질 백본 인페인팅(Backbone Inpainting)과 서열-구조 동시 생성(Co-generation)에서 기존의 다양한 방법들과 성능을 비교하였다.  
- **단백질 시퀀스 인페인팅**  
  - **기존 모델**: ESM2, EvoDiff  
  - **비교 기준**: 단백질 서열 복원율 (Recovery Rate), 구조 유사성 (TMScore)  
  - **결과**: FoldGPT가 기존의 **ESM2, EvoDiff**보다 높은 복원율과 더 높은 TMScore를 기록.  
- **단백질 구조 인페인팅**  
  - **각도 기반(Agnle-based) 모델**: FoldingDiff, DiffSDS  
  - **좌표 기반(Coordinate-based) 모델**: ProtDiff, RFDiffusion, Chroma  
  - **비교 기준**: 단백질 구조 복원율, TMScore, 구조 예측 정확도  
  - **결과**: FoldGPT는 각도 기반 방법(FoldingDiff, DiffSDS)보다 우수한 성능을 보였으며, 좌표 기반 모델(ProtDiff, RFDiffusion, Chroma)보다는 낮은 성능을 기록. 이는 좌표 기반 방식이 더 정밀한 구조 예측이 가능하다는 점을 반영하며, 향후 연구에서는 **FoldToken을 좌표 기반 방식으로 확장할 계획**임.  

#### **2. 테스트 데이터 (Test Dataset)**  
모델의 성능 검증을 위해 **CATH4.3** 데이터셋을 활용하여 실험을 진행하였다.  
- **재구성 실험**: 957개의 단백질 샘플 사용  
- **백본 인페인팅 실험**: ESMFold를 사용해 평균 pLDDT 0.9 이상의 고품질 단백질 117개를 선택하여 평가  
- **FoldTokenizer의 데이터 압축 성능 평가**: 원본 **CATH4.3** 데이터셋(2.7GB)을 **FoldToken을 활용하여 18MB로 압축 가능**  

#### **3. 평가 메트릭 (Evaluation Metrics)**  
다양한 성능 평가 지표를 활용하여 모델을 검증하였다.  
- **단백질 서열 재구성 (Sequence Recovery)**  
  - **Recovery Rate**: 원본 단백질 서열이 정확히 재구성된 비율  
- **단백질 구조 유사성 평가 (Structural Similarity)**  
  - **TMScore**: 예측된 단백질 구조가 실제 구조와 얼마나 유사한지 평가하는 지표  
  - **Lr, Lα, Lβ (Locational Reconstruction Loss)**: 특정 위치에서 원본 구조와의 차이 정도를 측정  
  - **Max Lr, Max Lα, Max Lβ**: 특정 단백질에서 최대 오차를 나타내는 지표  
- **백본 인페인팅 성공률 (Backbone Inpainting Success Rate)**  
  - **재구성 성공률**: 서열 복원율이 95% 이상이거나 TMScore가 0.5 이상일 경우 성공으로 간주  

#### **4. 주요 결과 요약**  
- **FoldTokenizer의 성능**  
  - 기존 벡터 양자화(VQ) 기법보다 **Soft Conditional Vector Quantization (SoftCVQ)**가 가장 높은 재구성 성능을 기록.  
  - TMScore와 재구성 성공률이 95% 이상으로, 신뢰할 수 있는 단백질 서열-구조 변환이 가능.  
- **FoldGPT의 성능**  
  - **ESM2, EvoDiff**보다 높은 서열 복원율과 구조 유사성을 보이며, GPT 스타일 모델이 단백질 서열 및 구조 예측에 효과적임을 입증.  
  - 기존의 **각도 기반(Angle-based) 방식보다 향상된 성능**을 보였으나, 좌표 기반(Coordinate-based) 모델에는 미치지 못함.  
  - 좌표 기반 방식으로 확장할 경우 더 높은 성능을 기대할 수 있음.  

이 연구는 단백질 서열과 구조를 단일한 언어로 변환하는 접근 방식이 실질적으로 높은 성능을 발휘함을 입증하였으며, 향후 좌표 기반 모델로 확장하는 방향을 제시하였다.  

---


This study demonstrates that **FoldTokenizer** and **FoldGPT** outperform existing models in protein sequence-structure tasks. The experiments were conducted using baseline models, test datasets, and multiple evaluation metrics.  

#### **1. Baseline Models (Competitors)**  
The proposed **FoldGPT** was compared against state-of-the-art methods in **backbone inpainting** and **sequence-structure co-generation** tasks.  
- **Protein Sequence Inpainting**  
  - **Baselines**: ESM2, EvoDiff  
  - **Metrics**: Sequence Recovery Rate, TMScore  
  - **Results**: FoldGPT achieved higher sequence recovery and TMScore than ESM2 and EvoDiff.  
- **Protein Structure Inpainting**  
  - **Angle-Based Models**: FoldingDiff, DiffSDS  
  - **Coordinate-Based Models**: ProtDiff, RFDiffusion, Chroma  
  - **Metrics**: Structure Recovery Rate, TMScore, Structural Prediction Accuracy  
  - **Results**: FoldGPT outperformed angle-based models (FoldingDiff, DiffSDS), but had slightly lower performance than coordinate-based models (ProtDiff, RFDiffusion, Chroma). This highlights the potential benefits of extending **FoldToken into a coordinate-based approach in future research**.  

#### **2. Test Dataset**  
The model was evaluated using the **CATH4.3** dataset:  
- **Reconstruction Task**: 957 protein samples  
- **Backbone Inpainting Task**: 117 high-quality proteins selected with an average **pLDDT ≥ 0.9** using ESMFold  
- **Compression Performance of FoldTokenizer**: **CATH4.3 dataset (2.7GB) was compressed to 18MB using FoldToken**  

#### **3. Evaluation Metrics**  
Multiple performance metrics were used to validate the model:  
- **Sequence Recovery Metrics**  
  - **Recovery Rate**: Percentage of correctly reconstructed protein sequences  
- **Structural Similarity Metrics**  
  - **TMScore**: Measures the similarity between predicted and actual protein structures  
  - **Lr, Lα, Lβ (Locational Reconstruction Loss)**: Quantifies structural differences at specific locations  
  - **Max Lr, Max Lα, Max Lβ**: Maximum observed error for specific proteins  
- **Backbone Inpainting Success Rate**  
  - **A reconstruction is deemed successful if the recovery rate is ≥95% or TMScore ≥ 0.5**  

#### **4. Key Findings**  
- **Performance of FoldTokenizer**  
  - The **Soft Conditional Vector Quantization (SoftCVQ)** method outperformed existing vector quantization techniques.  
  - Achieved **TMScore and reconstruction success rates above 95%**, indicating reliable sequence-structure transformation.  
- **Performance of FoldGPT**  
  - Outperformed **ESM2 and EvoDiff** in sequence recovery and structural similarity, confirming the effectiveness of a **GPT-style approach** for protein modeling.  
  - **Surpassed angle-based models (FoldingDiff, DiffSDS)** but fell short compared to **coordinate-based models (ProtDiff, RFDiffusion, Chroma)**.  
  - Extending **FoldToken to a coordinate-based representation could further improve performance**.  

This study establishes that transforming protein sequences and structures into a **unified language representation** significantly enhances modeling accuracy. Future work will explore **coordinate-based modeling** to further improve generative capabilities.



 
<br/>
# 예제  





이 연구에서는 **FoldTokenizer**와 **FoldGPT**를 평가하기 위해 대규모 단백질 데이터셋을 활용하였으며, 다양한 태스크에서 실제 단백질 서열 및 구조를 입력으로 사용하여 결과를 검증하였다.  

---

### **1. 테스트 및 트레이닝 데이터 (Test & Training Data)**  
- **사전학습 데이터 (Pretraining Data)**  
  - **데이터셋**: **AlphaFold Uniprot v3**  
  - **데이터 개수**: **1,323,729개 단백질**  
  - **사용 목적**: FoldTokenizer의 벡터 양자화(VQ) 학습  

- **백본 인페인팅 학습 데이터 (Backbone Inpainting Training Data)**  
  - **데이터셋**: **CATH4.3**  
  - **학습 데이터 (Train Set)**: **30,290개 단백질**  
  - **검증 데이터 (Validation Set)**: **638개 단백질**  
  - **테스트 데이터 (Test Set)**: **957개 단백질**  
  - **백본 인페인팅용 테스트 데이터**: **평균 pLDDT ≥ 0.9인 117개 단백질** (ESMFold를 활용하여 고품질 샘플 선별)  
  - **사용 목적**: FoldGPT의 백본 인페인팅 및 서열-구조 동시 생성 모델 평가  

---

### **2. 태스크 인풋/아웃풋 (Task Input/Output)**  
#### **(1) 단백질 재구성 (Protein Reconstruction)**
- **목적**: FoldTokenizer가 연속적인 단백질 서열과 구조를 이산적 토큰(FoldToken)으로 변환한 후, 이를 원래 서열 및 구조로 복원할 수 있는지 평가  
- **입력 (Input)**:  
  - 단백질 서열 (Amino Acid Sequence)  
  - 단백질 구조 (3D Coordinates of Cα Atoms)  
- **출력 (Output)**:  
  - 복원된 단백질 서열 (Reconstructed Amino Acid Sequence)  
  - 복원된 단백질 구조 (Reconstructed 3D Coordinates)  
- **성공 기준**:  
  - **Recovery Rate ≥ 95%**  
  - **TMScore ≥ 0.5**  

---

#### **(2) 백본 인페인팅 (Backbone Inpainting)**
- **목적**: 단백질의 일부 서열과 구조가 가려진 상태에서, FoldGPT가 이를 정확하게 복원할 수 있는지 평가  
- **입력 (Input)**:  
  - 기존 단백질의 일부 서열 및 구조 (Known Protein Sequence and Structure)  
  - 마스킹된 서열 및 구조 정보 (Masked Protein Fragments)  
- **출력 (Output)**:  
  - 예측된 단백질 서열 (Predicted Sequence)  
  - 예측된 단백질 구조 (Predicted Structure)  
- **성공 기준**:  
  - 예측된 구조를 **ESMFold**를 통해 접은 후, TMScore가 **기준 이상(≥0.5)**인지 평가  

---

#### **(3) 서열-구조 동시 생성 (Sequence-Structure Co-Generation)**
- **목적**: 새로운 단백질을 서열과 구조를 동시에 생성하여, 완전히 새로운 단백질 디자인 가능성을 탐색  
- **입력 (Input)**:  
  - 랜덤한 초기 서열 또는 조건이 주어진 서열  
- **출력 (Output)**:  
  - 생성된 단백질 서열 (Generated Protein Sequence)  
  - 해당 서열을 접은 후의 3D 구조 (Predicted Protein Structure)  
- **성공 기준**:  
  - 생성된 서열을 구조로 변환했을 때 안정적인 단백질 형성을 하는지 평가 (TMScore 및 pLDDT 기반)  

---



This study utilizes large-scale protein datasets to evaluate **FoldTokenizer** and **FoldGPT** across multiple tasks, using real protein sequences and structures as inputs to assess model performance.  

---

### **1. Test & Training Data**  
- **Pretraining Data**  
  - **Dataset**: **AlphaFold Uniprot v3**  
  - **Number of Proteins**: **1,323,729**  
  - **Purpose**: Train the vector quantization (VQ) model of FoldTokenizer  

- **Backbone Inpainting Training Data**  
  - **Dataset**: **CATH4.3**  
  - **Training Set**: **30,290 proteins**  
  - **Validation Set**: **638 proteins**  
  - **Test Set**: **957 proteins**  
  - **Backbone Inpainting Test Set**: **117 high-quality proteins with pLDDT ≥ 0.9** (selected using ESMFold)  
  - **Purpose**: Evaluate FoldGPT’s backbone inpainting and sequence-structure co-generation tasks  

---

### **2. Task Input/Output**  
#### **(1) Protein Reconstruction**  
- **Objective**: Evaluate FoldTokenizer’s ability to convert continuous protein sequences and structures into discrete tokens (FoldTokens) and reconstruct the original sequence-structure representation  
- **Input**:  
  - Protein Sequence (Amino Acid Sequence)  
  - Protein Structure (3D Coordinates of Cα Atoms)  
- **Output**:  
  - Reconstructed Protein Sequence  
  - Reconstructed Protein Structure (3D Coordinates)  
- **Success Criteria**:  
  - **Recovery Rate ≥ 95%**  
  - **TMScore ≥ 0.5**  

---

#### **(2) Backbone Inpainting**  
- **Objective**: Assess FoldGPT’s ability to predict masked fragments of protein sequences and structures  
- **Input**:  
  - Known Protein Sequence and Structure  
  - Masked Protein Fragments  
- **Output**:  
  - Predicted Protein Sequence  
  - Predicted Protein Structure  
- **Success Criteria**:  
  - The predicted structure is folded using **ESMFold**, and the TMScore is **≥0.5**  

---

#### **(3) Sequence-Structure Co-Generation**  
- **Objective**: Generate entirely new proteins by simultaneously predicting both sequences and structures  
- **Input**:  
  - A random initial sequence or a partially given sequence  
- **Output**:  
  - Generated Protein Sequence  
  - Folded 3D Protein Structure  
- **Success Criteria**:  
  - The generated sequence, when folded, forms a stable protein structure (evaluated based on TMScore and pLDDT)  

---

These tasks demonstrate **FoldTokenizer's** ability to **efficiently encode protein sequences and structures into discrete tokens** and **FoldGPT's** ability to **generate and reconstruct proteins** with high accuracy. The model’s performance is validated using large-scale datasets, ensuring its robustness in real-world applications.




<br/>  
# 요약   




이 연구에서는 **FoldTokenizer**를 통해 단백질 서열과 구조를 이산적 기호로 변환하고, **FoldGPT**를 활용하여 서열과 구조를 동시에 생성하는 새로운 단백질 언어 모델을 제안했다. 실험 결과, FoldGPT는 기존 시퀀스 및 구조 생성 모델보다 뛰어난 성능을 보였으며, 특히 백본 인페인팅 및 서열-구조 동시 생성에서 높은 정확도를 기록했다. CATH4.3 및 AlphaFold Uniprot v3 데이터를 활용하여 모델을 검증하였으며, 서열 복원율, TMScore, pLDDT 등의 평가 지표를 기반으로 높은 신뢰성을 입증했다.  

---


This study introduces a novel protein language model by transforming protein sequences and structures into discrete symbols using **FoldTokenizer** and generating both simultaneously with **FoldGPT**. Experimental results demonstrate that FoldGPT outperforms existing sequence and structure generation models, achieving high accuracy in backbone inpainting and sequence-structure co-generation. The model was validated using CATH4.3 and AlphaFold Uniprot v3 datasets, with performance assessed through sequence recovery rate, TMScore, and pLDDT, proving its reliability.


<br/>  
# 기타  





이 연구에서는 다양한 실험 결과를 시각적으로 나타내기 위해 **테이블(Table)과 피규어(Figure)**를 활용하였으며, 주요 결과를 정리하면 다음과 같다.  

---

### **1. 테이블(Table) 요약**  
#### **(1) 단백질 VQ 모델 성능 비교 (Table 2)**
- **목적**: 기존 벡터 양자화(VQ) 방법과 제안된 **Soft Conditional Vector Quantization (SoftCVQ)**의 단백질 재구성 성능 비교  
- **메트릭**:  
  - **Global Metrics**: TMScore (구조 유사성), Recovery Rate (서열 복원율)  
  - **Local Metrics**: Lr (위치별 구조 오차), Lα (각도 오차), Lβ (토션 각도 오차)  
  - **성공률(Success Rate)**: TMScore ≥ 0.5 또는 서열 복원율 ≥ 95%  
- **결과**:  
  - **SoftCVQ**가 기존 VQ 방법(Vanilla VQ, LFQ 등)보다 구조 재구성 성능이 뛰어나며, **TMScore 0.74, 복원 성공률 94.9%**를 기록.  
  - SoftVQ와 SoftGVQ도 기존 모델보다 향상된 성능을 보였으나, SoftCVQ가 가장 균형 잡힌 결과를 나타냄.  

#### **(2) 백본 인페인팅 성능 비교 (Table 3)**
- **목적**: FoldGPT가 기존 단백질 서열 및 구조 생성 모델(ESM2, EvoDiff, FoldingDiff 등)보다 높은 성능을 보이는지 평가  
- **메트릭**:  
  - **서열 복원율 (Rec)**: FoldGPT가 기존 서열을 얼마나 정확하게 복원하는지 평가  
  - **TMScore**: 구조 예측 정확도 비교  
- **결과**:  
  - **FoldGPT는 서열 복원율 96.2%, TMScore 0.80**으로 기존 모델보다 높은 성능을 기록.  
  - 특히, **각도 기반(Angle-based) 방식보다 향상된 성능을 보였으며**, 좌표 기반(Coordinate-based) 모델과 비교했을 때는 아직 개선 여지가 있음.  

#### **(3) SoftCVQ의 성능 분석 (Table 4)**
- **목적**: SoftCVQ의 성능을 다양한 하이퍼파라미터(MLP 레이어 수, 코드북 벡터 차원, 구형 정규화) 조합에서 평가  
- **결과**:  
  - MLP 레이어 수 증가 및 코드북 벡터 차원 증가가 성능 향상에 기여.  
  - 구형 정규화(Spherical Normalization) 적용 시 TMScore와 서열 복원율이 높아짐.  

---

### **2. 피규어(Figure) 요약**  
#### **(1) FoldToken 아키텍처 개요 (Figure 1)**
- **설명**:  
  - **Stage 1**: FoldTokenizer가 서열과 구조를 이산적 토큰으로 변환  
  - **Stage 2**: FoldGPT가 변환된 토큰을 활용해 단백질 구조 및 서열을 예측  
  - SoftCVQ를 이용해 VQ 과정에서 발생하는 문제(Gradient Mismatch, Semantic Irrelevance 등)를 해결  

#### **(2) 벡터 양자화 방법 비교 (Figure 2)**
- **설명**:  
  - Vanilla VQ와 Lookup-Free VQ(LFQ), SoftVQ, SoftGVQ, SoftCVQ를 비교  
  - SoftCVQ는 기존 방법보다 재구성 품질과 생성 성능을 모두 향상시킴  

#### **(3) SoftCVQ의 구조 (Figure 3)**
- **설명**:  
  - SoftVQ, SoftGVQ, SoftCVQ를 비교하며, SoftCVQ가 가장 효율적인 코드북 접근 방식을 사용하여 구조 재구성을 향상  

#### **(4) 재구성된 단백질 구조 시각화 (Figure 5, Appendix)**
- **설명**:  
  - SoftCVQ가 기존 VQ 방법보다 원본 단백질 구조를 더 정확하게 복원함  
  - TMScore와 복원율을 함께 시각화하여 성능 향상을 증명  

---



This study employs various **tables and figures** to visually present experimental results, highlighting the performance of **FoldTokenizer and FoldGPT** in protein sequence-structure tasks.  

---

### **1. Summary of Tables**  
#### **(1) Performance Comparison of Protein VQ Models (Table 2)**  
- **Purpose**: Compare the reconstruction performance of traditional **Vector Quantization (VQ) methods** with the proposed **Soft Conditional Vector Quantization (SoftCVQ)**  
- **Metrics**:  
  - **Global Metrics**: TMScore (structural similarity), Recovery Rate (sequence accuracy)  
  - **Local Metrics**: Lr (structural error), Lα (angle error), Lβ (torsion angle error)  
  - **Success Rate**: TMScore ≥ 0.5 or Recovery Rate ≥ 95%  
- **Results**:  
  - **SoftCVQ outperforms traditional VQ methods** (Vanilla VQ, LFQ) with a **TMScore of 0.74 and a success rate of 94.9%**.  
  - SoftVQ and SoftGVQ also show improvements over previous models, but SoftCVQ achieves the best balance between reconstruction accuracy and generative capability.  

#### **(2) Backbone Inpainting Performance (Table 3)**  
- **Purpose**: Evaluate whether **FoldGPT** outperforms existing protein sequence and structure generation models (e.g., ESM2, EvoDiff, FoldingDiff).  
- **Metrics**:  
  - **Sequence Recovery (Rec)**: Measures how accurately FoldGPT reconstructs missing protein sequences.  
  - **TMScore**: Evaluates structure prediction accuracy.  
- **Results**:  
  - **FoldGPT achieves a sequence recovery rate of 96.2% and a TMScore of 0.80**, surpassing baseline models.  
  - Notably, it **outperforms angle-based methods** (FoldingDiff, DiffSDS), though there is room for improvement compared to coordinate-based models (ProtDiff, RFDiffusion, Chroma).  

#### **(3) Hyperparameter Study for SoftCVQ (Table 4)**  
- **Purpose**: Analyze the impact of various hyperparameters (number of MLP layers, codebook vector dimensions, spherical normalization) on SoftCVQ’s performance.  
- **Results**:  
  - Increasing MLP layers and codebook vector dimensions improves performance.  
  - **Applying spherical normalization leads to higher TMScore and sequence recovery rates**, confirming its effectiveness.  

---

### **2. Summary of Figures**  
#### **(1) FoldToken Architecture Overview (Figure 1)**  
- **Explanation**:  
  - **Stage 1**: FoldTokenizer transforms protein sequences and structures into discrete tokens.  
  - **Stage 2**: FoldGPT predicts protein sequences and structures using the tokenized representation.  
  - SoftCVQ resolves common issues in VQ, such as **Gradient Mismatch and Semantic Irrelevance**.  

#### **(2) Comparison of Vector Quantization Methods (Figure 2)**  
- **Explanation**:  
  - Comparison of Vanilla VQ, Lookup-Free VQ (LFQ), SoftVQ, SoftGVQ, and SoftCVQ.  
  - SoftCVQ significantly improves both **reconstruction quality and generative performance**.  

#### **(3) Structure of SoftCVQ (Figure 3)**  
- **Explanation**:  
  - Compares SoftVQ, SoftGVQ, and SoftCVQ.  
  - SoftCVQ introduces an optimized **codebook access method**, improving protein reconstruction.  

#### **(4) Visualization of Reconstructed Protein Structures (Figure 5, Appendix)**  
- **Explanation**:  
  - SoftCVQ **recovers protein structures more accurately than previous VQ methods**.  
  - Visualizations include **TMScore and sequence recovery rates** to demonstrate performance improvements.  

---

These figures and tables collectively illustrate **the advantages of FoldTokenizer and FoldGPT**, demonstrating **superior sequence-structure modeling and generation capabilities** compared to previous methods.



<br/>
# refer format:     



@article{Gao2025FoldToken,
  author    = {Zhangyang Gao and Cheng Tan and Jue Wang and Yufei Huang and Lirong Wu and Stan Z. Li},
  title     = {FoldToken: Learning Protein Language via Vector Quantization and Beyond},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence (AAAI)}
}




Gao, Zhangyang, Cheng Tan, Jue Wang, Yufei Huang, Lirong Wu, and Stan Z. Li. 2025. "FoldToken: Learning Protein Language via Vector Quantization and Beyond." Proceedings of the AAAI Conference on Artificial Intelligence. Association for the Advancement of Artificial Intelligence (AAAI).  






