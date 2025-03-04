---
layout: post
title:  "[2025]A Unifying Information-theoretic Perspective on Evaluating Generative Models"  
date:   2025-03-04 09:17:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


재미있는 새로운 평가지표들  
(정밀도 크로스 엔트로피(Precision Cross-Entropy, PCE), 재현율 크로스 엔트로피(Recall Cross-Entropy, RCE), 재현율 엔트로피(Recall Entropy, RE))    


짧은 요약(Abstract) :    




생성 모델의 출력 해석이 어려운 문제를 해결하기 위해, 연구자들은 의미 있는 평가 지표를 찾는 데 집중하고 있다. 최근 연구에서는 분류 모델에서 차용한 "정확도(precision)"와 "재현율(recall)" 개념을 사용하여 생성된 출력의 사실성(현실성)과 다양성을 각각 측정한다. 하지만 다양한 평가 지표가 제안됨에 따라, 이들의 장단점을 명확히 비교할 수 있는 통합적 시각이 필요하다. 이를 위해, 저자들은 k-최근접 이웃(kNN) 밀도 추정을 기반으로 한 여러 평가 지표를 정보 이론적 관점에서 통합하였다. 또한, 정밀도 크로스 엔트로피(Precision Cross-Entropy, PCE), 재현율 크로스 엔트로피(Recall Cross-Entropy, RCE), 재현율 엔트로피(Recall Entropy, RE)로 구성된 삼차원 평가 지표를 제안하며, 이는 사실성과 다양성을 각각 측정하는 두 가지 측면(클래스 간 및 클래스 내)을 고려한다. 제안된 지표는 정보 이론의 엔트로피 및 크로스 엔트로피 개념을 기반으로 하며, 샘플 및 모드 수준에서 분석할 수 있다. 실험 결과, 이 지표는 개별 품질 요소에 민감하며, 기존 평가 지표가 가지는 문제점을 명확히 드러낸다.

---


To address the challenge of interpreting generative model outputs, researchers are focused on identifying meaningful evaluation metrics. Recent approaches adopt the classification-based concepts of "precision" and "recall" to separately quantify output fidelity (realism) and output diversity (representation of real data variation). However, with the increasing number of proposed metrics, a unifying perspective is necessary for clearer comparisons and explanations of their benefits and limitations. To this end, the authors unify a class of k-nearest neighbors (kNN)-based metrics through an information-theoretic lens using kNN density estimation techniques. Additionally, they propose a tri-dimensional metric composed of Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), and Recall Entropy (RE), which separately measure fidelity and two distinct aspects of diversity (inter-class and intra-class). The proposed metric, derived from the information-theoretic concepts of entropy and cross-entropy, allows for both sample- and mode-level analysis. Experimental results demonstrate that this metric is sensitive to its respective quality components and highlights the shortcomings of other existing metrics.


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




이 논문에서는 생성 모델 평가를 위한 새로운 정보 이론 기반 메트릭을 제안한다.  
메서드는 크게 **(1) 기존 평가 지표의 통합적 분석**, **(2) 새로운 평가 지표 제안**, **(3) 실험적 검증** 세 부분으로 구성된다.

#### **1. 모델 및 아키텍처**  
- 기존의 평가 지표로 사용되는 **k-최근접 이웃(kNN) 기반 정밀도 및 재현율 평가 방법**을 정보 이론적 관점에서 분석하였다.  
- 새로운 평가 지표로 **Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), Recall Entropy (RE)** 를 제안하여 생성 모델의 사실성과 다양성을 정량적으로 평가한다.  
- 제안된 지표는 kNN 기반 밀도 추정 및 정보 이론적 개념(엔트로피, 크로스 엔트로피)을 활용하여 샘플 수준과 모드 수준에서 분석 가능하도록 설계되었다.  

#### **2. 훈련 데이터**  
- 평가 실험은 대표적인 이미지 데이터셋 **ImageNet** 및 **CIFAR-10**을 사용하여 진행되었다.  
- ImageNet 데이터셋에서는 1,000개 클래스에서 각 클래스당 100개의 이미지를 샘플링하였고, CIFAR-10에서는 10개 클래스에서 각 클래스당 4,500개의 이미지를 샘플링하였다.  
- 평가를 위한 생성된 이미지는 **DiT-XL-2 모델** 및 **ADMG-ADMU (Diffusion 모델)**을 사용하여 생성되었다.  

#### **3. 실험 방법**  
- **모드 축소(Mode Shrinkage) 실험**: 클래스 내부 다양성을 줄이면서(= 더 정밀하게) 생성하는 설정을 사용하여 다양한 평가 지표의 변화를 분석하였다.  
- **모드 드롭핑(Mode Dropping) 실험**: ImageNet에서 점진적으로 클래스 수를 줄이며 모델이 다양성을 유지하는지를 확인하였다.  
- **인간 평가와의 상관 분석**: 제안된 PCE 지표가 사람이 판단한 이미지 품질과 높은 상관관계를 갖는지를 분석하였다.  

실험 결과, 제안된 정보 이론 기반 평가 지표(PCE, RCE, RE)는 기존 평가 방법보다 **생성 모델의 현실성(fidelity)과 다양성(diversity) 평가에 더욱 효과적**임을 보였다.

---


This paper proposes a new information-theoretic metric for evaluating generative models.  
The methodology consists of **(1) a unified analysis of existing evaluation metrics, (2) the proposal of a novel metric, and (3) experimental validation**.

#### **1. Models and Architecture**  
- The study examines **k-nearest neighbor (kNN)-based precision and recall evaluation methods** from an information-theoretic perspective.  
- A new set of metrics—**Precision Cross-Entropy (PCE), Recall Cross-Entropy (RCE), and Recall Entropy (RE)**—is introduced to quantitatively assess the fidelity and diversity of generative models.  
- The proposed metric leverages **kNN-based density estimation** and **information-theoretic concepts (entropy, cross-entropy)** for both sample-level and mode-level analysis.  

#### **2. Training Data**  
- The evaluation experiments were conducted using **ImageNet** and **CIFAR-10**, two widely used image datasets.  
- From **ImageNet**, 1,000 classes were sampled, with 100 images per class. From **CIFAR-10**, 10 classes were used, with 4,500 images per class.  
- Generated images were obtained using **DiT-XL-2** and **ADMG-ADMU (a Diffusion model)**.  

#### **3. Experimental Methodology**  
- **Mode Shrinkage Experiment**: Analyzed the effect of reducing intra-class diversity while increasing precision.  
- **Mode Dropping Experiment**: Evaluated how well models preserved diversity by gradually reducing the number of ImageNet classes.  
- **Correlation with Human Evaluation**: Investigated how well the proposed PCE metric aligns with human-perceived image quality.  

The experimental results demonstrated that the proposed **information-theoretic metrics (PCE, RCE, RE) outperform existing evaluation methods** in assessing the fidelity and diversity of generative models.


   
 
<br/>
# Results  



이 논문의 실험에서는 **기존 생성 모델 평가 메트릭과 비교하여 제안된 정보 이론 기반 메트릭(PCE, RCE, RE)의 성능을 분석**하였다.  
실험은 **(1) 비교 대상 경쟁 모델, (2) 테스트 데이터, (3) 평가 메트릭 분석** 세 가지로 구성된다.

#### **1. 경쟁 모델 (비교 모델)**  
제안된 메트릭을 검증하기 위해, **기존 생성 모델 평가 메트릭**과 비교 분석하였다.  
- **프레셰 인셉션 거리 (FID, Fréchet Inception Distance)**: 기존에 널리 사용되는 1차원 평가 지표로, 현실성과 다양성을 한 지표로 평가한다.  
- **Precision & Recall 기반 메트릭**: 다양한 kNN 기반 정밀도 및 재현율 평가 방법을 포함하며, 다음과 같은 모델을 비교 대상으로 삼았다.  
  - **Precision Coverage (PC), Recall Coverage (RC)** (Cheema & Urner, 2023)  
  - **Density & Coverage (D & C)** (Naeem et al., 2020)  

#### **2. 테스트 데이터**  
평가 실험은 **ImageNet**과 **CIFAR-10**을 사용하여 진행되었다.  
- **ImageNet**: 1,000개 클래스에서 각 클래스당 100개 이미지를 샘플링.  
- **CIFAR-10**: 10개 클래스에서 각 클래스당 4,500개 이미지를 샘플링.  
- **생성된 이미지 모델**: **DiT-XL-2 (Diffusion Transformer)** 및 **ADMG-ADMU (Diffusion 모델)**를 사용하여 생성.  

#### **3. 평가 메트릭 분석 (결과 요약)**  
**제안된 정보 이론 기반 메트릭(PCE, RCE, RE)은 기존 평가 방법보다 생성 모델의 현실성(fidelity)과 다양성(diversity) 평가에 더욱 효과적**임을 입증하였다.  

1. **인간 평가와의 상관 분석**  
   - 제안된 **Precision Cross-Entropy (PCE)**는 인간 평가와 높은 상관관계를 보였다.  
   - 기존의 **Density (D) 메트릭**은 데이터셋에 따라 상관관계가 불안정함을 보였다.  

2. **모드 축소(Mode Shrinkage) 실험**  
   - 생성 모델이 클래스 내부 다양성을 줄이며 사실성을 증가시키는 설정에서,  
     - 기존 **Precision Coverage (PC) 메트릭은 정확한 변화를 반영하지 못함**.  
     - 반면, **PCE 및 Recall Entropy (RE)는 적절한 변화를 반영하며 더욱 신뢰성 있는 지표**임을 확인.  

3. **모드 드롭핑(Mode Dropping) 실험**  
   - ImageNet에서 점진적으로 클래스 수를 줄이며 모델이 다양성을 유지하는지 확인한 실험에서,  
     - 기존 메트릭(FID 등)은 **정확한 다차원적 평가를 수행하지 못함**.  
     - **RCE (Recall Cross-Entropy) 및 RC (Recall Coverage)**는 **모드 드롭핑에 매우 민감한 반응을 보이며, 보다 신뢰할 수 있는 다양성 평가를 제공**.  

4. **모드-샘플 분석 가능성**  
   - 기존 메트릭들은 모델의 오류 원인을 구체적으로 분석하는 데 어려움이 있었음.  
   - **PCE, RCE, RE는 모드 수준과 개별 샘플 수준에서 생성 모델의 한계를 정량적으로 분석할 수 있도록 설계됨**.  

---

The experiments in this paper analyze the performance of the **proposed information-theoretic metrics (PCE, RCE, RE)** compared to existing generative model evaluation metrics.  
The evaluation consists of three key aspects: **(1) competing models, (2) test data, and (3) metric performance analysis**.

#### **1. Competing Models (Baseline Comparisons)**  
To validate the proposed metrics, comparisons were made with widely used **existing generative model evaluation metrics**:  
- **Fréchet Inception Distance (FID)**: A widely used one-dimensional metric that assesses both realism and diversity in a single score.  
- **Precision & Recall-based Metrics**: Various **kNN-based precision and recall evaluation methods**, including:  
  - **Precision Coverage (PC) & Recall Coverage (RC)** (Cheema & Urner, 2023)  
  - **Density & Coverage (D & C)** (Naeem et al., 2020)  

#### **2. Test Data**  
The experiments were conducted on **ImageNet** and **CIFAR-10** datasets:  
- **ImageNet**: 1,000 classes, with 100 images sampled per class.  
- **CIFAR-10**: 10 classes, with 4,500 images per class.  
- **Generated Image Models**: **DiT-XL-2 (Diffusion Transformer)** and **ADMG-ADMU (Diffusion model)** were used for image generation.  

#### **3. Evaluation Metrics Analysis (Key Findings)**  
The results **demonstrate that the proposed information-theoretic metrics (PCE, RCE, RE) provide a more reliable assessment of fidelity and diversity in generative models than existing methods**.  

1. **Correlation with Human Evaluation**  
   - **Precision Cross-Entropy (PCE) exhibited a strong correlation with human judgment**.  
   - **Density (D) metric showed inconsistencies across datasets**, making it unreliable.  

2. **Mode Shrinkage Experiment**  
   - When intra-class diversity was reduced to improve realism:  
     - **Precision Coverage (PC) failed to capture the expected changes**.  
     - **PCE and Recall Entropy (RE) correctly reflected these changes**, proving to be more reliable indicators.  

3. **Mode Dropping Experiment**  
   - As the number of classes in ImageNet was reduced to test diversity preservation:  
     - **Existing metrics (e.g., FID) failed to distinguish the multidimensional aspects of evaluation**.  
     - **RCE (Recall Cross-Entropy) and RC (Recall Coverage) effectively captured the loss of diversity**, making them more reliable for mode dropping analysis.  

4. **Mode & Sample-Level Analysis Capability**  
   - Traditional metrics struggle to **pinpoint the specific sources of errors in generative models**.  
   - **PCE, RCE, and RE were designed for both mode-level and sample-level analysis, allowing for detailed diagnostic insights into generative model failures**.  



<br/>
# 예제  



이 논문의 실험에서는 **ImageNet과 CIFAR-10을 사용하여 생성 모델 평가를 진행하며, 다양한 실험을 통해 제안된 메트릭(PCE, RCE, RE)의 성능을 분석**하였다.  

---

### **1. 훈련 데이터 (Train Data)**  
- **ImageNet**  
  - 1,000개 클래스에서 각 클래스당 **100개의 이미지를 샘플링**하여 사용.  
- **CIFAR-10**  
  - 10개 클래스에서 각 클래스당 **4,500개의 이미지를 샘플링**하여 사용.  
- **생성 모델**  
  - **DiT-XL-2 (Diffusion Transformer)** 및 **ADMG-ADMU (Diffusion 모델)**을 사용하여 이미지 생성.  

---

### **2. 테스트 데이터 (Test Data)**  
- **ImageNet 테스트셋**  
  - 생성된 이미지와 실제 이미지 간 비교를 위해, **1,000개 클래스의 각 클래스당 100개 샘플 사용**.  
- **CIFAR-10 테스트셋**  
  - 10개 클래스에서 **각 클래스당 4,500개 샘플 사용**.  
- **테스트 평가 방식**  
  - 기존 FID, Density & Coverage(D & C), Precision Coverage(PC)와 비교하여 **제안된 정보 이론 기반 평가 지표(PCE, RCE, RE)의 성능 분석**.  

---

### **3. 테스크 입력 및 출력 (Task Input/Output)**  

#### **(1) 모드 축소 실험 (Mode Shrinkage Experiment)**  
- **입력(Input)**:  
  - Diffusion 모델을 사용하여 **다양한 Classifier-Free Guidance (CFG) 값**으로 이미지 생성.  
  - **CFG 값을 높일수록 intra-class 다양성이 감소하고 realism(사실성)이 증가**.  
- **출력(Output)**:  
  - 기존 PC, Density 등의 메트릭은 **변화를 제대로 반영하지 못함**.  
  - 제안된 **PCE 및 Recall Entropy (RE)는 생성 모델의 intra-class 다양성 감소를 정확하게 측정**함.  

#### **(2) 모드 드롭핑 실험 (Mode Dropping Experiment)**  
- **입력(Input)**:  
  - ImageNet에서 **점진적으로 클래스 개수를 줄이면서(1,000 → 900 → ... → 100), 생성 모델이 모든 클래스를 학습하는지 평가**.  
- **출력(Output)**:  
  - 기존 FID는 **정확한 다차원적 평가를 수행하지 못함**.  
  - **RCE (Recall Cross-Entropy) 및 RC (Recall Coverage)는 클래스 감소에 민감하게 반응하며, 보다 신뢰할 수 있는 다양성 평가를 제공**.  

#### **(3) 인간 평가와의 상관 분석 (Correlation with Human Evaluation)**  
- **입력(Input)**:  
  - 인간 평가 실험 데이터 활용, 인간이 평가한 "이미지 품질 점수"를 비교.  
- **출력(Output)**:  
  - 기존 Density(D) 메트릭은 **데이터셋에 따라 상관관계가 불안정함**.  
  - **제안된 PCE는 인간 평가와 높은 상관관계를 가짐**, 즉 실제로 현실적으로 보이는 이미지를 잘 측정하는 지표임을 입증.  

---


The experiments in this paper **evaluate generative model performance using ImageNet and CIFAR-10 datasets, analyzing the effectiveness of the proposed metrics (PCE, RCE, RE)**.

---

### **1. Training Data**  
- **ImageNet**  
  - 1,000 classes, **100 images per class sampled**.  
- **CIFAR-10**  
  - 10 classes, **4,500 images per class sampled**.  
- **Generative Models Used**  
  - **DiT-XL-2 (Diffusion Transformer)** and **ADMG-ADMU (Diffusion model)** were used for image generation.  

---

### **2. Test Data**  
- **ImageNet Test Set**  
  - **1,000 classes, 100 samples per class** used for real-vs-generated image comparison.  
- **CIFAR-10 Test Set**  
  - **10 classes, 4,500 samples per class** used.  
- **Evaluation Method**  
  - The proposed **information-theoretic metrics (PCE, RCE, RE)** were compared against existing metrics like **FID, Density & Coverage (D & C), and Precision Coverage (PC)**.  

---

### **3. Task Input and Output**  

#### **(1) Mode Shrinkage Experiment**  
- **Input:**  
  - Images were generated using a diffusion model with **various Classifier-Free Guidance (CFG) values**.  
  - **Higher CFG values reduce intra-class diversity but improve realism**.  
- **Output:**  
  - **Existing PC and Density metrics failed to capture these changes properly**.  
  - **PCE and Recall Entropy (RE) successfully measured intra-class diversity loss**, making them more reliable.  

#### **(2) Mode Dropping Experiment**  
- **Input:**  
  - ImageNet **classes were gradually removed (1,000 → 900 → ... → 100) to evaluate whether the generative model retained full class diversity**.  
- **Output:**  
  - **Existing FID failed to provide a multi-dimensional evaluation**.  
  - **RCE (Recall Cross-Entropy) and RC (Recall Coverage) effectively captured class reduction**, providing a more reliable diversity assessment.  

#### **(3) Correlation with Human Evaluation**  
- **Input:**  
  - **Human-assessed image quality scores** were used as ground truth.  
- **Output:**  
  - **Density (D) metric showed inconsistent correlation across datasets**.  
  - **PCE exhibited a strong correlation with human judgment**, proving its reliability in measuring realism.  




<br/>  
# 요약   




이 논문은 생성 모델의 평가를 위해 정보 이론적 개념(엔트로피, 크로스 엔트로피)을 기반으로 한 새로운 메트릭(PCE, RCE, RE)을 제안하며, 기존 kNN 기반 정밀도 및 재현율 평가 지표를 통합적으로 분석하였다. 실험 결과, 제안된 메트릭은 기존 메트릭(FID, PC, D & C)보다 생성 모델의 현실성과 다양성을 효과적으로 평가하며, 특히 인간 평가와 높은 상관관계를 보였다. ImageNet 및 CIFAR-10 데이터셋을 사용한 실험에서, 모드 축소 및 모드 드롭핑을 정량적으로 분석할 수 있는 능력을 입증하며, 기존 평가 방식이 놓칠 수 있는 모델 오류를 보다 정확하게 식별할 수 있음을 확인하였다.  

---


This paper proposes new evaluation metrics (PCE, RCE, RE) based on information-theoretic concepts (entropy, cross-entropy) and unifies existing kNN-based precision and recall metrics for assessing generative models. Experimental results show that the proposed metrics outperform traditional methods (FID, PC, D & C) in measuring realism and diversity, exhibiting a strong correlation with human evaluations. Using ImageNet and CIFAR-10 datasets, the experiments demonstrate the ability to quantitatively analyze mode shrinkage and mode dropping, identifying model errors that conventional evaluation methods might overlook.


<br/>  
# 기타  




이 논문에는 **생성 모델 평가 메트릭의 성능을 시각적으로 분석하는 다양한 그래프 및 표(테이블)**가 포함되어 있다.  

1. **Figure 1: 생성 모델의 실패 유형 다이어그램**  
   - **설명:** 생성된 데이터 분포와 실제 데이터 분포 간의 차이를 나타내는 세 가지 주요 실패 유형(Mode Invention, Mode Dropping, Mode Shrinkage)을 시각적으로 표현.  
   - **핵심 내용:**  
     - **(a) Mode Dropping:** 특정 클래스(예: 픽업 트럭)가 생성되지 않음.  
     - **(b) Mode Invention:** 실제 존재하지 않는 이상한 데이터 포인트 생성.  
     - **(c) Mode Shrinkage:** 데이터가 특정 중심점에 과도하게 집중됨.  

2. **Table 1: KL 다이버전스 및 크로스 엔트로피 값 비교**  
   - **설명:** 다양한 분산(σ²) 값을 가진 생성 데이터 분포에 대한 KL 다이버전스(DKL), 크로스 엔트로피(CE), 및 엔트로피(H) 값 비교.  
   - **핵심 내용:**  
     - 생성 데이터의 분산이 너무 작거나 클 경우, KL 다이버전스 값이 증가함.  
     - 제안된 PCE 메트릭이 이러한 변화를 효과적으로 반영함을 시사.  

3. **Figure 2: 인간 평가와의 상관 분석 그래프**  
   - **설명:** 기존 메트릭(FID, PC, Density)과 제안된 메트릭(PCE) 간 인간 평가와의 상관관계를 분석한 Pearson 상관 행렬.  
   - **핵심 내용:**  
     - PCE는 인간 평가와 높은 상관관계를 가지며, 기존 Density 메트릭(D)은 데이터셋에 따라 불안정한 결과를 보임.  

4. **Figure 3: Mode Shrinkage 실험 결과 그래프**  
   - **설명:** Classifier-Free Guidance (CFG) 값 변화에 따른 다양한 평가 메트릭의 변화를 시각화한 그래프.  
   - **핵심 내용:**  
     - CFG 값 증가 → 기존 PC 메트릭은 비정상적 패턴을 보이지만, PCE 및 RE는 정확하게 변화를 포착.  

5. **Figure 4: Mode Dropping 실험 결과 그래프**  
   - **설명:** ImageNet 데이터셋에서 점진적으로 클래스를 줄이는 과정에서 다양한 평가 메트릭의 변화를 분석한 그래프.  
   - **핵심 내용:**  
     - 기존 메트릭(FID)은 변화에 둔감하지만, RCE와 RC는 클래스 감소에 민감하게 반응.  

---


This paper includes **various figures (tables, diagrams, and graphs) that visually analyze the performance of generative model evaluation metrics**.

1. **Figure 1: Diagram of Generative Model Failure Types**  
   - **Description:** A visual representation of three key failure types in generative models (Mode Invention, Mode Dropping, Mode Shrinkage).  
   - **Key Points:**  
     - **(a) Mode Dropping:** Certain classes (e.g., pickup trucks) are missing from generated data.  
     - **(b) Mode Invention:** Unrealistic, non-existent samples are produced.  
     - **(c) Mode Shrinkage:** Generated samples cluster too closely around a mode’s average.  

2. **Table 1: Comparison of KL Divergence and Cross-Entropy Values**  
   - **Description:** Comparison of KL divergence (DKL), cross-entropy (CE), and entropy (H) across different variance (σ²) values for generated data distributions.  
   - **Key Points:**  
     - When variance is too small or too large, KL divergence increases.  
     - The proposed PCE metric effectively captures these variations.  

3. **Figure 2: Correlation Analysis with Human Evaluation**  
   - **Description:** Pearson correlation matrix comparing human evaluation scores with existing metrics (FID, PC, Density) and the proposed PCE metric.  
   - **Key Points:**  
     - **PCE strongly correlates with human judgment**, whereas the existing Density (D) metric shows inconsistencies across datasets.  

4. **Figure 3: Mode Shrinkage Experiment Results**  
   - **Description:** Graph showing changes in evaluation metrics as Classifier-Free Guidance (CFG) values vary.  
   - **Key Points:**  
     - As CFG increases, **PC fails to reflect changes correctly, while PCE and RE accurately capture intra-class diversity loss**.  

5. **Figure 4: Mode Dropping Experiment Results**  
   - **Description:** Graph analyzing the changes in evaluation metrics as ImageNet classes are progressively removed.  
   - **Key Points:**  
     - **FID fails to detect class reduction, but RCE and RC show strong sensitivity to mode dropping**, making them more reliable diversity metrics.


<br/>
# refer format:     



@article{fox2025unifying,
  author = {Fox, Alexis and Swarup, Samarth and Adiga, Abhijin},
  title = {A Unifying Information-theoretic Perspective on Evaluating Generative Models},
  journal = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year = {2025},
  url = {https://arxiv.org/abs/2412.14340}
}




Alexis Fox, Samarth Swarup, and Abhijin Adiga. "A Unifying Information-theoretic Perspective on Evaluating Generative Models." Proceedings of the AAAI Conference on Artificial Intelligence, 2025. Available at https://arxiv.org/abs/2412.14340.





