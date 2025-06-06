---
layout: post
title:  "[2025]Synergy of GFlowNet and Protein Language Model Makes a Diverse Antibody Designer"  
date:   2025-03-03 18:31:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


강화 학습(RL)과 에너지 기반 모델(EBM)에서 영감을 받은 확률적 생성 모델인 GFlowNet과 
(분자 생성, 최적화 문제, 구조 탐색 등의 문제에서 강점, 화합물 및 단백질 서열 생성 등의 작업에 사용된 바 있음)  
단백질언어모델(ProtBERT, ProGen2) 등을 함께 활용  

* GFlowNet -> 항체 CDR(Complementarity-Determining Region, 단백질서열) 서열을 확률적으로 탐색하고 생성  
* 단백질 언어 모델  -> 생성된 항체 서열이 생물학적으로 자연스러운지 평가(파인튜닝 없이 Perplexity , log likelihood(높을 수록 자연스럽), Amino Acid Recovery (AAR, %) 기존 서열과 유사성으로)     
* Potts Model(수학모델, 단백질 스핀기반) -> 역시 서열이 자연스러운지 평가  

짧은 요약(Abstract) :    




항체는 항원과 높은 특이도로 결합하여 면역 방어 역할을 한다. 이 중 **Complementarity-Determining Region (CDR)**이 결합의 주요 역할을 하지만, 새로운 CDR을 실험적으로 발견하는 과정은 매우 시간 소모적이다. 최근 **단백질 언어 모델(Protein Language Model, PLM)**이 항체 설계에 활용되지만, 기존 모델들은 주로 개발 가능성(developability)에만 집중하여 다양한 CDR을 충분히 탐색하지 못하는 문제가 있다. 

이 연구에서는 **GFlowNet과 PLM을 결합한 PG-AbD 프레임워크**를 제안하여, 효과적이며 다양한 항체 후보를 생성하는 새로운 방법을 소개한다. PG-AbD는 **전역 분포 모델링을 수행하는 PLM과 지역 분포를 모델링하는 Potts Model을 결합하여 GFlowNet의 보상 함수로 활용**한다. 이때, 대조 발산(contrastive divergence) 기법을 적용하여 부정 샘플을 생성하고 GFlowNet이 다양한 항체 후보를 탐색하도록 유도하는 **공동 학습 방식(joint training paradigm)**을 도입하였다.

실험 결과, PG-AbD는 기존 방법 대비 항체 다양성을 크게 향상시키면서(예: RabDab 데이터셋에서 13.5% 증가, SabDab에서 31.1% 증가) 개발 가능성과 신규성(novelty)도 유지하였다. 생성된 항체들은 3D 구조 분석에서도 안정적인 결합을 보이며, 항체 설계의 실용성을 증명하였다. 

이 연구는 항체 설계를 가속화하는 데 중요한 기여를 할 수 있다.

---


Antibodies defend our health by binding to antigens with high specificity, primarily relying on the Complementarity-Determining Region (CDR). Yet, current experimental methods of discovering new antibody CDRs are heavily time-consuming. Computational design could alleviate this burden, with protein language models demonstrating remarkable utility in many recent studies. However, most existing models solely focus on antibody developability and struggle to encapsulate the diverse range of plausible CDR candidates, limiting their effectiveness in real-world scenarios as binding is only one factor in the multitude of drug-forming criteria.

In this paper, we introduce **PG-AbD**, a framework uniting **Generative Flow Networks (GFlowNets) and pretrained Protein Language Models (PLMs)** to successfully generate highly potent, diverse, and novel antibody candidates. We innovatively construct a **Products of Experts (PoE)** composed of the **global-distribution-modeling PLM and the local-distribution-modeling Potts Model** to serve as the reward function of GFlowNet. The **joint training paradigm** is introduced, where PoE is trained by contrastive divergence with the negative samples generated by GFlowNet, and then guides GFlowNet to sample diverse antibody candidates.

We evaluate PG-AbD on extensive antibody design benchmarks. It significantly outperforms existing methods in diversity (**13.5% on RabDab, 31.1% on SabDab**) while maintaining optimal developability and novelty. Generated antibodies are also found to form stable and regular 3D structures with their corresponding antigens, demonstrating the great potential of PG-AbD to accelerate real-world antibody discovery.



 

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




이 논문에서는 **PG-AbD**라는 새로운 프레임워크를 제안하여 **Generative Flow Networks (GFlowNets)**과 **Protein Language Models (PLMs)**을 결합하여 다양한 항체 후보를 생성하는 방법을 연구한다. 방법론은 크게 다음과 같은 요소들로 구성된다.

#### **1. PG-AbD 프레임워크 개요**
PG-AbD는 **항체 설계를 위한 생성 모델**로, **GFlowNet**을 활용하여 단백질 언어 모델(PLM)이 제공하는 정보를 효과적으로 학습하며, **Products of Experts (PoE) 보상 함수**를 통해 항체 서열의 개발 가능성, 다양성, 신규성을 최적화한다.

#### **2. GFlowNet 기반 항체 서열 생성**
- GFlowNet은 **비지도 탐색(self-exploring)** 능력을 활용하여 다양한 항체 서열을 생성한다.
- **보상 함수(reward function)**로 **사전 학습된 PLM (ProGen2)**을 사용하여 생성된 서열이 자연스러운지 평가한다.
- FiLM(feature-wise linear modulation) 모듈을 적용하여 항체 디자인 방향을 효과적으로 조정한다.

#### **3. Products of Experts (PoE) 보상 함수**
- 항체 설계는 **전역적(global) 제약**과 **국소적(local) 제약**을 모두 고려해야 한다.
- 전역적 제약: **사전 학습된 PLM(ProGen2)**을 사용하여 일반적인 단백질 서열의 특징을 반영.
- 국소적 제약: **Potts Model**을 활용하여 항체 서열 내 특정 아미노산 간 상호작용과 보존성(conservation)을 모델링.
- PoE는 PLM과 Potts Model의 결합을 통해 최적의 항체 서열을 평가하는 보상 함수를 구성한다.

#### **4. 공동 학습 (Joint Training)**
- GFlowNet과 PoE를 **대조 발산(contrastive divergence)** 기법을 통해 공동 학습한다.
- GFlowNet이 샘플링한 서열을 사용하여 PoE를 업데이트하고, PoE는 GFlowNet의 보상 함수로 작용하여 더욱 다양한 항체를 생성할 수 있도록 유도한다.
- 이를 통해 PG-AbD는 기존 항체 데이터가 부족한 경우에도 효과적인 항체 디자인이 가능하다.

#### **5. 트레이닝 데이터**
- PG-AbD는 두 가지 데이터셋에서 검증되었다:
  1. **RabDab 데이터셋**: 기존 학습 데이터를 사용하지 않고 항체 설계 능력을 평가.
  2. **SabDab 데이터셋**: 기존 항체 서열을 활용하여 최적화 모델을 학습하고 성능을 비교.

이러한 방법을 통해 PG-AbD는 기존 방법보다 **항체 다양성(13.5% 증가), 신규성(31.1% 증가), 개발 가능성 유지**를 모두 만족하는 성과를 보였다.

---


The paper introduces **PG-AbD**, a novel framework combining **Generative Flow Networks (GFlowNets)** and **Protein Language Models (PLMs)** to generate diverse and novel antibody candidates. The methodology consists of the following key components:

#### **1. Overview of PG-AbD Framework**
PG-AbD is a **generative model for antibody design**, leveraging **GFlowNet** for effective sequence exploration and using a **Products of Experts (PoE) reward function** to optimize developability, diversity, and novelty of antibody sequences.

#### **2. GFlowNet-Based Antibody Sequence Generation**
- GFlowNet enables **self-exploration** to generate diverse antibody sequences.
- The **reward function** is provided by a **pretrained PLM (ProGen2)**, which evaluates the naturalness of generated sequences.
- A **FiLM (feature-wise linear modulation) module** is applied to guide the design process effectively.

#### **3. Products of Experts (PoE) Reward Function**
- Antibody design requires both **global** and **local** constraints.
- **Global constraint**: A **pretrained PLM (ProGen2)** models general protein sequence features.
- **Local constraint**: A **Potts Model** captures specific amino acid interactions and conservation patterns within antibody sequences.
- PoE integrates these models, forming a **comprehensive reward function** to evaluate antibody sequences.

#### **4. Joint Training Approach**
- PG-AbD employs **contrastive divergence** to jointly train **GFlowNet and PoE**.
- GFlowNet generates candidate sequences, which PoE evaluates and updates accordingly.
- PoE then acts as the reward function for GFlowNet, ensuring an optimal balance between exploration and exploitation.
- This approach allows **PG-AbD to design antibodies effectively even with limited training data**.

#### **5. Training Data**
- PG-AbD was validated using two benchmark datasets:
  1. **RabDab Dataset**: Evaluates antibody generation ability **without additional training data**.
  2. **SabDab Dataset**: Uses existing antibody sequences to train an optimized model for **comparative evaluation**.

Through this approach, **PG-AbD significantly improves antibody diversity (+13.5%) and novelty (+31.1%) while maintaining optimal developability**, outperforming existing methods.

   




   
 
<br/>
# Results  



이 연구에서는 PG-AbD의 성능을 평가하기 위해 다양한 실험을 수행하였다. **경쟁 모델**, **테스트 데이터**, 그리고 **성능 평가 메트릭**을 포함한 결과를 종합적으로 분석하면 다음과 같다.

---

### **1. 비교 대상 모델 (경쟁 모델)**
PG-AbD는 기존의 여러 항체 설계 방법과 성능을 비교하였다.  
**비교 모델**은 다음과 같다:
- **기존 딥러닝 모델**  
  - LSTM (순환 신경망 기반 모델)  
  - CEM (Constrained Energy Model)  
  - AntibodyDiff (확산 기반 생성 모델)
- **그래프 신경망(GNN) 기반 모델**  
  - AR-GNN  
  - refine-GNN (항체 구조까지 고려하는 최적화 모델)
- **단백질 언어 모델 (PLMs)**  
  - ProtBert  
  - AbLang-H  
  - ESM-2  
  - reprogBERT  
  - RosettaAntibodyDesign (RAbD, 물리 기반 모델)

이러한 모델들과 PG-AbD를 비교하여 **다양성, 신규성, 개발 가능성** 측면에서 성능을 평가하였다.

---

### **2. 테스트 데이터셋**
모델 성능을 검증하기 위해 두 개의 대표적인 **항체 데이터셋**을 사용하였다:
1. **RabDab 데이터셋**  
   - 훈련 데이터를 사용하지 않고 항체를 생성하는 **"제로샷(Zero-Shot)"** 실험을 수행.
   - PG-AbD는 GFlowNet을 통해 **자율 탐색(Self-exploring)** 방식으로 항체를 생성.
2. **SabDab 데이터셋**  
   - 기존 항체 서열 데이터를 학습하여 CDR을 최적화하는 실험을 수행.
   - **PoE(Predicted Probability Function)를 통한 GFlowNet 훈련**을 적용.

---

### **3. 성능 평가 메트릭**
PG-AbD의 성능을 다양한 지표를 통해 평가하였다:
- **Perplexity (PPL)**: 모델이 생성한 항체 서열이 얼마나 자연스러운지를 평가 (낮을수록 좋음).
- **Amino Acid Recovery (AAR, %)**: 원본 항체 서열과의 일치율을 측정하여 개발 가능성(developability) 평가 (높을수록 좋음).
- **Novelty (NOV, %)**: 생성된 항체가 기존 데이터와 얼마나 다른지 평가 (높을수록 좋음).
- **Diversity (DIV, %)**: 생성된 항체 서열이 얼마나 다양한지를 평가 (높을수록 좋음).

---

### **4. 주요 실험 결과**
#### **(1) RabDab 데이터셋 결과 (Zero-shot 항체 설계)**
- PG-AbD는 **항체 다양성(DIV)에서 13.5% 향상** (기존 최고 성능: 62.73% → PG-AbD: 76.59%).
- 개발 가능성(AAR)과 PPL 또한 기존 모델과 비슷한 수준으로 유지.

#### **(2) SabDab 데이터셋 결과 (항체 최적화)**
- PG-AbD는 **항체 다양성을 31.1% 향상**, 신규성 또한 기존 모델 대비 높은 성능을 보임.
- **ProtBert는 AAR이 높지만 다양성이 낮음**, 반면 **AbLang-H와 ESM-2는 다양성은 높지만 개발 가능성이 떨어짐**.
- **PG-AbD는 개발 가능성(AAR 18.3% 향상)과 다양성(31.1% 향상)을 균형 있게 유지**.

#### **(3) 3D 구조 분석 (AlphaFold-Multimer)**
- PG-AbD로 생성된 항체와 항원의 3D 결합 구조를 AlphaFold-Multimer로 분석한 결과:
  - **평균 pLDDT (구조 신뢰도): 85.68**
  - **최대 pLDDT: 91.59**  
  → PG-AbD가 생성한 항체는 **안정적이며 생물학적으로 의미 있는 구조를 형성**함을 증명.

---

### **5. 종합적인 결론**
- PG-AbD는 **기존 모델 대비 항체 다양성과 신규성을 크게 향상**하면서도 개발 가능성을 유지.
- **Zero-shot 항체 설계 실험에서도 높은 성능을 보이며**, 추가 학습 없이도 유의미한 항체 서열을 생성 가능.
- PG-AbD가 **항체 설계를 가속화할 수 있는 새로운 접근법**으로 활용될 가능성이 높음.

---


This study evaluates **PG-AbD**'s performance through extensive experiments. The results are analyzed based on **competitor models, test datasets, and evaluation metrics**.

---

### **1. Competitor Models**
PG-AbD was compared against multiple **existing antibody design methods**, including:

- **Deep Learning-Based Models**  
  - LSTM (Recurrent Neural Network-based model)  
  - CEM (Constrained Energy Model)  
  - AntibodyDiff (Diffusion-based generative model)
- **Graph Neural Network (GNN)-Based Models**  
  - AR-GNN  
  - refine-GNN (Structure-aware antibody optimization model)
- **Protein Language Models (PLMs)**  
  - ProtBert  
  - AbLang-H  
  - ESM-2  
  - reprogBERT  
  - RosettaAntibodyDesign (RAbD, physics-based model)

These models were used to evaluate **diversity, novelty, and developability** of PG-AbD.

---

### **2. Test Datasets**
Two **benchmark antibody datasets** were used for evaluation:

1. **RabDab Dataset**  
   - Zero-shot **(without training data)** antibody generation.
   - PG-AbD utilizes **self-exploration via GFlowNet** to design new antibodies.
2. **SabDab Dataset**  
   - Used for antibody CDR optimization.
   - **GFlowNet training with PoE (Predicted Probability Function).**

---

### **3. Evaluation Metrics**
PG-AbD's performance was assessed using the following key metrics:

- **Perplexity (PPL)**: Evaluates the naturalness of generated antibody sequences (lower is better).
- **Amino Acid Recovery (AAR, %)**: Measures sequence similarity to the original antibody, assessing developability (higher is better).
- **Novelty (NOV, %)**: Determines how different the generated antibodies are from existing ones (higher is better).
- **Diversity (DIV, %)**: Measures sequence diversity among generated antibodies (higher is better).

---

### **4. Key Experimental Results**
#### **(1) RabDab Dataset Results (Zero-shot Antibody Design)**
- PG-AbD **improved antibody diversity (DIV) by 13.5%** (Best previous model: 62.73% → PG-AbD: 76.59%).
- Developability (AAR) and PPL remained at competitive levels.

#### **(2) SabDab Dataset Results (Antibody Optimization)**
- PG-AbD **improved antibody diversity by 31.1%**, outperforming existing models in novelty.
- **ProtBert achieved high AAR but low diversity**, while **AbLang-H and ESM-2 had high diversity but poor developability**.
- **PG-AbD balanced both developability (+18.3% AAR) and diversity (+31.1%)**, leading to superior overall performance.

#### **(3) 3D Structural Analysis (AlphaFold-Multimer)**
- PG-AbD-generated antibodies were analyzed for 3D binding structures using **AlphaFold-Multimer**:
  - **Average pLDDT (structure confidence score): 85.68**
  - **Maximum pLDDT: 91.59**  
  → This confirms that PG-AbD-generated antibodies form **stable and biologically meaningful structures**.

---

### **5. Overall Conclusion**
- **PG-AbD significantly improves antibody diversity and novelty while maintaining developability** compared to existing methods.
- **Excels in zero-shot antibody design**, generating meaningful sequences without additional training.
- PG-AbD presents a **novel approach to accelerating antibody discovery**, making it a promising tool for real-world applications.

 
<br/>
# 예제  





이 연구에서는 **PG-AbD 프레임워크의 성능을 평가하기 위해 다양한 실험을 수행**하였으며, 실험 설정, 테스트 및 훈련 데이터, 그리고 주요 입력/출력 예시는 다음과 같다.

---

### **1. 실험 설정 (테스트 방식)**
PG-AbD의 성능을 검증하기 위해 두 가지 주요 실험을 수행하였다:
1. **Zero-shot 항체 설계 (RabDab 데이터셋)**
   - 기존 데이터를 학습하지 않고 **GFlowNet이 자율 탐색(self-exploring) 방식**으로 항체를 생성.
   - 생성된 항체 서열이 자연스러운지(Perplexity), 기존 항체와 얼마나 다른지(Diversity, Novelty), 개발 가능성이 높은지(Amino Acid Recovery) 평가.

2. **항체 최적화 (SabDab 데이터셋)**
   - 기존 항체 서열 데이터를 학습하여, 특정 CDR(Complementarity-Determining Region)의 최적화를 수행.
   - **PoE(Products of Experts) 모델을 학습하여** GFlowNet이 최적의 항체 서열을 생성할 수 있도록 유도.

---

### **2. 데이터셋 (훈련 및 테스트 데이터)**
#### **(1) RabDab 데이터셋 (Zero-shot 학습)**
- **훈련 데이터 없음** → PG-AbD는 **자율 탐색(self-exploring)** 기능을 통해 학습 없이 항체 서열을 생성.
- 테스트 데이터는 60개의 항체 서열로 구성됨.

#### **(2) SabDab 데이터셋 (CDR 최적화)**
- 훈련 데이터: 기존 항체 서열을 포함한 **SabDab 데이터셋**.
- 테스트 데이터: 학습하지 않은 새로운 항체 서열을 포함하며, 최적화된 CDR을 평가.

---

### **3. 주요 입력 및 출력 예시**
PG-AbD의 테스트 과정에서 사용된 입력 및 출력 예시는 다음과 같다:

#### **(1) Zero-shot 항체 설계 예시**
##### **입력 (GFlowNet 탐색을 위한 초기 시퀀스)**
```text
VL: QVQLQESGPGLVKPSETLSLTCTVSGGSISSGYYYWSWIRQP
VH: EVQLEKPGASVKISCKASGYSFTNYGMNWVRQAPGKGLEWV
CDR: __________ (GFlowNet이 생성할 영역)
```

##### **출력 (PG-AbD가 생성한 새로운 CDR)**
```text
VL: QVQLQESGPGLVKPSETLSLTCTVSGGSISSGYYYWSWIRQP
VH: EVQLEKPGASVKISCKASGYSFTNYGMNWVRQAPGKGLEWV
CDR: ARDTVLGGMDV
```
- 생성된 CDR이 기존 항체 서열과 자연스럽게 연결됨.
- 평가 지표:
  - **Perplexity (PPL) = 4.26 (낮을수록 좋음)**
  - **Diversity (DIV) = 76.59% (높을수록 좋음)**

---

#### **(2) 항체 최적화 예시 (CDR-H3)**
##### **입력 (원본 항체 서열)**
```text
Antigen: SARS coronavirus
CDR-H3 (원본): ARGDTFLGGWMDV
```

##### **출력 (PG-AbD가 최적화한 CDR-H3)**
```text
Antigen: SARS coronavirus
CDR-H3 (최적화 결과): ARDTVLGGMDV
```
- 최적화된 CDR이 원본보다 **더 나은 항원 결합력을 가질 가능성이 높음**.
- 평가 지표:
  - **Amino Acid Recovery (AAR) = 58.40%**
  - **Diversity (DIV) = 62.73%**
  - **pLDDT (3D 구조 신뢰도) = 85.68**

---

### **4. 결론**
PG-AbD는 **Zero-shot 학습 및 최적화 실험에서 기존 모델을 뛰어넘는 성능을 보였으며**, 생성된 항체 서열이 **높은 다양성과 개발 가능성을 유지**하는 것이 확인되었다.

---



The study conducted extensive **PG-AbD performance evaluations**, and the experimental setup, test/training data, and key input/output examples are described as follows.

---

### **1. Experimental Setup (Testing Approach)**
Two major experiments were conducted to validate PG-AbD's performance:
1. **Zero-shot Antibody Design (RabDab Dataset)**
   - **GFlowNet generates antibody sequences through self-exploration**, without any training data.
   - The generated antibody sequences are evaluated based on **Perplexity (PPL), Diversity (DIV), Novelty (NOV), and Developability (AAR).**

2. **Antibody Optimization (SabDab Dataset)**
   - Existing antibody sequences are used for training.
   - **PoE (Products of Experts) reward model is trained** to guide GFlowNet in generating optimized sequences.

---

### **2. Dataset (Training & Test Data)**
#### **(1) RabDab Dataset (Zero-shot learning)**
- **No training data** → PG-AbD relies solely on **self-exploration** to generate antibody sequences.
- Test set: 60 antibody sequences.

#### **(2) SabDab Dataset (CDR Optimization)**
- Training data: **SabDab dataset containing known antibody sequences.**
- Test data: New antibody sequences for evaluating CDR optimization.

---

### **3. Key Input & Output Examples**
The following examples illustrate how PG-AbD operates in test scenarios:

#### **(1) Zero-shot Antibody Design Example**
##### **Input (Initial sequence for GFlowNet exploration)**
```text
VL: QVQLQESGPGLVKPSETLSLTCTVSGGSISSGYYYWSWIRQP
VH: EVQLEKPGASVKISCKASGYSFTNYGMNWVRQAPGKGLEWV
CDR: __________ (Generated by GFlowNet)
```

##### **Output (New CDR generated by PG-AbD)**
```text
VL: QVQLQESGPGLVKPSETLSLTCTVSGGSISSGYYYWSWIRQP
VH: EVQLEKPGASVKISCKASGYSFTNYGMNWVRQAPGKGLEWV
CDR: ARDTVLGGMDV
```
- The generated CDR naturally integrates with the existing antibody sequence.
- Evaluation metrics:
  - **Perplexity (PPL) = 4.26 (Lower is better)**
  - **Diversity (DIV) = 76.59% (Higher is better)**

---

#### **(2) Antibody Optimization Example (CDR-H3)**
##### **Input (Original antibody sequence)**
```text
Antigen: SARS coronavirus
CDR-H3 (Original): ARGDTFLGGWMDV
```

##### **Output (Optimized CDR-H3 by PG-AbD)**
```text
Antigen: SARS coronavirus
CDR-H3 (Optimized): ARDTVLGGMDV
```
- The optimized CDR is expected to have **better antigen-binding potential**.
- Evaluation metrics:
  - **Amino Acid Recovery (AAR) = 58.40%**
  - **Diversity (DIV) = 62.73%**
  - **pLDDT (3D structure confidence) = 85.68**

---

### **4. Conclusion**
PG-AbD demonstrated **superior performance in both Zero-shot and Optimization tasks**, successfully generating **highly diverse and developable antibody sequences** compared to existing models.



<br/>  
# 요약   



PG-AbD는 **GFlowNet과 단백질 언어 모델(PLM)**을 결합하여 **항체 다양성, 신규성, 개발 가능성을 최적화하는 새로운 항체 설계 프레임워크**이다. 실험 결과, PG-AbD는 **RabDab 데이터셋에서 13.5%, SabDab 데이터셋에서 31.1% 더 높은 다양성을 보이며, 기존 모델 대비 우수한 성능을 입증**하였다. Zero-shot 학습과 최적화 실험에서 **생성된 항체 CDR 서열이 기존보다 더 안정적이고 효과적인 결합력을 가질 가능성이 확인되었다**.  

---



PG-AbD is a **novel antibody design framework combining GFlowNet and Protein Language Models (PLMs) to optimize diversity, novelty, and developability**. Experimental results show that PG-AbD **outperforms existing models, achieving 13.5% higher diversity on RabDab and 31.1% on SabDab datasets**. In zero-shot and optimization tasks, **the generated CDR sequences demonstrate greater stability and enhanced antigen-binding potential**.


<br/>  
# 기타  



이 논문에는 PG-AbD의 성능과 방법론을 설명하는 다양한 **피규어(Figures)와 테이블(Tables)**이 포함되어 있다. 주요 내용을 정리하면 다음과 같다.

---

### **1. PG-AbD의 개요 (Figure 2, Figure 3)**  
- **Figure 2: PG-AbD 프레임워크 개요**  
  - (a) 기존 항체 설계 방법: **사전 학습된 PLM을 이용하여 항체 서열을 생성**하지만 다양성과 신규성이 부족함.  
  - (b) PG-AbD 방법론: **GFlowNet이 PLM을 활용하여 자율 탐색**하며, **FiLM 모듈을 통해 디자인 방향을 조정**.  
  - 결과적으로 **GFlowNet은 더욱 다양한 항체 서열을 생성**할 수 있음.  

- **Figure 3: PG-AbD의 Products of Experts (PoE) 보상 모델**  
  - GFlowNet과 PoE의 **공동 학습(joint training) 과정**을 시각화.  
  - **PoE는 PLM(전역 제약)과 Potts Model(국소 제약)을 결합**하여 보상 함수 역할을 수행.  
  - GFlowNet은 **PoE에서 제공하는 보상 신호를 기반으로 항체 서열을 지속적으로 탐색**.  
  - **Contrastive Divergence 기법을 사용하여 PoE와 GFlowNet이 상호 보완적으로 학습**.  

---

### **2. 실험 결과 (Table 1, Table 2, Figure 4, Figure 5)**  
- **Table 1: RabDab 데이터셋에서 PG-AbD의 성능**  
  - PG-AbD는 **항체 다양성(DIV)에서 기존 모델 대비 13.5% 향상**.  
  - **Perplexity (PPL)가 낮고, Amino Acid Recovery (AAR)가 높아 개발 가능성도 유지**.  
  - 특히 **FiLM 모듈을 추가했을 때 성능이 더욱 향상됨**.  

- **Table 2: SabDab 데이터셋에서 PG-AbD의 성능**  
  - 항체 다양성(DIV)이 기존 대비 **31.1% 증가**.  
  - PG-AbD는 **다른 모델들과 비교했을 때, AAR과 PPL에서도 균형 잡힌 성능을 보임**.  
  - ProtBert, AbLang, ESM-2 등과 비교하여 **PG-AbD가 개발 가능성과 다양성에서 가장 높은 점수를 기록**.  

- **Figure 4: 3D 구조 신뢰도 평가 (AlphaFold-Multimer 분석)**  
  - PG-AbD가 생성한 항체-항원 복합체의 **pLDDT 점수(구조 신뢰도)가 평균 85.68, 최대 91.59**로 높음.  
  - PG-AbD가 기존 PLM 기반 모델보다 **안정적인 3D 결합 구조를 생성함을 증명**.  

- **Figure 5: 항체-항원 결합 구조 예시**  
  - PG-AbD가 생성한 **항체가 SARS-CoV-2, 인간 TFPI, 인플루엔자 A 바이러스와 결합**한 사례를 시각화.  
  - 항체의 CDR-H3 서열이 최적화되어 **안정적이고 효과적인 결합을 형성**.  

---

### **3. 추가 실험 (Table 3, Table 4)**  
- **Table 3: 다양한 PLM을 활용한 ablation study**  
  - PG-AbD의 보상 함수로 **ProGen2 PLM을 사용했을 때 가장 높은 다양성과 낮은 PPL을 기록**.  
  - **ESM-2, ProtBert, AbLang-H, reprogBERT와 비교**하여 **ProGen2가 최적의 성능을 보임**.  

- **Table 4: PoE 모델 구성 요소별 성능 비교**  
  - **Potts Model을 제거하면 AAR이 크게 감소**하고, **PLM을 제거하면 PPL이 상승**.  
  - 즉, **PoE 모델이 항체 설계에서 중요한 역할을 수행함을 입증**.  

---

### **결론**
- PG-AbD는 **PLM과 GFlowNet을 결합하여 높은 다양성과 개발 가능성을 갖춘 항체 서열을 생성**함.  
- **RabDab 및 SabDab 데이터셋 실험에서 기존 방법보다 우수한 성능을 보였으며**,  
- **3D 구조 분석에서도 안정적인 결합이 확인됨**.  
- 실험 결과를 종합적으로 볼 때, **PG-AbD는 항체 설계 자동화를 위한 강력한 도구가 될 가능성이 높음**.  

---



This paper includes multiple **figures and tables** that illustrate PG-AbD's methodology and performance. Below is a summary of key visuals:

---

### **1. Overview of PG-AbD (Figure 2, Figure 3)**  
- **Figure 2: PG-AbD Framework Overview**  
  - (a) **Existing antibody design methods**: Use **pretrained PLMs to generate antibody sequences**, but they lack diversity and novelty.  
  - (b) **PG-AbD approach**: **GFlowNet leverages PLMs for self-exploration**, with a **FiLM module guiding the design process**.  
  - As a result, **GFlowNet generates more diverse antibody sequences**.  

- **Figure 3: Products of Experts (PoE) Reward Model in PG-AbD**  
  - Visualizes the **joint training process between GFlowNet and PoE**.  
  - **PoE combines PLM (global constraints) and Potts Model (local constraints) to form a reward function**.  
  - **GFlowNet explores antibody sequences based on PoE's reward signals**, enhancing diversity and novelty.  
  - **Contrastive divergence technique enables PoE and GFlowNet to learn in a complementary manner**.  

---

### **2. Experimental Results (Table 1, Table 2, Figure 4, Figure 5)**  
- **Table 1: PG-AbD performance on RabDab dataset**  
  - **PG-AbD improves antibody diversity (DIV) by 13.5%** over existing models.  
  - **Low Perplexity (PPL) and high Amino Acid Recovery (AAR) indicate strong developability**.  
  - **Adding the FiLM module further enhances performance**.  

- **Table 2: PG-AbD performance on SabDab dataset**  
  - **PG-AbD increases antibody diversity by 31.1%**, outperforming prior models.  
  - Compared to **ProtBert, AbLang, and ESM-2**, **PG-AbD achieves the best balance between developability and diversity**.  

- **Figure 4: 3D Structural Confidence Evaluation (AlphaFold-Multimer Analysis)**  
  - **PG-AbD-generated antibodies achieve an average pLDDT of 85.68 and a maximum of 91.59**.  
  - **Demonstrates PG-AbD's ability to generate structurally stable antigen-antibody complexes**.  

- **Figure 5: Antibody-Antigen Binding Examples**  
  - **Visualization of PG-AbD-generated antibodies binding to SARS-CoV-2, human TFPI, and influenza A virus**.  
  - **Optimized CDR-H3 sequences form stable and effective antigen binding sites**.  

---

### **3. Additional Experiments (Table 3, Table 4)**  
- **Table 3: Ablation Study on Different PLMs**  
  - **Using ProGen2 as the reward function yielded the highest diversity and lowest PPL**.  
  - **Compared to ESM-2, ProtBert, AbLang-H, and reprogBERT, ProGen2 performed the best**.  

- **Table 4: Impact of PoE Model Components**  
  - **Removing the Potts Model significantly reduced AAR**, and **removing PLM increased PPL**.  
  - **Confirms that PoE plays a crucial role in antibody design**.  

---

### **Conclusion**
- **PG-AbD integrates PLMs and GFlowNet to generate antibody sequences with superior diversity and developability**.  
- **It outperforms existing methods in both RabDab and SabDab experiments**,  
- **and its generated structures exhibit stable antigen-binding interactions in 3D analysis**.  
- **Overall, PG-AbD presents a powerful approach for automated antibody discovery**.




<br/>
# refer format:     



@article{Yin2025PGAbD,
  author    = {Mingze Yin and Hanjing Zhou and Yiheng Zhu and Jialu Wu and Wei Wu and Mingyang Li and Kun Fu and Zheng Wang and Chang-Yu Hsieh and Tingjun Hou and Jian Wu},
  title     = {Synergy of GFlowNet and Protein Language Model Makes a Diverse Antibody Designer},
  journal   = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025},
  publisher = {Association for the Advancement of Artificial Intelligence},
  
}






Yin, Mingze, Hanjing Zhou, Yiheng Zhu, Jialu Wu, Wei Wu, Mingyang Li, Kun Fu, Zheng Wang, Chang-Yu Hsieh, Tingjun Hou, and Jian Wu. 2025. "Synergy of GFlowNet and Protein Language Model Makes a Diverse Antibody Designer." Proceedings of the AAAI Conference on Artificial Intelligence. Association for the Advancement of Artificial Intelligence.

