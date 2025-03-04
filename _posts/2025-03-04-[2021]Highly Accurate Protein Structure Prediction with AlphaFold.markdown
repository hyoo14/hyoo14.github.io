---
layout: post
title:  "[2025]Highly Accurate Protein Structure Prediction with AlphaFold"  
date:   2025-03-04 10:10:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

단백질 서열 입력받아서 3d구조인 좌표를 출력  
(트랜스포머 사용->아미노산 간의 관계, 서열들간의 보존관계 두가지를 학습 : 알파폴드라 일컬음)    



짧은 요약(Abstract) :    




이 연구에서는 **AlphaFold**라는 신경망 기반 모델을 이용하여 **단백질의 3D 구조를 높은 정확도로 예측**하는 방법을 제시한다. 현재까지 100,000개 이상의 단백질 구조가 실험적으로 밝혀졌으나, 이는 알려진 단백질 서열의 극히 일부에 불과하다. 단백질 접힘 문제(protein folding problem)에서 **아미노산 서열만을 이용하여 단백질의 3D 구조를 예측**하는 것은 50년 이상 해결되지 않은 도전 과제였다. 기존의 예측 방법들은 원자 수준의 정확도를 제공하지 못했으며, 특히 기존의 유사한 단백질 구조 정보가 없는 경우 그 정확도가 떨어졌다. 

이번 연구에서 개발된 **새로운 AlphaFold 모델은, 유사한 구조가 존재하지 않는 경우에도 원자 수준의 정확도로 단백질 구조를 예측할 수 있는 최초의 컴퓨터 모델**이다. 이 모델은 국제 단백질 구조 예측 대회 **CASP14**에서 검증되었으며, **대부분의 사례에서 실험적으로 결정된 구조와 경쟁할 수 있는 수준의 정확도를 보였다**. AlphaFold의 핵심 요소는 **물리적 및 생물학적 지식을 심층 신경망 설계에 통합하고, 다중 서열 정렬(multiple sequence alignments, MSA) 정보를 활용하는 새로운 머신러닝 접근법**이다. 

이 연구는 **대규모 구조 생물정보학 연구를 가능하게 하고, 단백질 기능 및 상호작용의 이해를 가속화할 수 있는 획기적인 진보**를 이루었다.

---


This study introduces **AlphaFold**, a neural network-based model that achieves **highly accurate 3D protein structure prediction**. Despite experimental efforts determining over **100,000 protein structures**, this represents only a fraction of known protein sequences. The **protein folding problem**, predicting a protein's 3D structure solely from its amino acid sequence, has remained unsolved for over 50 years. Existing computational methods have failed to reach atomic-level accuracy, particularly when no homologous structures are available. 

The **new AlphaFold model** developed in this study is the **first computational approach capable of predicting protein structures with atomic accuracy, even in the absence of similar known structures**. The model was validated in the **CASP14 protein structure prediction challenge**, where it **achieved accuracy comparable to experimentally determined structures in most cases, significantly outperforming competing methods**. The breakthrough comes from a **novel machine learning approach that integrates physical and biological knowledge into deep neural network design while leveraging multiple sequence alignments (MSA)**.

This advancement **enables large-scale structural bioinformatics research and accelerates our understanding of protein functions and interactions**.





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




이 논문에서는 **AlphaFold**의 핵심적인 모델 아키텍처와 학습 방법을 소개한다. AlphaFold는 **심층 신경망(deep neural network)** 기반의 접근법을 사용하여 단백질의 3D 구조를 예측하며, 특히 **진화적 제약(evolutionary constraints)과 물리적 원리를 통합**하여 기존 모델보다 더 높은 정확도를 달성했다. 

#### **1. 모델 아키텍처**
AlphaFold의 모델 아키텍처는 크게 **두 단계**로 구성된다. 
1. **Evoformer**: 단백질 서열에서 다중 서열 정렬(MSA)을 활용하여 **서열 간의 관계를 학습**하고, **각 아미노산 간의 구조적 관계를 추론**하는 신경망 블록.
2. **구조 모듈 (Structure Module)**: Evoformer가 생성한 정보를 기반으로 **단백질의 3D 좌표를 직접 예측**하는 모듈.

Evoformer는 **다중 서열 정렬(MSA)과 아미노산 간의 관계(pair representation)** 를 입력으로 받아 반복적으로 정보를 교환하며 단백질의 구조적 특징을 학습한다. 이후 **구조 모듈은 각 잔기의 회전과 변환을 예측하여** 최종적으로 원자 수준의 3D 구조를 산출한다.

#### **2. 학습 데이터 (Training Data)**
AlphaFold는 **PDB (Protein Data Bank)**의 단백질 구조 데이터를 학습에 사용했다. 주요 데이터 소스는 다음과 같다.
- **Uniref90, BFD (Big Fantastic Database), MGnify** 등의 단백질 서열 데이터
- **PDB70** (단백질 템플릿 검색을 위한 데이터)
- **MSA (Multiple Sequence Alignments) 데이터**: HHblits 및 Jackhmmer를 사용하여 구축

#### **3. 학습 과정**
- 모델은 **약 10M개의 샘플을 학습**하며, 작은 크기의 단백질 조각(256개 아미노산)부터 점차 큰 구조로 확장하는 방식으로 훈련되었다.
- **자기 지도 학습(self-distillation)** 을 활용하여 **비지도 학습(unlabeled sequence data)** 도 함께 학습했다.
- 모델의 예측 성능을 향상시키기 위해 **재귀적 반복(recycling)** 기법을 사용하여 점진적으로 구조를 정교화했다.

이러한 방법을 통해 AlphaFold는 **CASP14 평가에서 실험적으로 확인된 단백질 구조와 거의 동등한 정확도**를 달성했으며, 기존 방법을 크게 능가하는 결과를 보였다.

---



This paper introduces the **core architecture and training methodology** of AlphaFold, a deep learning-based model that predicts protein structures with high accuracy. AlphaFold integrates **evolutionary constraints and physical principles**, outperforming previous computational approaches.

#### **1. Model Architecture**
AlphaFold's model consists of **two main stages**:
1. **Evoformer**: A neural network block that **extracts relational information from multiple sequence alignments (MSAs)** and **infers inter-residue relationships**.
2. **Structure Module**: Uses the learned representations from Evoformer to **directly predict the 3D atomic coordinates** of the protein.

Evoformer takes **MSA representations and pairwise relationships** as inputs and refines the structural features iteratively. The **Structure Module then predicts the final 3D conformation by determining the rotation and translation of each residue**.

#### **2. Training Data**
AlphaFold was trained using protein structure data from the **Protein Data Bank (PDB)**, along with additional sources:
- **Uniref90, BFD (Big Fantastic Database), and MGnify** for sequence information.
- **PDB70** for protein template search.
- **MSA data** generated using HHblits and Jackhmmer.

#### **3. Training Process**
- The model was trained on **~10 million samples**, starting with small protein fragments (256 residues) and gradually scaling up.
- **Self-distillation** was employed, leveraging **unlabeled sequence data** for improved learning.
- **Iterative refinement (recycling)** was used to progressively enhance structural predictions.

With these approaches, AlphaFold **achieved near-experimental accuracy in the CASP14 assessment**, significantly outperforming previous methods.



   
 
<br/>
# Results  





AlphaFold는 **단백질 구조 예측의 최신 벤치마크인 CASP14(Critical Assessment of protein Structure Prediction)**에서 **기존 방법들을 크게 능가하는 성능**을 보였다. 실험적으로 결정된 단백질 구조와 비교했을 때 **대부분의 경우에서 원자 수준의 정확도를 달성**했다.

#### **1. 경쟁 모델 및 비교**
AlphaFold는 CASP14에서 다른 단백질 구조 예측 모델들과 비교되었으며, 대표적인 경쟁 모델은 다음과 같다.
- **기존의 동역학 기반 모델 (Physics-based models)**: 분자 시뮬레이션을 통해 단백질 접힘을 예측하는 방식이지만, **계산 비용이 높고 정확도가 낮음**.
- **전통적인 생물정보학 기반 모델 (Bioinformatics-based models)**: 이미 밝혀진 단백질 구조와의 상동성(homology) 분석을 활용하는 방법으로, **유사한 구조가 없는 경우 성능이 크게 떨어짐**.
- **딥러닝 기반 모델** (예: RoseTTAFold): 신경망을 이용하여 구조 예측을 수행하지만, AlphaFold만큼의 **원자 수준 정확도를 제공하지 못함**.

AlphaFold는 이러한 모델들과 비교하여 **대부분의 단백질에서 가장 정확한 예측을 수행**했다.

#### **2. 테스트 데이터**
AlphaFold는 CASP14에서 제공한 **비공개 데이터셋**을 사용하여 평가되었다.
- CASP14 데이터셋은 **실험적으로 밝혀진 단백질 구조**이지만, 예측 모델이 사전에 접근할 수 없도록 설정됨.
- AlphaFold는 **유사한 단백질 구조가 없는 경우에도 높은 정확도를 유지**하는 것이 확인됨.

추가적으로, CASP14 평가 이후 **PDB (Protein Data Bank)에서 최근 공개된 단백질 구조**에서도 평가를 진행하였으며, CASP14 데이터에서와 유사한 성능을 보였다.

#### **3. 평가 메트릭 (Metrics)**
AlphaFold의 성능은 다음과 같은 메트릭을 사용하여 평가되었다.
- **Cα RMSD (Root Mean Square Deviation)**: 단백질 백본의 원자 좌표 차이를 측정하는 지표. **AlphaFold의 중간값: 0.96Å**, 경쟁 모델(2.8Å) 대비 **훨씬 높은 정확도**.
- **TM-score (Template Modeling Score)**: 단백질 구조의 전반적인 유사성을 측정하는 지표. **AlphaFold는 0.93~0.96 범위의 높은 TM-score**를 기록.
- **lDDT (Local Distance Difference Test)**: 지역적인 구조 정확도를 평가하는 지표. AlphaFold는 기존 모델 대비 **가장 높은 lDDT 점수를 기록**.

AlphaFold의 결과는 **현재까지 개발된 단백질 구조 예측 모델 중 가장 높은 정확도를 기록**했으며, 특히 **원자 수준의 세밀한 구조까지 정확하게 예측할 수 있음**을 입증했다.

---



AlphaFold **outperformed all existing protein structure prediction methods** in **CASP14 (Critical Assessment of protein Structure Prediction)**, achieving **near-experimental accuracy in most cases**.

#### **1. Competitor Models & Comparisons**
AlphaFold was compared against several competing models in CASP14, including:
- **Physics-based models**: These methods rely on molecular dynamics simulations but **are computationally expensive and less accurate**.
- **Bioinformatics-based models**: Utilize homology modeling but **fail when no similar structures exist**.
- **Deep learning-based models** (e.g., RoseTTAFold): Use neural networks for structure prediction but **lack AlphaFold’s atomic-level accuracy**.

Compared to these models, **AlphaFold consistently provided the most accurate predictions**.

#### **2. Test Data**
AlphaFold was evaluated on the **CASP14 blind dataset**, which consists of **experimentally determined protein structures that were not publicly available**.
- AlphaFold **maintained high accuracy even for proteins with no homologous structures**.
- Additional testing on **recently released PDB structures** confirmed that AlphaFold **performed similarly well beyond CASP14**.

#### **3. Evaluation Metrics**
AlphaFold’s performance was assessed using the following metrics:
- **Cα RMSD (Root Mean Square Deviation)**: Measures atomic-level deviation from experimental structures. **AlphaFold achieved a median Cα RMSD of 0.96Å**, significantly better than the next-best model (2.8Å).
- **TM-score (Template Modeling Score)**: Assesses overall structural similarity. AlphaFold achieved **high TM-scores (0.93–0.96)**.
- **lDDT (Local Distance Difference Test)**: Evaluates local accuracy. **AlphaFold consistently achieved the highest lDDT scores** among all models.

AlphaFold’s results demonstrated **unprecedented accuracy in protein structure prediction**, making it the most precise model developed to date, with **atomic-level structural resolution**.


<br/>
# 예제  






AlphaFold의 성능을 평가하기 위해 **학습 데이터(Training Data)**, **테스트 데이터(Test Data)**, 그리고 **테스크 입력/출력(Task Input/Output)**이 설정되었다.

---

### **1. 학습 데이터 (Training Data)**
AlphaFold는 **대규모 단백질 데이터베이스**를 사용하여 훈련되었으며, 대표적인 데이터 소스는 다음과 같다.
- **PDB (Protein Data Bank)**: 실험적으로 밝혀진 단백질 구조 데이터.
- **Uniref90**: 단백질 서열을 포함한 고유한 데이터셋.
- **BFD (Big Fantastic Database)**: 약 2억 개 이상의 단백질 서열을 포함하는 대규모 데이터베이스.
- **MGnify**: 메타유전체 데이터를 포함한 단백질 서열 데이터베이스.
- **PDB70**: 단백질 템플릿 검색을 위한 데이터.

학습 과정에서 **다중 서열 정렬(Multiple Sequence Alignment, MSA)**과 **구조 템플릿(Structural Templates)**이 결합되어 사용되었으며, 모델이 단백질 서열 간의 관계를 효과적으로 학습할 수 있도록 설계되었다.

---

### **2. 테스트 데이터 (Test Data)**
AlphaFold의 성능을 검증하기 위해 **CASP14 데이터셋**과 **새로운 PDB 데이터셋**이 사용되었다.
- **CASP14 데이터셋**: 
  - 국제 단백질 구조 예측 대회(CASP)에서 제공한 **비공개 단백질 구조 데이터**.
  - 실험적으로 밝혀졌지만, 예측 모델이 접근할 수 없는 상태에서 평가됨.
  - **유사한 단백질 구조가 없는 단백질을 포함하여 평가**.
- **새로운 PDB 데이터셋**:
  - CASP14 이후 **PDB에 새롭게 추가된 단백질 구조**를 대상으로 추가 평가 진행.
  - **기존 학습 데이터에 포함되지 않은 새로운 단백질 서열**을 통해 일반화 성능 검증.

---

### **3. 테스크 입력/출력 (Task Input/Output)**
#### **입력 (Input)**
- **단백질 서열(Amino Acid Sequence)**: 
  - 단백질을 구성하는 **아미노산 서열 (예: "MKTLLVLLYTF…")**.
  - 다중 서열 정렬(MSA)과 결합하여 신경망에 입력됨.

#### **출력 (Output)**
- **단백질의 3D 구조(3D Protein Structure)**:
  - 각 아미노산의 **원자 수준의 좌표**(XYZ 좌표 포함).
  - **단백질의 접힘(folding) 형태**를 결정하는 원자 배치 정보.
  - **구조의 신뢰도(Confidence Score)**를 포함한 예측값 제공.

---

### **실제 예제**
**예제 입력 (Input Example)**
```plaintext
MKTLLVLLYTFATANADSGM
```
**예제 출력 (Output Example)**
```plaintext
Residue: 1  XYZ: (12.34, 5.67, 8.90)
Residue: 2  XYZ: (13.12, 6.45, 9.23)
...
Confidence Score: 92.3%
```
이러한 구조 예측은 **실험적으로 결정된 구조와 비교하여 원자 수준에서 정확도를 평가**할 수 있도록 한다.

---



To evaluate AlphaFold’s performance, **training data**, **test data**, and **task input/output** were defined.

---

### **1. Training Data**
AlphaFold was trained using **large-scale protein databases**, including:
- **PDB (Protein Data Bank)**: Experimentally determined protein structures.
- **Uniref90**: A dataset containing unique protein sequences.
- **BFD (Big Fantastic Database)**: A large-scale database with over 200 million protein sequences.
- **MGnify**: A metagenomic protein sequence database.
- **PDB70**: Used for searching structural templates.

The training process involved **multiple sequence alignments (MSA)** and **structural templates**, allowing the model to effectively learn relationships between protein sequences.

---

### **2. Test Data**
To validate AlphaFold’s accuracy, **CASP14 dataset** and **new PDB dataset** were used.
- **CASP14 Dataset**:
  - **Blind protein structure dataset** provided by the international CASP competition.
  - Consists of **experimentally determined but undisclosed protein structures**.
  - **Includes proteins with no homologous structures** for fair evaluation.
- **New PDB Dataset**:
  - Additional validation using **recently added structures from PDB**.
  - **Ensures generalization to previously unseen protein sequences**.

---

### **3. Task Input/Output**
#### **Input**
- **Amino Acid Sequence**:
  - A **string representing the amino acid sequence** (e.g., `"MKTLLVLLYTF…"`)
  - Combined with **multiple sequence alignments (MSA)** as input to the neural network.

#### **Output**
- **3D Protein Structure**:
  - **Atomic coordinates (XYZ positions) for each residue**.
  - **Folding conformation of the protein**.
  - **Confidence score for the predicted structure**.

---

### **Actual Example**
**Example Input:**
```plaintext
MKTLLVLLYTFATANADSGM
```
**Example Output:**
```plaintext
Residue: 1  XYZ: (12.34, 5.67, 8.90)
Residue: 2  XYZ: (13.12, 6.45, 9.23)
...
Confidence Score: 92.3%
```
This structural prediction can then be compared to **experimentally determined structures to assess atomic-level accuracy**.


<br/>  
# 요약   




AlphaFold는 다중 서열 정렬(MSA)과 구조 템플릿을 활용하는 심층 신경망 모델을 통해 단백질의 3D 구조를 예측한다. CASP14 평가에서 기존 모델보다 높은 원자 수준의 정확도를 기록했으며, 새로운 PDB 데이터에서도 우수한 성능을 보였다. 입력된 단백질 서열을 바탕으로 원자 좌표와 신뢰도 점수를 포함한 3D 구조를 출력한다.

---


AlphaFold predicts 3D protein structures using a deep neural network that leverages multiple sequence alignments (MSA) and structural templates. It achieved superior atomic-level accuracy in CASP14 and performed well on newly released PDB data. Given a protein sequence, it outputs atomic coordinates and confidence scores for the predicted structure.


<br/>  
# 기타  






논문에는 AlphaFold의 성능과 모델 구조를 시각적으로 설명하는 여러 **테이블, 다이어그램, 그래프**가 포함되어 있다.  

1. **AlphaFold 모델 아키텍처 다이어그램**  
   - **Evoformer 블록과 구조 모듈(Structure Module)의 관계**를 도식화한 그림.  
   - Evoformer가 다중 서열 정렬(MSA)과 아미노산 간 관계(pair representation)를 입력으로 받아 정보를 정제하고, 이후 구조 모듈이 이를 바탕으로 3D 구조를 예측하는 과정을 설명.  
   - 모델 내부에서 재귀적 반복(recycling)을 통해 점진적으로 구조를 정제하는 과정도 시각적으로 나타냄.  

2. **CASP14 성능 비교 테이블**  
   - AlphaFold가 다른 경쟁 모델 대비 **Cα RMSD, TM-score, lDDT 점수에서 압도적인 성능을 보였음**을 보여줌.  
   - **평균 Cα RMSD 값이 0.96Å로 기존 모델(2.8Å)보다 뛰어나며, TM-score도 0.93~0.96 범위로 기록됨**.  
   - AlphaFold의 **예측 신뢰도(Confidence Score)도 높은 수준**을 유지함.  

3. **AlphaFold 예측 vs 실험적으로 결정된 구조 비교 그래프**  
   - CASP14 데이터에서 예측한 단백질 구조와 실험적으로 결정된 구조를 겹쳐서 시각적으로 비교한 그림.  
   - 대부분의 단백질에서 **실험값과 거의 일치하는 구조를 예측**했음을 보여줌.  
   - 일부 어려운 단백질에서도 기존 모델보다 **더 나은 구조 정렬을 보임**.  

4. **AlphaFold 성능 평가 히트맵(Heatmap)**  
   - 다양한 단백질에서 **AlphaFold의 예측 성능을 lDDT 점수 기준으로 색상으로 표현한 그래프**.  
   - 단백질 길이, 구조 복잡도, 동종성(Homology) 여부에 따라 성능 차이를 나타냄.  
   - 대부분의 경우 높은 정확도를 보였지만, **긴 무질서 단백질(disordered proteins)에서는 상대적으로 낮은 신뢰도를 보임**.  

---



The paper includes various **tables, diagrams, and graphs** that visually illustrate AlphaFold's performance and model structure.  

1. **AlphaFold Model Architecture Diagram**  
   - A schematic representation of **the relationship between the Evoformer block and the Structure Module**.  
   - Evoformer **processes multiple sequence alignments (MSA) and pair representations** before the Structure Module predicts the final 3D structure.  
   - The diagram also highlights **the iterative refinement (recycling) process**, which improves predictions progressively.  

2. **CASP14 Performance Comparison Table**  
   - A table showing **AlphaFold's superior performance in Cα RMSD, TM-score, and lDDT metrics** compared to competing models.  
   - **AlphaFold achieves a median Cα RMSD of 0.96Å**, outperforming the next-best model (2.8Å).  
   - It also maintains **high confidence scores** for predictions.  

3. **AlphaFold Predictions vs. Experimental Structures Graph**  
   - A graphical overlay comparing **AlphaFold’s predicted protein structures with experimentally determined structures** from CASP14.  
   - The visual demonstrates that **AlphaFold’s predictions closely match real structures** in most cases.  
   - Even for challenging proteins, **AlphaFold achieves better structural alignment than previous models**.  

4. **AlphaFold Performance Heatmap**  
   - A heatmap depicting **AlphaFold’s prediction accuracy across different proteins using lDDT scores**.  
   - Performance variations are shown based on **protein length, structural complexity, and homology**.  
   - While **AlphaFold performs exceptionally well overall, its confidence is lower for long disordered proteins**.  




<br/>
# refer format:     


@article{Jumper2021AlphaFold,
  author = {Jumper, John and Evans, Richard and Pritzel, Alexander and Green, Tim and Figurnov, Michael and Ronneberger, Olaf and Tunyasuvunakool, Kathryn and Bates, Russ and Žídek, Augustin and Potapenko, Anna and Bridgland, Alex and Meyer, Clemens and Kohl, Simon A. A. and Ballard, Andrew J. and Cowie, Andrew and Romera-Paredes, Bernardino and Nikolov, Stanislav and Jain, Rishub and Adler, Jonas and Back, Trevor and Petersen, Stig and Reiman, David and Clancy, Ellen and Zielinski, Michal and Steinegger, Martin and Pacholska, Michalina and Berghammer, Tamas and Bodenstein, Sebastian and Silver, David and Vinyals, Oriol and Senior, Andrew W. and Kavukcuoglu, Koray and Kohli, Pushmeet and Hassabis, Demis},
  title = {Highly Accurate Protein Structure Prediction with AlphaFold},
  journal = {Nature},
  volume = {596},
  number = {7873},
  pages = {583--589},
  year = {2021},
  doi = {10.1038/s41586-021-03819-2}
}




Jumper, John, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, et al. 2021. “Highly Accurate Protein Structure Prediction with AlphaFold.” Nature 596 (7873): 583–89. https://doi.org/10.1038/s41586-021-03819-2.





