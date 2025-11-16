---
layout: post
title:  "[2024]Fragment databases from screened ligands for drug discovery (FDSL-DD)"
date:   2025-11-16 16:13:59 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 FDSL-DD라는 새로운 방법을 제안하여, 이미 도킹된 리간드 라이브러리에서 분자 조각 데이터베이스를 생성하고 이를 통해 약물 후보를 설계하였다.


짧은 요약(Abstract) :


이 논문에서는 약물 발견 과정에서의 새로운 방법인 "스크리닝된 리간드로부터의 조각 데이터베이스 약물 설계(FDSL-DD)"를 제안합니다. 이 방법은 조각의 특성에 대한 정보를 지능적으로 통합하여 약물 개발 과정에서의 조각 기반 설계 접근 방식을 개선합니다. FDSL-DD의 초기 단계는 특정 타겟에 대해 도킹된 약물 같은 리간드의 라이브러리로부터 조각 데이터베이스를 생성하는 것입니다. 이는 전통적인 컴퓨터 기반 조각 약물 설계(FBDD) 전략과는 다르며, 구조 기반 설계 스크리닝 기술을 통합하여 두 접근 방식의 장점을 결합합니다. 이 연구에서는 세 가지 서로 다른 단백질 타겟을 테스트하여 생성된 조각 라이브러리와 FDSL-DD의 잠재력을 입증하였습니다. FDSL-DD를 활용한 결과, 각 단백질 타겟에 대해 결합 친화도가 증가하였으며, TIPE2에 대해 설계된 리간드는 FDSL-DD에서 얻은 최상위 리간드와 고속 가상 스크리닝(HTVS)에서 얻은 최상위 리간드 간의 차이가 3.6 kcal/mol에 달했습니다. 초기 HTVS에서 약물 같은 리간드를 사용함으로써 화학 공간의 탐색이 더 넓어지고, 조각 선택의 효율성이 높아지며, 더 많은 상호작용을 식별할 수 있는 가능성이 커집니다.



This paper presents a new method for drug discovery called "Fragment Databases from Screened Ligands Drug Design" (FDSL-DD). This method intelligently incorporates information about fragment characteristics into a fragment-based design approach to improve the drug development process. The initial step of FDSL-DD is the creation of a fragment database from a library of docked, drug-like ligands for a specific target, which deviates from the traditional in silico fragment-based drug design (FBDD) strategy by integrating structure-based design screening techniques to combine the advantages of both approaches. Three different protein targets were tested in this study to demonstrate the potential of the created fragment library and FDSL-DD. Utilizing FDSL-DD led to an increase in binding affinity for each protein target, with the ligand designed for TIPE2 showing a 3.6 kcal/mol difference between the top ligand from FDSL-DD and the top ligand from high throughput virtual screening (HTVS). Using drug-like ligands in the initial HTVS allows for a greater search of chemical space, higher efficiency in fragment selection, and potentially identifying more interactions.


* Useful sentences(or words) :  

리간드(ligand) 는 표적 단백질의 결합 부위에 붙을 수 있는 작은 분자(small molecule)  

결합친화도(binding affinity)는 리간드가 단백질 결합 부위에 얼마나 잘 붙는지를 수치로 표현한 값인데, 실제 실험 없이 in silico 로 계산할 때는 주로 도킹 점수(docking score) 나 물리 기반 에너지 함수를 이용해서 추정  
여기서는 HTVS + Docking을 통해 얻은 docking score (kcal/mol 단위의 predicted binding energy) 일 것임  




{% endhighlight %}

<br/>

[Paper link]()
[~~Lecture link~~]()

<br/>

# 단어정리
*


<br/>
# Methodology


이 논문에서 제안하는 방법은 "FDSL-DD(Fragments from Screened Ligands Drug Design)"라는 새로운 약물 설계 접근법입니다. 이 방법은 약물 발견 과정에서 조각 기반 설계(fragment-based drug design, FBDD)의 한계를 극복하기 위해 개발되었습니다. FDSL-DD의 주요 단계는 다음과 같습니다.

1. **프래그먼트 데이터베이스 생성**: FDSL-DD의 첫 번째 단계는 특정 단백질 타겟에 대해 이미 도킹된 약물 유사 리간드의 라이브러리에서 프래그먼트 데이터베이스를 생성하는 것입니다. 이 과정은 전통적인 FBDD 전략과는 다르게, 구조 기반 설계(screening) 기술을 통합하여 두 접근법의 장점을 결합합니다.

2. **리간드 스크리닝**: 대량의 "약물 유사" 리간드를 컴퓨터 도킹 소프트웨어를 사용하여 스크리닝합니다. 이 단계에서 각 리간드와 단백질 타겟 간의 예측 결합 친화도(binding affinity)와 리간드가 단백질에 결합하는 방식(화학 결합의 위치 및 종류)을 파악합니다.

3. **프래그먼트화**: 스크리닝 및 프로파일링 단계가 끝난 후, 리간드는 컴퓨터적으로 "프래그먼트화"됩니다. 이 과정에서 각 리간드의 프래그먼트에 대한 통계와 결합 친화도 정보를 포함하는 데이터베이스가 생성됩니다.

4. **프래그먼트 조합**: 생성된 프래그먼트 데이터베이스를 사용하여 새로운 약물 후보를 설계합니다. 이 과정에서는 프래그먼트를 결합하여 새로운 리간드를 생성하고, 이 리간드의 결합 친화도를 평가합니다.

5. **ADME 및 약물 유사성 평가**: 생성된 리간드는 약물의 흡수, 분포, 대사 및 배설(ADME) 특성과 약물 유사성(Lipinski의 규칙 등)을 평가하여 최적의 후보를 선택합니다.

이 방법은 세 가지 단백질 타겟(TIPE2, RelA, SARS-CoV-2 스파이크 단백질)에 대해 테스트되었으며, FDSL-DD를 사용하여 각 단백질 타겟에 대한 결합 친화도가 증가하는 결과를 보여주었습니다. 특히, TIPE2에 대한 리간드는 HTVS(high throughput virtual screening)에서 얻은 리간드보다 3.6 kcal/mol 더 높은 결합 친화도를 나타냈습니다.




The method proposed in this paper is a new drug design approach called "FDSL-DD (Fragments from Screened Ligands Drug Design)." This method was developed to overcome the limitations of fragment-based drug design (FBDD) in the drug discovery process. The main steps of FDSL-DD are as follows:

1. **Fragment Database Creation**: The first step of FDSL-DD is to create a fragment database from a library of already docked, drug-like ligands for a specific protein target. This process deviates from traditional FBDD strategies by incorporating structure-based design screening techniques to combine the advantages of both approaches.

2. **Ligand Screening**: A large number of "drug-like" ligands are screened using computational docking software. In this step, the predicted binding affinity between each ligand and the protein target is obtained, along with information on how the ligand binds to the protein (the location and type of chemical bonds).

3. **Fragmentation**: After the screening and profiling steps, the ligands are computationally "fragmented." This process generates a database that includes statistics for each fragment and binding affinity information from the parent ligands.

4. **Fragment Combination**: The resulting fragment database is utilized to design new drug candidates. In this process, fragments are combined to generate new ligands, and the binding affinity of these ligands is evaluated.

5. **ADME and Drug-likeness Assessment**: The generated ligands are evaluated for their absorption, distribution, metabolism, and excretion (ADME) properties, as well as drug-likeness (such as Lipinski's rule), to select the optimal candidates.

This method was tested on three different protein targets (TIPE2, RelA, and the SARS-CoV-2 spike protein) and demonstrated an increase in binding affinity for each protein target when using FDSL-DD. Notably, the ligand designed for TIPE2 exhibited a binding affinity that was 3.6 kcal/mol higher than the ligand obtained from high throughput virtual screening (HTVS).


<br/>
# Results



이 연구에서는 FDSL-DD(Fragments from Screened Ligands Drug Design) 방법을 통해 세 가지 단백질 타겟(TIPE2, RelA, S-단백질)에 대한 약물 후보 물질을 설계하고, 이들 각각에 대해 생성된 조각 데이터베이스를 활용하여 결합 친화도를 향상시켰습니다. 

1. **경쟁 모델**: FDSL-DD 방법은 기존의 고속 가상 스크리닝(HTVS) 방법과 비교되었습니다. HTVS는 Enamine 라이브러리에서 약 250,000개의 약물 유사 화합물을 스크리닝하여 최적의 리간드를 찾는 전통적인 방법입니다. FDSL-DD는 이와 달리 이미 도킹된 리간드 라이브러리에서 조각을 생성하여 새로운 리간드를 설계하는 접근법입니다.

2. **테스트 데이터**: 연구에서는 TIPE2, RelA, S-단백질의 세 가지 단백질 타겟을 선택하여 각각에 대해 FDSL-DD 방법을 적용했습니다. 각 단백질에 대해 생성된 조각 데이터베이스에서 가장 높은 결합 친화도를 가진 리간드를 비교했습니다.

3. **메트릭**: 결합 친화도는 kcal/mol 단위로 측정되었으며, FDSL-DD 방법으로 설계된 리간드는 HTVS 방법으로 설계된 리간드에 비해 평균적으로 더 높은 결합 친화도를 보였습니다. 예를 들어, TIPE2에 대한 FDSL-DD에서 설계된 리간드는 HTVS에서 설계된 리간드보다 3.6 kcal/mol 더 높은 결합 친화도를 나타냈습니다.

4. **비교**: 각 단백질에 대해 FDSL-DD 방법으로 설계된 리간드와 HTVS 방법으로 설계된 리간드의 결합 친화도를 비교한 결과, 모든 단백질 타겟에서 FDSL-DD 방법이 더 높은 결합 친화도를 기록했습니다. TIPE2의 경우, FDSL-DD에서 설계된 리간드는 -13.4 kcal/mol의 결합 친화도를 보였고, RelA는 -12.4 kcal/mol, S-단백질은 -12.0 kcal/mol로 나타났습니다. 반면, HTVS 방법으로 설계된 리간드는 각각 -9.8 kcal/mol, -10.2 kcal/mol, -9.1 kcal/mol의 결합 친화도를 보였습니다.

이러한 결과는 FDSL-DD 방법이 기존의 HTVS 방법보다 더 효과적으로 약물 후보 물질을 설계할 수 있음을 보여줍니다. 또한, FDSL-DD 방법은 조각 데이터베이스를 활용하여 약물 개발 과정에서의 시간과 자원을 절약할 수 있는 가능성을 제시합니다.

---



In this study, the FDSL-DD (Fragments from Screened Ligands Drug Design) method was employed to design drug candidates for three protein targets (TIPE2, RelA, and S-protein), utilizing the generated fragment database to enhance binding affinity for each target.

1. **Competing Models**: The FDSL-DD method was compared with the traditional high-throughput virtual screening (HTVS) approach. HTVS screens approximately 250,000 drug-like compounds from the Enamine library to identify optimal ligands. In contrast, FDSL-DD generates fragments from an already docked ligand library to design new ligands.

2. **Test Data**: The study selected three protein targets (TIPE2, RelA, and S-protein) and applied the FDSL-DD method to each. The highest binding affinity ligands generated from the fragment database were compared against those obtained from HTVS.

3. **Metrics**: Binding affinity was measured in kcal/mol, and ligands designed using the FDSL-DD method exhibited higher binding affinities on average compared to those from the HTVS method. For instance, the ligand designed for TIPE2 using FDSL-DD showed a binding affinity that was 3.6 kcal/mol higher than the ligand from HTVS.

4. **Comparison**: The binding affinities of ligands designed for each protein using FDSL-DD were compared to those from HTVS. In all cases, FDSL-DD demonstrated superior binding affinities. For TIPE2, the FDSL-DD ligand had a binding affinity of -13.4 kcal/mol, while the HTVS ligand had -9.8 kcal/mol. For RelA, the FDSL-DD ligand showed -12.4 kcal/mol compared to -10.2 kcal/mol from HTVS, and for S-protein, the FDSL-DD ligand had -12.0 kcal/mol versus -9.1 kcal/mol from HTVS.

These results indicate that the FDSL-DD method is more effective in designing drug candidates compared to the traditional HTVS approach. Furthermore, the FDSL-DD method presents the potential to save time and resources in the drug development process by leveraging the fragment database.


<br/>
# 예제


이 논문에서는 새로운 약물 설계 방법인 FDSL-DD(Fragments from Screened Ligands Drug Design)를 제안하고 있습니다. 이 방법은 약물 발견 과정에서 조각의 특성 정보를 통합하여 약물 후보를 설계하는 데 도움을 줍니다. FDSL-DD의 초기 단계는 특정 타겟에 대해 도킹된 약물 같은 리간드의 라이브러리에서 조각 데이터베이스를 생성하는 것입니다. 이 과정은 전통적인 FBDD(조각 기반 약물 설계) 전략과는 다르게, 구조 기반 설계 스크리닝 기술을 통합하여 두 접근 방식의 장점을 결합합니다.

#### 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋

1. **트레이닝 데이터**:
   - **인풋**: 
     - 도킹된 리간드의 PDBQT 파일 (리간드 구조와 예측된 결합 친화도 포함)
     - 리간드와 단백질 간의 결합 상호작용 데이터 (PLIP를 통해 생성)
   - **아웃풋**: 
     - 각 조각에 대한 통계 데이터 (예: 평균 결합 친화도, 결합된 아미노산 잔기 정보)
     - 조각-서브리전 조합에 대한 데이터베이스 생성

2. **테스트 데이터**:
   - **인풋**: 
     - 새로운 리간드 후보의 PDBQT 파일
     - FDSL-DD에서 생성된 조각 데이터베이스
   - **아웃풋**: 
     - 새로운 리간드의 예측된 결합 친화도
     - ADME(흡수, 분포, 대사, 배설) 특성 및 약물 유사성 평가 결과

#### 구체적인 테스크
- **조각 데이터베이스 생성**: 도킹된 리간드에서 조각을 추출하고, 각 조각의 결합 친화도 통계 및 결합된 아미노산 정보를 포함한 데이터베이스를 생성합니다.
- **리간드 후보 설계**: 생성된 조각 데이터베이스를 사용하여 새로운 리간드 후보를 조합하고, 이들의 결합 친화도를 예측합니다.
- **약물 유사성 평가**: 생성된 리간드 후보의 ADME 특성과 약물 유사성을 평가하여 최적의 후보를 선택합니다.




This paper presents a new drug design method called FDSL-DD (Fragments from Screened Ligands Drug Design). This method helps in designing drug candidates by intelligently incorporating information about fragment characteristics into the drug discovery process. The initial step of FDSL-DD involves creating a fragment database from a library of docked, drug-like ligands for a specific target. This process deviates from traditional FBDD (Fragment-Based Drug Design) strategies by integrating structure-based design screening techniques to combine the advantages of both approaches.

#### Specific Inputs and Outputs of Training and Test Data

1. **Training Data**:
   - **Input**: 
     - PDBQT files of docked ligands (including ligand structures and predicted binding affinities)
     - Binding interaction data between ligands and proteins (generated through PLIP)
   - **Output**: 
     - Statistical data for each fragment (e.g., mean binding affinity, information on bound amino acid residues)
     - Creation of a database for fragment-subregion combinations

2. **Test Data**:
   - **Input**: 
     - PDBQT files of new ligand candidates
     - Fragment database generated from FDSL-DD
   - **Output**: 
     - Predicted binding affinities of new ligands
     - Results of ADME (Absorption, Distribution, Metabolism, Excretion) properties and drug-likeness evaluation

#### Specific Tasks
- **Fragment Database Creation**: Extract fragments from docked ligands and create a database that includes binding affinity statistics and binding amino acid information for each fragment.
- **Ligand Candidate Design**: Use the generated fragment database to combine new ligand candidates and predict their binding affinities.
- **Drug-likeness Evaluation**: Assess the ADME properties and drug-likeness of the generated ligand candidates to select the optimal candidates.

<br/>
# 요약


이 논문에서는 FDSL-DD라는 새로운 방법을 제안하여, 이미 도킹된 리간드 라이브러리에서 분자 조각 데이터베이스를 생성하고 이를 통해 약물 후보를 설계하였다. 세 가지 단백질 목표에 대해 실험한 결과, FDSL-DD를 사용하여 각 단백질에 대한 결합 친화도가 증가하였으며, 특히 TIPE2에 대한 리간드는 3.6 kcal/mol의 차이를 보였다. 이 방법은 약물 발견 과정에서 시간과 효율성을 개선하는 데 기여할 것으로 기대된다.

---

In this paper, a novel method called FDSL-DD is proposed, which creates a fragment database from an already docked ligand library to design drug candidates. Experiments on three different protein targets showed that using FDSL-DD increased binding affinity for each target, with the ligand for TIPE2 showing a difference of 3.6 kcal/mol. This method is expected to contribute to improving time efficiency in the drug discovery process.

<br/>
# 기타




#### 다이어그램 및 피규어
1. **Scheme 1**: FDSL-DD 방법의 개요를 보여줍니다. 이 다이어그램은 초기 단계에서 약물-단백질 결합 친화도를 예측하고, 리간드를 조각으로 나누어 데이터베이스를 생성하는 과정을 설명합니다. 이 방법은 기존의 FBDD 접근 방식과 구조 기반 설계 기법을 결합하여 더 나은 약물 후보를 발굴할 수 있는 가능성을 보여줍니다.

2. **Fig. 1**: FDSL-DD 흐름도는 리간드 스크리닝의 출력 파일을 입력으로 사용하여 단백질-리간드 상호작용을 예측하고, 이를 통해 조각을 생성하는 과정을 시각적으로 나타냅니다. 이 과정은 리간드의 결합 친화도를 기반으로 조각을 정렬하고 새로운 리간드를 생성하는 방법을 설명합니다.

3. **Fig. 2**: 각 단백질에 대해 가장 자주 발생하는 조각을 보여줍니다. TIPE2, RelA, S-단백질에 대한 조각의 빈도와 평균 결합 친화도를 나타내며, 이는 특정 단백질에 대한 약물 설계에서 유용한 정보를 제공합니다.

4. **Fig. 3**: HTVS 방법과 FDSL-DD 방법으로 생성된 상위 3개의 리간드 구조를 비교합니다. FDSL-DD 방법으로 생성된 리간드가 HTVS 방법으로 생성된 리간드보다 더 높은 결합 친화도를 보이는 것을 확인할 수 있습니다.

5. **Fig. 4**: 스크리닝된 리간드와 FDSL-DD 리간드 간의 상호작용을 3D 및 2D로 나타냅니다. 이 그림은 두 리간드가 단백질의 결합 부위에서 어떻게 상호작용하는지를 보여주며, 결합 친화도의 차이를 설명하는 데 도움을 줍니다.

#### 테이블
1. **Table 1**: 각 단백질에 대한 상위 10%의 리간드에서 가장 자주 발생하는 조각을 나열합니다. 각 조각의 평균 결합 친화도와 발생 빈도를 제공하여, 특정 조각이 약물 설계에서 얼마나 중요한지를 보여줍니다.

2. **Table 2**: HTVS 방법과 FDSL-DD 방법으로 생성된 리간드의 결합 친화도를 비교합니다. FDSL-DD 방법으로 생성된 리간드가 HTVS 방법으로 생성된 리간드보다 높은 결합 친화도를 보이는 것을 확인할 수 있습니다.

3. **Table 3**: 약물 유사성을 비교하는 테이블로, Lipinski, Ghose, Veber, Egan, Muegge 필터를 기준으로 리간드의 약물 유사성을 평가합니다. FDSL-DD 방법으로 생성된 리간드가 여러 필터를 통과하지 못하는 경우가 있지만, 이는 향후 개선의 여지를 나타냅니다.






#### Diagrams and Figures
1. **Scheme 1**: This diagram outlines the FDSL-DD method. It illustrates the initial step of predicting drug-protein binding affinity and fragmenting ligands to create a database. This method combines traditional FBDD approaches with structure-based design techniques, showcasing the potential for discovering better drug candidates.

2. **Fig. 1**: The flow diagram of FDSL-DD visually represents the process of using output files from ligand screening as input to predict protein-ligand interactions and generate fragments. It explains how fragments are sorted based on binding affinity to create new ligands.

3. **Fig. 2**: This figure shows the most frequently occurring fragments for each protein. It provides the frequency and average binding affinity of fragments for TIPE2, RelA, and the S-protein, offering valuable insights for drug design targeting specific proteins.

4. **Fig. 3**: It compares the top three ligand structures generated by the HTVS method and the FDSL-DD method. The figure confirms that ligands generated by the FDSL-DD method exhibit higher binding affinities than those produced by the HTVS method.

5. **Fig. 4**: This figure presents the interactions between screened ligands and FDSL-DD ligands in both 3D and 2D formats. It helps explain the differences in binding affinities by showing how the two ligands interact within the protein's binding site.

#### Tables
1. **Table 1**: Lists the most frequently occurring fragments in the top 10% of ligands for each protein, providing average binding affinities and occurrence counts. This highlights the significance of specific fragments in drug design.

2. **Table 2**: Compares the binding affinities of ligands generated by the HTVS method and the FDSL-DD method. It shows that ligands produced by the FDSL-DD method have higher binding affinities than those from the HTVS method.

3. **Table 3**: A comparison of drug-likeness based on various filters (Lipinski, Ghose, Veber, Egan, Muegge). It indicates that while some ligands generated by the FDSL-DD method do not pass several filters, this suggests room for future improvements.

<br/>
# refer format:




### BibTeX 형식
```bibtex
@article{Wilson2024,
  author = {Jerica Wilson and Bahrad A. Sokhansanj and Wei Chuen Chong and Rohan Chandraghatgi and Gail L. Rosen and Hai-Feng Jia},
  title = {Fragment databases from screened ligands for drug discovery (FDSL-DD)},
  journal = {Journal of Molecular Graphics and Modelling},
  volume = {127},
  pages = {108669},
  year = {2024},
  publisher = {Elsevier Inc.},
  doi = {10.1016/j.jmgm.2023.108669},
  url = {https://doi.org/10.1016/j.jmgm.2023.108669},
  note = {Available online 22 November 2023}
}
```

### 시카고 스타일 인용
Wilson, Jerica, Bahrad A. Sokhansanj, Wei Chuen Chong, Rohan Chandraghatgi, Gail L. Rosen, and Hai-Feng Jia. "Fragment Databases from Screened Ligands for Drug Discovery (FDSL-DD)." *Journal of Molecular Graphics and Modelling* 127 (2024): 108669. https://doi.org/10.1016/j.jmgm.2023.108669.
