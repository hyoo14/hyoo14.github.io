---
layout: post
title:  "[2024]Streamlining Computational Fragment-Based Drug Discovery through Evolutionary Optimization Informed by Ligand-Based Virtual Prescreening"
date:   2025-11-16 16:16:56 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 FDSL-DD(Fagments from Screened Ligands for Drug Discovery) 방법론을 기반으로 한 두 단계 최적화 방법을 제안하여, 초기 가상 스크리닝 정보를 활용하여 리간드 합성을 최적화합니다.(유전자 알고리즘을 사용하여 조각을 결합, 리간드의 결합 포켓 내에서의 조각 위치와 부모 리간드의 속성을 고려하여 작은 조각을 추가->이를 통한 최적화(QED스코어도 사용) )      


짧은 요약(Abstract) :


이 논문은 최근의 계산 방법들이 약물 발견을 가속화할 수 있는 가능성을 제시하고 있습니다. 특히, 저자들은 "fragment databases from screened ligand drug discovery (FDSL-DD)"라는 새로운 계산 기반의 조각 약물 발견(FBDD) 방법을 발전시켰습니다. 이 방법은 방대한 라이브러리에서 리간드를 식별하고, 이를 조각으로 나눈 후, 예측된 결합 친화도와 타겟 서브도메인과의 상호작용에 기반하여 특정 속성을 부여합니다. 논문에서는 초기 스크리닝 정보를 활용하여 계산 리간드 합성을 최적화하는 두 단계의 최적화 방법을 제안합니다. 이 방법은 검색 공간을 축소하고 유망한 영역에 집중함으로써 후보 리간드의 최적화를 개선할 수 있다고 가정합니다. 첫 번째 단계에서는 유전 알고리즘을 사용하여 조각을 더 큰 화합물로 조합하고, 두 번째 단계에서는 반복적인 정제를 통해 생물활성이 향상된 화합물을 생성합니다. 이 방법은 인간의 고형암, 박테리아 항균 저항성, SARS-CoV-2 바이러스와 같은 세 가지 다양한 단백질 타겟에 적용되어, 제안된 FDSL-DD와 두 단계 최적화 접근 방식이 다른 최신 FBDD 방법보다 더 효율적으로 고친화성 리간드 후보를 생성할 수 있음을 보여줍니다.


This paper presents the potential of recent computational methods to dramatically accelerate drug discovery. The authors build upon a newly developed computational fragment-based drug discovery (FBDD) method called "fragment databases from screened ligand drug discovery (FDSL-DD)." This method utilizes in silico screening to identify ligands from a vast library, fragmenting them while attaching specific attributes based on predicted binding affinity and interaction with the target subdomain. The paper proposes a two-stage optimization method that leverages prescreening information to optimize computational ligand synthesis. It hypothesizes that using prescreening information for optimization shrinks the search space and focuses on promising regions, thereby improving the optimization for candidate ligands. The first stage assembles these fragments into larger compounds using genetic algorithms, followed by a second stage of iterative refinement to produce compounds with enhanced bioactivity. The methodology is demonstrated on three diverse protein targets found in human solid cancers, bacterial antimicrobial resistance, and the SARS-CoV-2 virus, showing that the proposed FDSL-DD and two-stage optimization approach yield high-affinity ligand candidates more efficiently than other state-of-the-art computational FBDD methods.


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



이 논문에서는 약물 발견을 위한 새로운 계산적 방법론인 FDSL-DD(Fragments from Screened Ligands for Drug Discovery)와 이를 기반으로 한 두 단계 최적화 방법을 제안합니다. 이 방법은 초기 가상 스크리닝을 통해 얻은 정보를 활용하여 후보 리간드의 합성을 최적화하는 데 중점을 두고 있습니다.

#### 1. 초기 가상 스크리닝
FDSL-DD는 먼저 대규모 리간드 데이터베이스를 특정 단백질 타겟에 대해 스크리닝합니다. 이 과정에서 AutoDock Vina와 같은 분자 도킹 소프트웨어를 사용하여 리간드의 결합 친화도를 예측합니다. 스크리닝 결과로 얻어진 리간드는 이후 BRICS 알고리즘을 통해 화학적으로 실현 가능한 방식으로 분할됩니다. 각 분할된 조각은 부모 리간드의 결합 친화도와 상호작용하는 아미노산 정보를 기반으로 속성을 부여받습니다.

#### 2. 두 단계 최적화
제안된 두 단계 최적화 방법은 다음과 같이 진행됩니다:

- **1단계: 진화적 최적화**  
  이 단계에서는 유전 알고리즘을 사용하여 초기 리간드 조각들을 조합하여 더 큰 화합물을 생성합니다. 각 조각은 부모 리간드의 속성을 기반으로 선택되며, 이 과정에서 결합 친화도를 최대화하는 방향으로 조정됩니다.

- **2단계: 반복적 최적화**  
  첫 번째 단계에서 생성된 화합물은 반복적으로 개선됩니다. 이 단계에서는 리간드의 결합 포켓 내에서의 조각 위치와 부모 리간드의 속성을 고려하여 작은 조각을 추가하여 생물활성을 향상시키는 방식으로 진행됩니다.

이러한 두 단계의 최적화 과정은 화학 공간을 줄이고, 유망한 영역에 집중함으로써 후보 리간드의 탐색을 더 효율적으로 만듭니다. 최종적으로, 이 방법은 높은 결합 친화도와 약물 유사성을 가진 리간드 후보를 생성하는 데 기여합니다.

#### 3. 다목적 최적화
또한, 이 연구에서는 약물 유사성(QED) 점수를 포함한 다목적 최적화 방법을 제안합니다. 이 방법은 결합 친화도와 약물 유사성을 동시에 고려하여 최적화된 리간드를 생성할 수 있도록 합니다. 이를 통해 생성된 리간드는 높은 결합 친화도를 유지하면서도 약물로서의 특성을 개선할 수 있습니다.




This paper proposes a novel computational methodology for drug discovery called FDSL-DD (Fragments from Screened Ligands for Drug Discovery) and a two-stage optimization approach based on it. This method focuses on optimizing the synthesis of candidate ligands by leveraging information obtained from initial virtual screening.

#### 1. Initial Virtual Screening
FDSL-DD begins with screening a large ligand database against a specific protein target. In this process, molecular docking software such as AutoDock Vina is used to predict the binding affinity of the ligands. The ligands obtained from the screening results are then fragmented using the BRICS algorithm in a chemically feasible manner. Each fragmented piece is assigned attributes based on the binding affinity of the parent ligands and the interacting amino acids.

#### 2. Two-Stage Optimization
The proposed two-stage optimization method proceeds as follows:

- **Stage 1: Evolutionary Optimization**  
  In this stage, a genetic algorithm is employed to combine the initial ligand fragments to generate larger compounds. Each fragment is selected based on the attributes of the parent ligands, and the process is adjusted to maximize binding affinity.

- **Stage 2: Iterative Refinement**  
  The compounds generated in the first stage are iteratively improved. In this stage, small fragments are added to enhance bioactivity, considering the position of the fragments in the binding pocket and the attributes of the parent ligands.

This two-stage optimization process reduces the chemical space and focuses on promising regions, making the search for candidate ligands more efficient. Ultimately, this method contributes to generating high-affinity ligands with desirable drug-like properties.

#### 3. Multi-Objective Optimization
Additionally, the study proposes a multi-objective optimization method that incorporates drug-likeness (QED) scores. This approach allows for the simultaneous consideration of binding affinity and drug-likeness, enabling the generation of optimized ligands. The resulting ligands can maintain high binding affinities while improving their characteristics as drugs.


<br/>
# Results



이 연구에서는 제안된 두 단계 최적화 방법이 기존의 경쟁 모델들과 비교하여 약물 발견의 효율성과 효과성을 어떻게 향상시키는지를 평가하였다. 연구의 주요 결과는 다음과 같다:

1. **경쟁 모델과의 비교**:
   - **AutoGrow4**: 이 모델은 유전 알고리즘을 기반으로 하여 약물 디자인을 수행한다. 연구 결과, AutoGrow4는 TIPE2, RelA, Spike RBD에 대해 각각 -12.6, -11.8, -10.3 kcal/mol의 VINA 점수를 기록하였다. 반면, 제안된 방법은 -14.37, -14.06, -12.49 kcal/mol의 점수를 기록하여 더 높은 결합 친화도를 보였다.
   - **DeepFrag**: 이 딥러닝 기반 모델은 약물 디자인에서 상대적으로 낮은 성능을 보였다. DeepFrag의 최상의 점수는 TIPE2, RelA, Spike RBD에 대해 각각 -12.17, -11.47, -9.895 kcal/mol로, 제안된 방법의 성능에 미치지 못하였다.

2. **테스트 데이터**:
   - 연구에서는 세 가지 서로 다른 단백질 타겟(TIPE2, RelA, Spike RBD)에 대해 최적화된 리간드를 생성하였다. 각 타겟에 대해 다양한 리간드 풀을 사용하여 최적화 과정을 진행하였다.

3. **메트릭**:
   - VINA 점수는 리간드의 결합 친화도를 평가하는 데 사용되었으며, QED 점수는 약물 유사성을 평가하는 데 사용되었다. 제안된 방법은 두 가지 메트릭 모두에서 우수한 성능을 보였다.
   - 다목적 최적화 접근법을 사용하여 QED 점수를 고려한 결과, 결합 친화도와 약물 유사성을 동시에 개선할 수 있었다.

4. **비교 결과**:
   - 제안된 방법은 기존의 AutoGrow4 및 DeepFrag와 비교하여 더 높은 결합 친화도를 기록하였으며, 이는 두 단계 최적화 방법이 효과적으로 작용했음을 나타낸다.
   - 다목적 최적화 접근법을 통해 생성된 리간드는 QED 점수가 개선되었으며, 이는 약물 디자인에서 중요한 요소로 작용한다.

결론적으로, 제안된 두 단계 최적화 방법은 기존의 경쟁 모델들보다 더 높은 결합 친화도를 달성하며, 약물 발견 과정에서의 효율성을 크게 향상시킬 수 있음을 보여주었다.

---




This study evaluated how the proposed two-stage optimization method enhances the efficiency and effectiveness of drug discovery compared to existing competitive models. The main findings are as follows:

1. **Comparison with Competitive Models**:
   - **AutoGrow4**: This model is based on genetic algorithms for drug design. The results showed that AutoGrow4 achieved VINA scores of -12.6, -11.8, and -10.3 kcal/mol for TIPE2, RelA, and Spike RBD, respectively. In contrast, the proposed method recorded scores of -14.37, -14.06, and -12.49 kcal/mol, demonstrating higher binding affinities.
   - **DeepFrag**: This deep learning-based model exhibited relatively poor performance in drug design. The best scores from DeepFrag were -12.17, -11.47, and -9.895 kcal/mol for TIPE2, RelA, and Spike RBD, respectively, which did not match the performance of the proposed method.

2. **Test Data**:
   - The study generated optimized ligands for three distinct protein targets (TIPE2, RelA, Spike RBD). The optimization process was conducted using various ligand pools for each target.

3. **Metrics**:
   - VINA scores were used to evaluate the binding affinity of the ligands, while QED scores assessed drug-likeness. The proposed method demonstrated superior performance in both metrics.
   - The multi-objective optimization approach allowed for improvements in both binding affinity and drug-likeness by considering QED scores.

4. **Comparison Results**:
   - The proposed method achieved higher binding affinities compared to existing models like AutoGrow4 and DeepFrag, indicating the effectiveness of the two-stage optimization approach.
   - Ligands generated through the multi-objective optimization approach showed improved QED scores, which is a crucial factor in drug design.

In conclusion, the proposed two-stage optimization method demonstrates the ability to achieve higher binding affinities than existing competitive models, significantly enhancing the drug discovery process's efficiency.


<br/>
# 예제



이 논문에서는 약물 발견을 위한 새로운 컴퓨터 기반 방법론인 FDSL-DD(Fragments from Screened Ligands for Drug Discovery)와 두 단계 최적화 방법을 제안합니다. 이 방법은 초기 가상 스크리닝을 통해 얻은 정보를 활용하여 후보 리간드를 최적화하는 데 중점을 두고 있습니다. 

#### 1. 트레이닝 데이터와 테스트 데이터
- **트레이닝 데이터**: 초기 단계에서, 연구자들은 Enamine Ltd.의 "Drug-like" 라이브러리에서 약 250,000개의 리간드를 수집합니다. 이 리간드는 AutoDock Vina를 사용하여 특정 단백질 타겟에 대해 가상 스크리닝을 수행하여 각 리간드의 결합 친화도(바인딩 애피니티)를 예측합니다. 이 과정에서 리간드는 결합 친화도에 따라 점수를 부여받고, 이후 이 정보를 바탕으로 리간드를 조각(fragment)으로 나누고 각 조각에 대한 특성을 할당합니다.

- **테스트 데이터**: 최적화된 리간드는 세 가지 서로 다른 단백질 타겟(인간의 고형암, 박테리아의 항균 저항성, SARS-CoV-2 바이러스)에 대해 평가됩니다. 이 단백질 타겟들은 각각의 질병 맥락에서 중요한 역할을 하며, 최적화된 리간드의 결합 친화도를 측정하여 성능을 평가합니다.

#### 2. 구체적인 인풋과 아웃풋
- **인풋**: 
  - 초기 리간드 라이브러리 (약 250,000개)
  - 각 리간드에 대한 가상 스크리닝 결과 (결합 친화도 점수)
  - 조각화된 리간드와 그 특성 (예: 결합 친화도, 상호작용하는 아미노산 정보)

- **아웃풋**: 
  - 최적화된 리간드 후보 (결합 친화도가 높은 리간드)
  - 각 리간드의 최종 결합 친화도 점수
  - 다중 목표 최적화 결과 (결합 친화도와 약물 유사성(QED) 점수를 모두 고려한 결과)

#### 3. 구체적인 테스크
- **테스크 1**: 초기 리간드 라이브러리에서 가상 스크리닝을 통해 결합 친화도를 예측하고, 이를 바탕으로 조각화된 리간드를 생성합니다.
- **테스크 2**: 생성된 조각을 사용하여 유전 알고리즘을 통해 새로운 리간드를 조합하고, 이 리간드의 결합 친화도를 평가합니다.
- **테스크 3**: 최적화된 리간드를 반복적으로 개선하여 최종 후보 리간드를 도출하고, 이 리간드의 약물 유사성을 평가합니다.

이러한 과정을 통해 연구자들은 약물 발견의 초기 단계에서 효율성을 높이고, 더 나은 후보 리간드를 생성할 수 있는 방법을 제시하고 있습니다.

---



This paper proposes a new computational approach for drug discovery called FDSL-DD (Fragments from Screened Ligands for Drug Discovery) and a two-stage optimization method. This method focuses on optimizing candidate ligands using information obtained from initial virtual screening.

#### 1. Training Data and Test Data
- **Training Data**: In the initial phase, researchers collect approximately 250,000 ligands from the Enamine Ltd. "Drug-like" library. These ligands undergo virtual screening against specific protein targets using AutoDock Vina to predict the binding affinities of each ligand. During this process, ligands are assigned scores based on their binding affinities, and this information is used to fragment the ligands and assign attributes to each fragment.

- **Test Data**: The optimized ligands are evaluated against three distinct protein targets (human solid cancers, bacterial antimicrobial resistance, and the SARS-CoV-2 virus). These protein targets play significant roles in their respective disease contexts, and the binding affinities of the optimized ligands are measured to assess performance.

#### 2. Specific Inputs and Outputs
- **Inputs**: 
  - Initial ligand library (approximately 250,000 ligands)
  - Virtual screening results for each ligand (binding affinity scores)
  - Fragmented ligands and their attributes (e.g., binding affinity, interacting amino acid information)

- **Outputs**: 
  - Optimized ligand candidates (ligands with high binding affinities)
  - Final binding affinity scores for each ligand
  - Multi-objective optimization results (considering both binding affinity and drug-likeness (QED) scores)

#### 3. Specific Tasks
- **Task 1**: Perform virtual screening on the initial ligand library to predict binding affinities and generate fragmented ligands based on this information.
- **Task 2**: Use the generated fragments to combine new ligands through a genetic algorithm and evaluate the binding affinities of these ligands.
- **Task 3**: Iteratively improve the optimized ligands to derive final candidate ligands and assess their drug-likeness.

Through these processes, the researchers present a method to enhance efficiency in the early stages of drug discovery and generate better candidate ligands.

<br/>
# 요약


이 논문에서는 FDSL-DD(Fagments from Screened Ligands for Drug Discovery) 방법론을 기반으로 한 두 단계 최적화 방법을 제안하여, 초기 가상 스크리닝 정보를 활용하여 리간드 합성을 최적화합니다. 이 방법은 유전자 알고리즘을 사용하여 조각을 결합하고, 반복적인 최적화를 통해 생물활성 향상된 화합물을 생성하며, 다양한 단백질 표적에 대해 높은 친화력을 가진 리간드를 효율적으로 도출합니다. 결과적으로, 제안된 방법은 기존의 최첨단 FBDD 방법들보다 더 높은 친화력을 가진 후보 리간드를 생성하는 데 성공했습니다.

---

This paper proposes a two-stage optimization method based on the FDSL-DD (Fragments from Screened Ligands for Drug Discovery) approach, utilizing initial virtual screening information to optimize ligand synthesis. The method employs genetic algorithms to assemble fragments and iterative refinement to produce compounds with enhanced bioactivity, efficiently generating high-affinity ligands for diverse protein targets. Ultimately, the proposed method successfully yields candidate ligands with higher affinities compared to other state-of-the-art FBDD methods.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: FDSL-DD와 최적화 방법의 통합된 접근 방식을 보여줍니다. 이 다이어그램은 초기 리간드 라이브러리의 스크리닝, 분할, 그리고 진화 알고리즘을 통한 리간드 조합 과정을 시각적으로 설명합니다. 이 과정은 리간드의 생물학적 활성을 높이는 데 기여합니다.
   - **Figure 2**: 유전 알고리즘을 사용하여 리간드를 생성하는 과정을 보여줍니다. 이 피규어는 초기 세대의 리간드가 어떻게 평가되고, 변이, 교차, 엘리트 선택을 통해 다음 세대로 발전하는지를 설명합니다.
   - **Figure 3**: 반복적인 조각 추가 단계의 개요를 제공합니다. 이 단계에서는 각 리간드가 평가되고, 최적의 결합 친화도를 위해 조각이 추가되는 과정을 보여줍니다.
   - **Figure 4**: 다양한 조각 풀 생성 방법에 따른 결과를 비교하는 히스토그램입니다. 이 그래프는 조각의 품질이 최종 리간드의 결합 친화도에 미치는 영향을 시각적으로 나타냅니다.
   - **Figure 5**: AutoGrow4와 FDSL-DD의 반복 생성 리간드를 비교하는 히스토그램입니다. 이 그래프는 두 방법의 성능 차이를 명확히 보여줍니다.
   - **Figure 6**: DeepFrag의 반복 실행 결과를 보여주는 바 플롯입니다. 이 그래프는 DeepFrag이 생성한 리간드의 결합 친화도가 FDSL-DD 방법보다 낮음을 나타냅니다.
   - **Figure 7**: 다중 목표 최적화와 VINA 우선 순위화의 결과를 비교하는 그래프입니다. 이 그래프는 두 접근 방식의 리간드가 결합 친화도와 약물 유사성에서 어떻게 다른지를 보여줍니다.
   - **Figure 8**: 다중 목표 최적화에서 생성된 리간드의 95번째 백분위수와 QED 점수를 비교하는 히스토그램입니다. 이 그래프는 다중 목표 최적화가 결합 친화도와 약물 유사성을 동시에 개선할 수 있음을 보여줍니다.

2. **테이블**
   - **Table 1**: 각 단백질 타겟에 대한 반복 실행 통계입니다. 이 테이블은 다양한 조각 풀을 사용하여 생성된 리간드의 결합 친화도 통계를 제공합니다. 우선 순위가 매겨진 조각 풀을 사용할 때 결합 친화도가 개선되는 경향을 보여줍니다.
   - **Table 2**: AutoGrow4와 FDSL-DD의 비교 결과를 요약한 테이블입니다. 이 테이블은 두 방법의 리간드 생성 성능을 비교하여 FDSL-DD가 더 높은 결합 친화도를 생성함을 나타냅니다.
   - **Table 3**: 다중 목표 최적화 실행 통계입니다. 이 테이블은 다중 목표 최적화가 결합 친화도와 약물 유사성을 동시에 고려하여 생성한 리간드의 성능을 보여줍니다.

3. **어펜딕스**
   - 어펜딕스에는 후보 리간드의 잠재적 합성 경로, 구조 분석, 그리고 추가 실험 결과가 포함되어 있습니다. 이 정보는 제안된 리간드의 합성 가능성과 생물학적 활성을 평가하는 데 유용합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the integrated approach of FDSL-DD and optimization methods. This diagram visually explains the process of screening, fragmenting, and combining ligands through evolutionary algorithms, contributing to enhancing ligand bioactivity.
   - **Figure 2**: Shows the process of generating ligands using a genetic algorithm. This figure explains how initial generations of ligands are evaluated and evolve through mutation, crossover, and elitism.
   - **Figure 3**: Provides an overview of the iterative fragment addition stage. This stage illustrates how each ligand is evaluated and fragments are added to achieve optimal binding affinity.
   - **Figure 4**: A histogram comparing results from different fragment pool generation methods. This graph visually represents the impact of fragment quality on the final ligand's binding affinity.
   - **Figure 5**: A histogram comparing ligands generated by AutoGrow4 and FDSL-DD. This graph clearly shows the performance differences between the two methods.
   - **Figure 6**: A bar plot showing the results of DeepFrag's iterative runs. This graph indicates that the binding affinities of ligands generated by DeepFrag are lower than those produced by the FDSL-DD method.
   - **Figure 7**: Compares the results of multi-objective optimization and VINA prioritization. This graph shows how the ligands from both approaches differ in binding affinity and drug-likeness.
   - **Figure 8**: A histogram comparing the 95th percentile ligands by VINA score with QED scores from multi-objective optimization. This graph demonstrates that multi-objective optimization can improve both binding affinity and drug-likeness.

2. **Tables**
   - **Table 1**: Statistics from iterative runs for each protein target. This table provides binding affinity statistics for ligands generated using various fragment pools, showing a trend of improved binding affinity when using prioritized fragment pools.
   - **Table 2**: A summary comparing AutoGrow4 and FDSL-DD. This table highlights the performance differences, indicating that FDSL-DD generates ligands with higher binding affinities.
   - **Table 3**: Statistics from multi-objective optimization runs. This table shows the performance of ligands generated by considering both binding affinity and drug-likeness.

3. **Appendices**
   - The appendices include potential synthetic pathways for candidate ligands, structural analyses, and additional experimental results. This information is useful for evaluating the synthetic feasibility and biological activity of the proposed ligands.

<br/>
# refer format:


### BibTeX 형식
```bibtex
@article{Chandraghatgi2024,
  author = {Rohan Chandraghatgi and Hai-Feng Ji and Gail L. Rosen and Bahrad A. Sokhansanj},
  title = {Streamlining Computational Fragment-Based Drug Discovery through Evolutionary Optimization Informed by Ligand-Based Virtual Prescreening},
  journal = {Journal of Chemical Information and Modeling},
  volume = {64},
  number = {10},
  pages = {3826--3840},
  year = {2024},
  doi = {10.1021/acs.jcim.4c00234},
  publisher = {American Chemical Society}
}
```

### 시카고 스타일 인용
Chandraghatgi, Rohan, Hai-Feng Ji, Gail L. Rosen, and Bahrad A. Sokhansanj. "Streamlining Computational Fragment-Based Drug Discovery through Evolutionary Optimization Informed by Ligand-Based Virtual Prescreening." *Journal of Chemical Information and Modeling* 64, no. 10 (2024): 3826-3840. https://doi.org/10.1021/acs.jcim.4c00234.
