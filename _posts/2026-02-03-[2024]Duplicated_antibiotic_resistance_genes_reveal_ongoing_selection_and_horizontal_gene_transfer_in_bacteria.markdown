---
layout: post
title:  "[2024]Duplicated antibiotic resistance genes reveal ongoing selection and horizontal gene transfer in bacteria"
date:   2026-02-03 18:59:31 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 연구에서는 수학적 모델링과 실험적 진화를 통해 항생제 선택이 항생제 내성 유전자(ARG)의 복제를 유도할 수 있음을 보여주었다.


짧은 요약(Abstract) :


이 연구에서는 수직 유전자 전달과 수평 유전자 전달(HGT) 및 유전자 중복이 박테리아의 항생제 내성 유전자(ARG) 진화에 미치는 영향을 조사합니다. 연구팀은 수학적 모델링과 실험적 진화를 결합하여 항생제 선택이 HGT에 의해 유도된 유전자 중복을 통해 항생제 내성 유전자의 진화를 촉진할 수 있음을 보여주었습니다. 연구 결과, 인간과 가축에서 분리된 박테리아에서 중복된 ARG가 높은 비율로 발견되었으며, 이는 항생제 사용과 관련이 있습니다. 또한, 중복된 ARG는 임상 항생제 내성 균주에서도 더욱 풍부하게 나타났습니다. 이 연구는 중복된 유전자가 미생물 군집 내에서 긍정적인 선택과 수평 유전자 전달을 겪고 있음을 시사합니다.



This study investigates the impact of vertical gene transfer, horizontal gene transfer (HGT), and gene duplication on the evolution of antibiotic resistance genes (ARGs) in bacteria. The research team combines mathematical modeling and experimental evolution to demonstrate that antibiotic selection can drive the evolution of duplicated ARGs through HGT. The findings reveal that duplicated ARGs are highly prevalent in bacteria isolated from humans and livestock, correlating with antibiotic use. Additionally, duplicated ARGs are further enriched in clinical antibiotic-resistant isolates. This study suggests that duplicated genes often undergo positive selection and horizontal gene transfer within microbial communities.


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



이 논문에서는 항생제 내성 유전자(ARG)의 중복 진화 메커니즘을 이해하기 위해 수학적 모델링, 실험적 진화, 그리고 생물정보학적 분석을 결합한 방법론을 사용했습니다. 연구의 주요 목표는 수직적 유전자 복제와 수평적 유전자 전이(HGT)가 어떻게 상호작용하여 ARG의 중복을 촉진하는지를 규명하는 것이었습니다.

#### 1. 수학적 모델링
연구자들은 세 가지 하위 집단의 박테리아를 포함하는 수학적 모델을 구축했습니다:
- **Type 1**: 염색체에 ARG를 가진 박테리아
- **Type 2**: 염색체에 중복된 ARG를 가진 박테리아
- **Type 3**: 플라스미드에 중복된 ARG를 가진 박테리아

모델은 항생제 농도, ARG 발현 비용, 그리고 각 하위 집단의 ARG 복사 수에 따라 박테리아의 적합성을 결정하는 함수로 구성되었습니다. 이 모델은 항생제 선택이 중복 ARG의 진화를 어떻게 촉진하는지를 설명합니다. 특히, 항생제 농도가 특정 임계값을 초과할 때 중복 ARG를 가진 박테리아가 다른 하위 집단을 빠르게 대체하는 경향이 있음을 보여주었습니다.

#### 2. 실험적 진화
E. coli 균주를 사용하여 실험적 진화를 수행했습니다. 연구자들은 최소한의 전이 요소를 포함한 플라스미드를 사용하여 항생제 선택 압력을 가했습니다. 9일간의 선택 실험에서, 항생제 농도를 점진적으로 증가시키며 중복 ARG의 발생을 관찰했습니다. 이 실험을 통해 항생제 선택이 중복 ARG의 진화를 유도한다는 것을 확인했습니다.

#### 3. 생물정보학적 분석
18,938개의 완전한 박테리아 유전체를 분석하여 중복 ARG의 분포를 조사했습니다. 이 과정에서 각 유전체의 생태적 메타데이터를 활용하여 인간과 가축에서 분리된 박테리아에서 중복 ARG가 유의미하게 증가한다는 결과를 도출했습니다. 또한, 임상 항생제 내성 균주에서도 중복 ARG가 풍부하다는 것을 확인했습니다.

#### 4. 결론
이 연구는 MGEs(이동 유전자 요소)가 ARG의 중복을 촉진하는 강력한 요인임을 보여주며, 항생제 사용이 내성 하위 집단을 풍부하게 할 뿐만 아니라 진화적 혁신을 위한 중복 유전자를 선택한다는 것을 시사합니다. 이러한 결과는 박테리아의 유전자 복제와 수평적 유전자 전이가 서로 연결되어 있음을 강조합니다.

---




In this paper, the authors employed a combination of mathematical modeling, experimental evolution, and bioinformatics analysis to understand the mechanisms of antibiotic resistance gene (ARG) duplication. The primary goal of the study was to elucidate how vertical gene duplication and horizontal gene transfer (HGT) interact to promote the duplication of ARGs.

#### 1. Mathematical Modeling
The researchers constructed a mathematical model involving three subpopulations of bacteria:
- **Type 1**: Bacteria carrying an ARG on the chromosome
- **Type 2**: Bacteria with a duplicated ARG on the chromosome
- **Type 3**: Bacteria with a duplicated ARG on a plasmid

The model was structured to determine the fitness of each subpopulation based on antibiotic concentration, the cost of ARG expression, and the effective number of ARG copies per cell. This model illustrates how antibiotic selection can drive the evolution of duplicated ARGs. Notably, it showed that when antibiotic concentrations exceed a certain threshold, bacteria with duplicated ARGs rapidly outcompete other subpopulations.

#### 2. Experimental Evolution
Experimental evolution was conducted using E. coli strains. The researchers employed a minimal transposon containing ARGs and applied antibiotic selection pressure over a period of nine days, gradually increasing the antibiotic concentration. This experiment confirmed that antibiotic selection drives the evolution of duplicated ARGs.

#### 3. Bioinformatics Analysis
The distribution of duplicated ARGs was examined across 18,938 complete bacterial genomes. The analysis utilized ecological metadata to demonstrate that duplicated ARGs are significantly enriched in bacteria isolated from humans and livestock. Additionally, it was confirmed that duplicated ARGs are abundant in clinical antibiotic-resistant isolates.

#### 4. Conclusion
This study demonstrates that mobile genetic elements (MGEs) serve as potent drivers of ARG duplication and suggests that antibiotic use not only enriches resistant subpopulations but also selects for mutants with a higher capacity for evolutionary innovation through gene duplication. These findings emphasize the interconnectedness of gene duplication and horizontal gene transfer in bacteria.


<br/>
# Results



이 연구에서는 항생제 선택이 항생제 내성 유전자(ARG)의 중복 진화를 어떻게 촉진하는지를 이해하기 위해 수학적 모델링, 실험적 진화 및 생물정보학적 분석을 결합했습니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구자들은 세 가지 하위 집단의 박테리아를 포함하는 수학적 모델을 구축했습니다. 첫 번째 집단은 염색체에 ARG를 가진 박테리아(Type 1), 두 번째 집단은 염색체에 중복된 ARG를 가진 박테리아(Type 2), 세 번째 집단은 플라스미드에 중복된 ARG를 가진 박테리아(Type 3)입니다. 이 모델은 항생제 농도, ARG 발현 비용, ARG 복사 수에 따라 각 집단의 적합성을 평가했습니다. 모델 결과는 항생제 선택이 중복된 ARG를 가진 집단이 다른 집단을 빠르게 대체하도록 유도함을 보여주었습니다.

2. **실험적 데이터**: E. coli를 사용한 실험에서, 연구자들은 항생제 선택이 중복된 ARG의 진화를 유도한다는 가설을 테스트했습니다. 9일간의 선택 실험에서, 항생제에 노출된 집단에서 중복된 ARG의 복사 수가 증가하는 것을 관찰했습니다. 반면, 항생제가 없는 대조군에서는 중복이 관찰되지 않았습니다. 이러한 결과는 항생제 치료가 중복된 ARG의 진화를 직접적으로 선택한다는 것을 시사합니다.

3. **메트릭 및 비교**: 연구자들은 18,938개의 완전한 박테리아 유전체를 분석하여 중복된 ARG의 분포를 조사했습니다. 인간과 가축에서 분리된 박테리아에서 중복된 ARG의 비율이 유의미하게 높았으며, 이는 항생제 사용과 관련이 있음을 나타냅니다. 또한, 321개의 임상 항생제 내성 균주에서도 중복된 ARG가 풍부하게 발견되었습니다. 이 결과는 중복된 ARG가 항생제 사용과 관련된 환경에서 선택 압력을 받는다는 것을 강조합니다.

4. **결론**: 연구 결과는 중복된 ARG가 항생제 선택과 수평 유전자 전이의 결과로 발생하며, 이는 미생물 군집 내에서 긍정적인 선택을 받고 있음을 나타냅니다. 이러한 발견은 항생제 사용이 내성 집단을 풍부하게 할 뿐만 아니라, 진화적 혁신을 위한 더 높은 능력을 가진 돌연변이를 선택한다는 것을 시사합니다.




This study combined mathematical modeling, experimental evolution, and bioinformatics analysis to understand how antibiotic selection drives the evolution of antibiotic resistance genes (ARGs). The main results are as follows:

1. **Competition Model**: The researchers built a mathematical model involving three subpopulations of bacteria: the first carrying an ARG on the chromosome (Type 1), the second having a duplicated ARG on the chromosome (Type 2), and the third carrying a duplicated ARG on a plasmid (Type 3). This model assessed the fitness of each subpopulation based on antibiotic concentration, the cost of ARG expression, and the effective number of ARG copies per cell. The model results indicated that antibiotic selection favored the subpopulation with duplicated ARGs, allowing it to rapidly outcompete the others.

2. **Experimental Data**: In experiments using E. coli, the researchers tested the hypothesis that antibiotic selection drives the evolution of duplicated ARGs. In a 9-day selection experiment, they observed an increase in the copy number of duplicated ARGs in populations exposed to antibiotics. In contrast, no duplications were observed in the control populations without antibiotics. These findings suggest that antibiotic treatment directly selects for the observed duplications of ARGs.

3. **Metrics and Comparisons**: The researchers analyzed 18,938 complete bacterial genomes to investigate the distribution of duplicated ARGs. They found significantly higher proportions of duplicated ARGs in bacteria isolated from humans and livestock, indicating a correlation with antibiotic use. Additionally, duplicated ARGs were enriched in an independent set of 321 clinical antibiotic-resistant isolates. This result emphasizes that duplicated ARGs are under selection pressure in environments associated with antibiotic use.

4. **Conclusion**: The findings indicate that duplicated ARGs arise as a result of antibiotic selection and horizontal gene transfer, reflecting ongoing positive selection in microbial communities. This suggests that antibiotic use not only enriches resistant subpopulations but also selects for mutants with a higher capacity for evolutionary innovation through gene duplication.


<br/>
# 예제



이 논문에서는 항생제 내성 유전자(ARG)의 중복이 수평 유전자 전이(HGT)와 관련이 있다는 가설을 검증하기 위해 수학적 모델링, 실험적 진화 및 생물정보학적 분석을 사용했습니다. 연구의 주요 목표는 항생제 선택이 어떻게 중복된 ARG의 진화를 촉진하는지를 이해하는 것이었습니다.

#### 1. 데이터 수집
- **트레이닝 데이터**: 18,938개의 완전한 박테리아 유전체를 NCBI RefSeq에서 다운로드하고, 이들 유전체의 생태학적 메타데이터를 수집했습니다. 이 데이터는 다양한 환경(인간, 가축 등)에서 분리된 박테리아를 포함합니다.
- **테스트 데이터**: 321개의 임상 항생제 내성 균주 유전체를 포함한 데이터셋을 수집했습니다. 이 데이터는 고품질의 공개된 유전체로, 항생제 내성과 관련된 유전자 중복을 검증하는 데 사용되었습니다.

#### 2. 모델링 및 실험
- **수학적 모델**: 세 가지 박테리아 아형(단일 복사 ARG, 중복 복사 ARG, 플라스미드에 있는 중복 복사 ARG)을 포함하는 모델을 구축했습니다. 이 모델은 항생제 농도, ARG 발현 비용, 복사 수에 따른 적합도를 고려하여 중복 ARG의 확산을 예측합니다.
- **실험적 진화**: E. coli 균주를 사용하여 항생제 선택이 중복 ARG의 진화를 어떻게 촉진하는지를 실험적으로 검증했습니다. 항생제 농도를 점진적으로 증가시키며 9일간 배양한 후, 유전체를 시퀀싱하여 변이를 분석했습니다.

#### 3. 결과 분석
- **중복 ARG의 분포**: 생태학적 카테고리별로 중복 ARG의 비율을 계산하고, 인간 및 가축에서 분리된 박테리아에서 중복 ARG가 유의미하게 높은 비율로 발견됨을 확인했습니다.
- **임상 균주 분석**: 321개의 임상 항생제 내성 균주에서 중복 ARG의 존재를 확인하여, 임상 환경에서의 중복 ARG의 중요성을 강조했습니다.

이 연구는 항생제 사용이 중복 ARG의 진화를 촉진하고, 이러한 중복이 수평 유전자 전이와 관련이 있음을 보여줍니다.

---




This paper investigates the hypothesis that duplicated antibiotic resistance genes (ARGs) are related to horizontal gene transfer (HGT) by employing mathematical modeling, experimental evolution, and bioinformatics analyses. The main goal of the study is to understand how antibiotic selection drives the evolution of duplicated ARGs.

#### 1. Data Collection
- **Training Data**: A total of 18,938 complete bacterial genomes were downloaded from NCBI RefSeq, along with ecological metadata for these genomes. This data includes bacteria isolated from various environments (humans, livestock, etc.).
- **Test Data**: A dataset of 321 clinical antibiotic-resistant isolates was collected. This dataset consists of high-quality, publicly available genomes used to validate the presence of duplicated ARGs.

#### 2. Modeling and Experiments
- **Mathematical Model**: A model was constructed involving three subpopulations of bacteria (single-copy ARG, duplicated ARG, and duplicated ARG on plasmids). This model predicts the spread of duplicated ARGs based on antibiotic concentration, ARG expression costs, and copy number.
- **Experimental Evolution**: E. coli strains were used to experimentally validate how antibiotic selection promotes the evolution of duplicated ARGs. The strains were cultured for 9 days with gradually increasing antibiotic concentrations, followed by genome sequencing to analyze mutations.

#### 3. Results Analysis
- **Distribution of Duplicated ARGs**: The proportion of duplicated ARGs was calculated across ecological categories, confirming that bacteria isolated from humans and livestock had significantly higher rates of duplicated ARGs.
- **Clinical Isolate Analysis**: The presence of duplicated ARGs was confirmed in the 321 clinical antibiotic-resistant isolates, emphasizing the importance of duplicated ARGs in clinical settings.

This study demonstrates that antibiotic use promotes the evolution of duplicated ARGs and that these duplications are associated with horizontal gene transfer.

<br/>
# 요약

이 연구에서는 수학적 모델링과 실험적 진화를 통해 항생제 선택이 항생제 내성 유전자(ARG)의 복제를 유도할 수 있음을 보여주었다. 18,938개의 완전한 박테리아 유전체 분석 결과, 인간과 가축에서 분리된 박테리아에서 복제된 ARG가 유의미하게 증가한 것으로 나타났다. 이 연구는 복제된 ARG가 수평 유전자 전이와 긍정적 선택을 통해 미생물 군집에서 진화하는 중요한 메커니즘임을 강조한다.

---

This study demonstrated that antibiotic selection can drive the duplication of antibiotic resistance genes (ARGs) through mathematical modeling and experimental evolution. Analysis of 18,938 complete bacterial genomes revealed a significant increase in duplicated ARGs in bacteria isolated from humans and livestock. The findings highlight that duplicated ARGs are an important mechanism for evolution through horizontal gene transfer and positive selection in microbial communities.

<br/>
# 기타



### 1. 다이어그램
- **모델 다이어그램 (Fig. 1A)**: 이 다이어그램은 세 가지 세포 집단을 보여줍니다: (1) 염색체에 ARG를 가진 세포, (2) 염색체에 중복된 ARG를 가진 세포, (3) 플라스미드에 중복된 ARG를 가진 세포. 이 모델은 항생제 선택이 중복된 ARG의 진화를 어떻게 촉진하는지를 설명합니다.

### 2. 피규어
- **항생제 선택에 따른 중복 ARG의 진화 (Fig. 1B-D)**: 이 피규어는 항생제 농도와 ARG의 발현 비용에 따라 중복 ARG가 어떻게 인구 내에서 확산되는지를 보여줍니다. 높은 항생제 농도에서 중복 ARG를 가진 세포가 우세해지는 경향을 나타냅니다.
- **진화 실험 결과 (Fig. 2)**: E. coli에서의 실험 결과를 보여주며, 항생제 선택이 중복 ARG의 진화를 유도한다는 것을 입증합니다. 항생제 없는 대조군에서는 중복이 관찰되지 않았습니다.

### 3. 테이블
- **생태적 카테고리별 중복 ARG 비율 (Table 1)**: 인간과 가축에서 분리된 세균에서 중복 ARG의 비율이 다른 환경에서 분리된 세균보다 유의미하게 높다는 것을 보여줍니다. 이는 항생제 사용과 관련이 있습니다.
- **임상 항생제 내성 균주에서의 중복 ARG (Table 2)**: 임상에서 분리된 항생제 내성 균주에서 중복 ARG의 비율이 일반적인 인간 균주보다 높다는 것을 보여줍니다.

### 4. 어펜딕스
- **어펜딕스 A**: 실험 방법과 데이터 분석에 대한 세부 정보를 제공합니다. 이 부분은 연구의 재현성을 높이는 데 중요한 역할을 합니다.
- **어펜딕스 B**: 추가적인 데이터와 결과를 제공하여 연구의 신뢰성을 높입니다.

---




### 1. Diagrams
- **Model Diagram (Fig. 1A)**: This diagram illustrates three subpopulations of bacteria: (1) cells with an ARG on the chromosome, (2) cells with a duplicated ARG on the chromosome, and (3) cells with a duplicated ARG on a plasmid. The model explains how antibiotic selection drives the evolution of duplicated ARGs.

### 2. Figures
- **Evolution of Duplicated ARGs under Antibiotic Selection (Fig. 1B-D)**: This figure shows how duplicated ARGs spread within a population depending on antibiotic concentration and the cost of ARG expression. It indicates that cells with duplicated ARGs tend to dominate at high antibiotic concentrations.
- **Evolution Experiment Results (Fig. 2)**: This figure presents results from experiments with E. coli, demonstrating that antibiotic selection drives the evolution of duplicated ARGs. No duplications were observed in the control populations without antibiotics.

### 3. Tables
- **Proportion of Duplicated ARGs by Ecological Category (Table 1)**: This table shows that the proportion of bacteria with duplicated ARGs is significantly higher in isolates from humans and livestock compared to those from other environments, indicating a link to antibiotic use.
- **Duplicated ARGs in Clinical Antibiotic-Resistant Isolates (Table 2)**: This table indicates that the proportion of duplicated ARGs in clinical antibiotic-resistant isolates is higher than in general human isolates.

### 4. Appendices
- **Appendix A**: Provides detailed information on experimental methods and data analysis, which is crucial for the reproducibility of the study.
- **Appendix B**: Offers additional data and results to enhance the reliability of the research findings.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@article{Maddamsetti2024,
  author = {Rohan Maddamsetti and Yi Yao and Teng Wang and Junheng Gao and Vincent T. Huang and Grayson S. Hamrick and Hye-In Son and Lingchong You},
  title = {Duplicated antibiotic resistance genes reveal ongoing selection and horizontal gene transfer in bacteria},
  journal = {Nature Communications},
  volume = {15},
  number = {1449},
  year = {2024},
  doi = {10.1038/s41467-024-45638-9},
  url = {https://doi.org/10.1038/s41467-024-45638-9}
}
```

### 시카고 스타일

Maddamsetti, Rohan, Yi Yao, Teng Wang, Junheng Gao, Vincent T. Huang, Grayson S. Hamrick, Hye-In Son, and Lingchong You. "Duplicated Antibiotic Resistance Genes Reveal Ongoing Selection and Horizontal Gene Transfer in Bacteria." *Nature Communications* 15 (2024): 1449. https://doi.org/10.1038/s41467-024-45638-9.
