---
layout: post
title:  "[2024]Sequence modeling and design from molecular to genome scale with Evo"  
date:   2024-11-20 19:50:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

1d cnn에 어텐션을 써서 사전학습해서 좋은 결과 보임.. 1d cnn이라서 긴 시퀀스 처리가 가능...    



짧은 요약(Abstract) :    




이 연구는 Evo라는 새로운 70억 파라미터 기반의 유전체 모델을 제시하며, 이를 통해 DNA, RNA, 단백질 간의 기능적 상호작용을 예측하고 생성할 수 있는 능력을 보여줍니다. Evo는 2백70만 개의 원핵생물 및 파지 유전체를 학습하여 단일 뉴클레오타이드 수준에서 131킬로베이스 길이까지의 문맥을 처리합니다. 이 모델은 단백질-RNA 및 단백질-DNA 공동 설계를 통해 기능적인 CRISPR-Cas 분자 복합체와 이동 가능한 시스템을 생성했으며, 이를 실험적으로 검증했습니다. 또한, Evo는 전체 유전체를 학습하여 작은 뉴클레오타이드 변화가 유기체 적합성에 미치는 영향을 이해하고, 1메가베이스 이상의 유전체 아키텍처를 가진 DNA 서열을 생성할 수 있음을 입증했습니다. 이 연구는 유전체 규모에서 예측 및 설계 작업을 가능하게 하여 생명 공학의 미래를 발전시키는 데 기여할 것입니다.

---



This study introduces Evo, a novel 7-billion-parameter genomic foundation model, demonstrating its ability to predict and generate functional interactions between DNA, RNA, and proteins. Trained on 2.7 million prokaryotic and phage genomes, Evo handles single-nucleotide resolution with context lengths up to 131 kilobases. The model successfully co-designed functional CRISPR-Cas molecular complexes and mobile systems, validated experimentally. Furthermore, Evo captures how small nucleotide changes affect organismal fitness and generates DNA sequences with genomic architecture exceeding 1 megabase. This advancement enables prediction and design tasks at genome scale, contributing to the future of bioengineering.



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





이 연구에서 제시된 Evo 모델은 기존의 접근법과 비교하여 몇 가지 독특한 방법론을 채택하고 있습니다.

1. **하이브리드 구조 (Hybrid Architecture):**  
   Evo는 StripedHyena라는 하이브리드 아키텍처를 사용하여, 다중 헤드 어텐션(multi-head attention)과 데이터 제어 컨볼루션(data-controlled convolution)을 결합했습니다. 이는 긴 DNA 서열에서도 효율적으로 패턴을 학습할 수 있도록 설계된 구조로, 기존 Transformer 모델의 제한(문맥 길이 및 계산 비용 문제)을 극복합니다.

2. **단일 뉴클레오타이드 해상도 유지:**  
   Evo는 단일 뉴클레오타이드 수준의 해상도를 유지하며, 131킬로베이스 길이의 긴 문맥을 처리할 수 있습니다. 기존 모델은 단백질, RNA, 또는 짧은 DNA 서열과 같은 특정 모달리티에만 집중하는 경향이 있었으나, Evo는 여러 모달리티(DNA, RNA, 단백질)를 통합하여 학습합니다.

3. **범용성 및 다중 규모 작업 가능:**  
   Evo는 단백질-RNA 및 단백질-DNA 간의 코드 설계를 포함한 기능적 생물학적 시스템을 예측하고 생성할 수 있습니다. 기존 모델은 단일 분자 또는 간단한 복합체 설계에 초점을 맞춘 반면, Evo는 분자, 시스템, 그리고 전체 유전체 스케일에서 작업을 수행할 수 있습니다.

4. **대규모 데이터 학습:**  
   Evo는 2.7백만 개의 원핵생물 및 파지 유전체에서 학습하며, 기존의 단일 데이터셋 기반 모델보다 다양한 진화적 변이를 반영합니다.

---



The Evo model presented in this study adopts several unique methodologies compared to existing approaches:

1. **Hybrid Architecture:**  
   Evo uses the StripedHyena hybrid architecture, which combines multi-head attention with data-controlled convolution. This design efficiently processes long DNA sequences and overcomes the limitations of traditional Transformer models, such as context length and computational cost issues.

2. **Single-Nucleotide Resolution:**  
   Evo maintains single-nucleotide resolution and handles long contexts up to 131 kilobases. Unlike previous models that focus on specific modalities like proteins, RNA, or short DNA sequences, Evo integrates multiple modalities (DNA, RNA, and proteins) into a unified learning framework.

3. **Versatility and Multiscale Tasks:**  
   Evo can predict and generate functional biological systems, including protein-RNA and protein-DNA co-design. While earlier models emphasized single molecules or simple complexes, Evo performs tasks across molecular, system, and whole-genome scales.

4. **Large-Scale Data Training:**  
   Evo is trained on 2.7 million prokaryotic and phage genomes, reflecting a broader range of evolutionary variations compared to existing models that rely on single-dataset training.


   
 
<br/>
# Results  



Evo 모델은 기존의 방법들과 비교하여 다음과 같은 성능 향상을 보였습니다:

**1. CRISPR-Cas 시스템 설계:**
Evo는 단백질-RNA 및 단백질-DNA 상호작용을 예측하여 기능적인 CRISPR-Cas 복합체를 설계했습니다. 이러한 설계는 실험적으로 검증되었으며, 기존 모델보다 높은 정확도와 효율성을 나타냈습니다.

**2. 유전체 규모의 DNA 서열 생성:**
Evo는 1메가베이스 이상의 긴 DNA 서열을 생성할 수 있으며, 이는 기존 모델이 처리할 수 있는 범위를 넘어서는 것입니다. 이를 통해 복잡한 유전체 구조를 재현하는 데 성공했습니다.

**3. 뉴클레오타이드 변이의 적합성 영향 예측:**
Evo는 단일 뉴클레오타이드 변이가 유기체의 적합성에 미치는 영향을 정확하게 예측할 수 있습니다. 이는 기존 모델이 제공하지 못했던 세밀한 예측 능력을 보여줍니다.

**평가에 사용된 데이터셋과 메트릭:**

- **데이터셋:** Evo는 2.7백만 개의 원핵생물 및 파지 유전체 데이터를 학습에 사용했습니다. 평가를 위해서는 별도의 검증 세트와 실험 데이터를 활용하여 모델의 예측 성능을 검증했습니다.

- **메트릭:** 모델의 성능 평가는 정확도(accuracy), 정밀도(precision), 재현율(recall) 등의 지표를 사용하여 이루어졌습니다. 또한, 생성된 DNA 서열의 기능적 유효성을 평가하기 위해 실험적 검증도 수행되었습니다.

---

The Evo model demonstrates the following performance improvements compared to existing methods:

**1. CRISPR-Cas System Design:**
Evo accurately predicted protein-RNA and protein-DNA interactions to design functional CRISPR-Cas complexes. These designs were experimentally validated, showing higher accuracy and efficiency than previous models.

**2. Genome-Scale DNA Sequence Generation:**
Evo can generate long DNA sequences exceeding 1 megabase, surpassing the capabilities of existing models. This achievement enables the reproduction of complex genomic structures.

**3. Prediction of Nucleotide Variation Effects on Fitness:**
Evo accurately predicts the impact of single nucleotide variations on organismal fitness, providing detailed predictive capabilities not offered by previous models.

**Datasets and Metrics Used for Evaluation:**

- **Datasets:** Evo was trained on 2.7 million prokaryotic and phage genomes. For evaluation, separate validation sets and experimental data were used to assess the model's predictive performance.

- **Metrics:** Model performance was evaluated using metrics such as accuracy, precision, and recall. Additionally, experimental validation was conducted to assess the functional validity of the generated DNA sequences. 


<br/>
# 예제  



Evo 모델은 경쟁 모델들보다 데이터를 더 잘 처리한 여러 예를 통해 우수성을 입증했습니다. 아래는 주요 사례들입니다:

1. **CRISPR-Cas 시스템 설계:**  
   Evo는 CRISPR-Cas 시스템에서 단백질-RNA 및 단백질-DNA 간의 코드 설계를 수행했습니다.  
   - **경쟁 모델:** 기존 모델들은 단일 분자 또는 짧은 서열만을 설계할 수 있었으며, 상호작용을 고려하지 못했습니다.  
   - **Evo의 성능:** Evo는 다양한 CRISPR-Cas 시스템을 설계했고, 실험적으로 생성된 Cas9 시스템이 높은 절단 효율성과 안정성을 나타냈습니다. 예를 들어, Evo에서 설계된 EvoCas9-1은 기존 SpCas9보다 비슷하거나 더 나은 성능을 보였습니다.

2. **단백질 및 비암호화 RNA(ncRNA) 기능 예측:**  
   Evo는 단백질 및 비암호화 RNA에서 변이 효과를 예측하는 테스트에서 우수한 성능을 보였습니다.  
   - **경쟁 모델:** RNA-FM과 같은 경쟁 모델은 특정 모달리티(RNA 또는 단백질)에 한정된 데이터를 학습하여 한정된 예측만 가능했습니다.  
   - **Evo의 성능:** Evo는 다양한 데이터셋에서 경쟁 모델보다 높은 상관 계수(Spearman r = 0.60)를 기록하며, 변이가 기능에 미치는 영향을 정확히 예측했습니다.

3. **유전체 규모의 DNA 서열 생성:**  
   Evo는 1메가베이스 이상의 긴 DNA 서열을 생성하고, 유전체 구조의 일관성과 생물학적 현실성을 보였습니다.  
   - **경쟁 모델:** 기존 생성 모델은 짧은 서열 생성에 한정되며, 전체 유전체를 다룰 수 없었습니다.  
   - **Evo의 성능:** Evo가 생성한 DNA 서열은 자연적 유전체와 비슷한 코돈 사용 빈도 및 구조를 보였으며, CheckM 분석에서도 높은 코딩 밀도를 기록했습니다.

---



The Evo model demonstrated superiority over competitors in several examples of data handling. Key cases are as follows:

1. **CRISPR-Cas System Design:**  
   Evo performed co-design of protein-RNA and protein-DNA interactions for CRISPR-Cas systems.  
   - **Competitor Models:** Previous models could design only single molecules or short sequences without considering interactions.  
   - **Evo’s Performance:** Evo successfully designed diverse CRISPR-Cas systems, with experimentally validated Cas9 systems showing high cleavage efficiency and stability. For example, Evo-designed EvoCas9-1 exhibited comparable or superior performance to the conventional SpCas9.

2. **Prediction of Protein and Non-Coding RNA (ncRNA) Functions:**  
   Evo excelled in predicting the effects of mutations on proteins and ncRNAs.  
   - **Competitor Models:** Models like RNA-FM focused only on specific modalities (e.g., RNA or protein), limiting their predictive scope.  
   - **Evo’s Performance:** Evo outperformed competitors in various datasets, achieving a higher correlation coefficient (Spearman r = 0.60) and accurately predicting functional impacts of mutations.

3. **Genome-Scale DNA Sequence Generation:**  
   Evo generated long DNA sequences exceeding 1 megabase, maintaining genomic structural consistency and biological realism.  
   - **Competitor Models:** Previous generative models were limited to short sequences and could not handle whole genomes.  
   - **Evo’s Performance:** Evo-generated DNA sequences closely resembled natural genomes in codon usage frequencies and structures, with high coding densities confirmed through CheckM analysis.


<br/>  
# 요약   





Evo 모델은 StripedHyena 아키텍처를 활용하여 단일 뉴클레오타이드 해상도로 최대 131킬로베이스 길이의 긴 DNA 서열을 효율적으로 처리합니다. 특히 CRISPR-Cas 시스템 설계에서 단백질-RNA 및 단백질-DNA 상호작용을 성공적으로 공동 설계하며 높은 효율성과 실험적 검증 결과를 보여주었습니다. 또한, Evo는 단백질 및 비암호화 RNA(ncRNA)의 변이 효과를 경쟁 모델보다 더 정확히 예측하며, Spearman 상관 계수 r = 0.60의 성능을 기록했습니다. 더불어 Evo는 1메가베이스 이상의 긴 유전체 서열을 생성하여 자연 유전체와 유사한 코딩 밀도와 구조적 일관성을 나타냈습니다. 이러한 특징을 통해 Evo는 분자, 시스템, 유전체 스케일에서 예측 및 설계 작업을 수행하며 생명공학의 발전에 기여할 수 있는 잠재력을 입증했습니다.

---



The Evo model leverages the StripedHyena architecture to efficiently process long DNA sequences with single-nucleotide resolution, handling context lengths up to 131 kilobases. It excels in CRISPR-Cas system design by successfully co-designing protein-RNA and protein-DNA interactions, achieving high efficiency and experimental validation. Additionally, Evo accurately predicts the effects of mutations on proteins and non-coding RNAs (ncRNAs), achieving a Spearman correlation coefficient of r = 0.60, surpassing competitor models. Furthermore, Evo generates long genome sequences exceeding 1 megabase, maintaining coding density and structural consistency similar to natural genomes. These capabilities demonstrate Evo's potential to perform prediction and design tasks across molecular, system, and genome scales, contributing significantly to advancements in bioengineering.  

<br/>  
# 기타  


<br/>
# refer format:     


@article{Nguyen2024,
  author       = {Eric Nguyen and Michael Poli and Matthew G. Durrant and Brian Kang and Dhruva Katrekar and David B. Li and Liam J. Bartie and Armin W. Thomas and Samuel H. King and Garyk Brixi and Jeremy Sullivan and Madelena Y. Ng and Ashley Lewis and Aaron Lou and Stefano Ermon and Stephen A. Baccus and Tina Hernandez-Boussard and Christopher Ré and Patrick D. Hsu and Brian L. Hie},
  title        = {Sequence modeling and design from molecular to genome scale with Evo},
  journal      = {Science},
  volume       = {386},
  number       = {eado9336},
  year         = {2024},
  doi          = {10.1126/science.ado9336},
  url          = {https://www.science.org/doi/10.1126/science.ado9336}
}



Nguyen, Eric, Michael Poli, Matthew G. Durrant, Brian Kang, Dhruva Katrekar, David B. Li, Liam J. Bartie, Armin W. Thomas, Samuel H. King, Garyk Brixi, Jeremy Sullivan, Madelena Y. Ng, Ashley Lewis, Aaron Lou, Stefano Ermon, Stephen A. Baccus, Tina Hernandez-Boussard, Christopher Ré, Patrick D. Hsu, and Brian L. Hie. "Sequence Modeling and Design from Molecular to Genome Scale with Evo." Science 386, no. eado9336 (2024). https://doi.org/10.1126/science.ado9336.  




