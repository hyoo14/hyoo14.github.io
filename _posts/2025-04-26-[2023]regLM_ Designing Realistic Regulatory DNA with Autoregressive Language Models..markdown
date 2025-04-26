---
layout: post
title:  "[2023]regLM: Designing Realistic Regulatory DNA with Autoregressive Language Models."  
date:   2025-04-26 11:12:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

heyna 기반 언어 특성 추가해서 dna 생성   



짧은 요약(Abstract) :    






이 논문은 원하는 기능을 가진 **조절 DNA(cis-regulatory elements, CREs)** 를 설계하는 문제를 다룹니다. 저자들은 **regLM**이라는 새로운 프레임워크를 제안했는데, 이는 **HyenaDNA** 기반 **자동회귀 언어 모델(autoregressive language models)** 과 **예측용 지도 학습 모델**을 결합하여 현실적이고 다양한 조절 DNA를 생성하는 방법입니다. regLM은 기존의 gradient-based 방법이나 directed evolution 방법보다, **실험적으로 검증된 CRE들과 더 유사하고 생물학적으로 자연스러운** 시퀀스를 생성할 수 있습니다. 이 프레임워크는 **효모(yeast) 프로모터**와 **인간 세포 특이적 인핸서** 데이터셋에 대해 테스트되었으며, 다양한 강도와 세포 특이성을 가진 시퀀스들을 성공적으로 만들어냈습니다. 결과적으로 regLM은 원하는 생물학적 특성을 가진 합성 CRE를 효과적으로 생성할 수 있음을 보여주었고, 미래에는 안정성 및 안전성 기준까지 추가하여 응용할 수 있음을 시사합니다.

---


This paper addresses the challenge of designing **cis-regulatory DNA elements (CREs)** with specific desired properties. The authors introduce **regLM**, a new framework combining **HyenaDNA-based autoregressive language models** with **supervised sequence-to-function models** to generate realistic and diverse regulatory DNA sequences. Compared to traditional gradient-based or directed evolution approaches, regLM generates sequences that are more biologically realistic and closer to experimentally validated CREs. The method was tested on **yeast promoters** and **cell type-specific human enhancers**, successfully generating synthetic sequences with various strengths and specificities. Overall, regLM demonstrates the potential to design synthetic CREs with targeted biological functions, and future extensions may include filtering generated sequences for additional criteria such as safety and stability.





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




- **백본 (Backbone)**  
  regLM은 **HyenaDNA** 모델을 기반으로 합니다.  
  HyenaDNA는 **autoregressive (자동회귀) 언어 모델**로, 긴 DNA 시퀀스를 고해상도로 다룰 수 있도록 설계된 모델입니다. 기존에 흔히 쓰이던 **masked language model** (예: DNABERT, Nucleotide Transformer 등)과 달리, 시퀀스를 실제로 "생성"하는 데 적합합니다.

- **모델 구조 (Model Architecture)**  
  - Yeast (효모 프로모터 데이터):  
    ➔ **HyenaDNA 구조**를 그대로 사용하여 **처음부터 학습(train from scratch)**.
  - Human (인간 인핸서 데이터):  
    ➔ **사전 학습된 HyenaDNA** ('hyenadna-medium-160k-seqlen' 모델)을 **파인튜닝(fine-tuning)** 하여 사용.
  - 학습 방법:  
    ➔ 입력 DNA 시퀀스 앞에 **기능성 레이블(활성도 등급)** 을 **토큰으로 프리픽스(prefix)** 하여 추가 → **다음 뉴클레오타이드 예측(next token prediction)** 학습.

- **트레이닝 데이터 (Training Data)**  
  - **Yeast promoters (효모 프로모터)**  
    - 약 80bp 길이의 무작위 DNA 조각과, 두 종류의 배지(complex/defined media)에서의 활성도 측정값.
    - 2개의 레이블 토큰(복잡/단순 배지 각각에 대해 0~4로 활성도 구간 표시) 추가.
  - **Human enhancers (인간 인핸서)**  
    - 200bp 길이의 인간 게놈 기반 인핸서 조각.
    - 세 종류의 세포주(HepG2, K562, SK-N-SH)에서 각각 측정된 활성도 값.
    - 3개의 레이블 토큰(각 세포주에 대해 0~4로 활성도 구간 표시) 추가.

- **훈련 세팅 (Training Setup)**  
  - Optimizer: **AdamW**  
  - Batch size: **128**  
  - Epoch: **100**  
  - Loss: **Cross-entropy loss**  
  - 사용한 하드웨어: **NVIDIA A100 GPU**  
  - Yeast 학습 시 learning rate = 3e-4, Human fine-tuning 시 learning rate = 6e-4

---



- **Backbone**  
  regLM is based on the **HyenaDNA** model.  
  HyenaDNA is an **autoregressive language model** designed to handle long genomic sequences at high resolution. Unlike masked language models (e.g., DNABERT, Nucleotide Transformer), HyenaDNA is suitable for **sequence generation** tasks.

- **Model Architecture**  
  - Yeast (promoter data):  
    ➔ A **HyenaDNA model was trained from scratch** using yeast promoter data.
  - Human (enhancer data):  
    ➔ A **pretrained HyenaDNA model** ('hyenadna-medium-160k-seqlen') was **fine-tuned** on human enhancer data.
  - Training approach:  
    ➔ Functional labels (activity levels) were **prepended as prompt tokens** to each sequence, and the model was trained to perform **next token prediction** starting from these tokens.

- **Training Data**  
  - **Yeast promoters**:  
    - Random 80bp DNA fragments with measured promoter activities in two media types (complex and defined).
    - Two label tokens were prepended, each representing the activity quintile (0-4) for each medium.
  - **Human enhancers**:  
    - 200bp human genomic enhancer sequences.
    - Activity measured across three cell lines (HepG2, K562, SK-N-SH).
    - Three label tokens were prepended, each representing activity quintiles (0-4) for each cell line.

- **Training Setup**  
  - Optimizer: **AdamW**  
  - Batch size: **128**  
  - Epochs: **100**  
  - Loss function: **Cross-entropy loss**  
  - Hardware: **NVIDIA A100 GPU**  
  - Learning rate: **3e-4** for yeast training, **6e-4** for human fine-tuning





   
 
<br/>
# Results  




- **테스트 데이터**  
  - **Yeast promoters**:  
    - 약 80bp 길이의 효모 프로모터들. complex/defined 배지 각각에서 활성도 측정값 존재.
    - 50,000개 시퀀스를 테스트 셋으로 사용.
  - **Human enhancers**:  
    - 약 200bp 길이의 인간 인핸서들. HepG2, K562, SK-N-SH 세포주에서 활성도 측정.
    - 2,456개 (chromosome 13) 시퀀스를 테스트 셋으로 사용.

- **경쟁 모델 (Comparative Models)**  
  - **Directed Evolution (DE)**:  
    - 반복적으로 돌연변이(mutation)를 가하고, 예측된 활성도가 가장 높은 변이를 선택하는 방식.
  - **Ledidi**:  
    - 활성도를 최대화하면서도 가능한 적은 수의 변이를 적용하는 **gradient-based** 방법.

- **평가 메트릭 (Evaluation Metrics)**  
  - **활성도 예측 일치도**:  
    - 독립적으로 학습된 회귀 모델이 생성된 시퀀스들의 목표 활성도를 잘 예측하는지 확인.
  - **GC Content 유사도**:  
    - 합성 시퀀스들의 GC 비율이 실제 실험 데이터와 유사한지 측정.
  - **K-mer 빈도**:  
    - 합성된 시퀀스의 모든 4-mer 빈도를 계산 → 실험 데이터와의 차이 분석 (Mann-Whitney U-test 사용).
  - **TF Motif 유사도**:  
    - 전사인자 결합 모티프(TF motifs)의 빈도를 비교하여 생물학적 유사성 평가.
  - **Nearest Neighbor Analysis**:  
    - 4-mer/모티프 기반으로 합성 시퀀스의 최근접 이웃이 실제 실험 시퀀스인지 확인.

- **주요 결과 (Main Results)**  
  - regLM이 생성한 시퀀스는:
    - **활성도**:  
      ➔ 지정된 레이블(활성도 범주)과 잘 일치.
    - **생물학적 유사성**:  
      ➔ Directed Evolution이나 Ledidi보다 **GC content**, **4-mer 패턴**, **TF motif 빈도** 모두 실험 데이터와 더 유사.
      ➔ 예: DE는 47개의 4-mer에서 차이가 있었으나, regLM은 차이가 없음.
    - **다양성**:  
      ➔ regLM이 생성한 시퀀스들은 motif 다양성도 유지(Directed Evolution은 다양성 낮음).
    - **종 특이적 규칙 학습**:  
      ➔ 특정 TF 모티프가 고활성/저활성과 연결되는 패턴을 모델이 스스로 학습.

- **세포주 특이적 인핸서 결과 (Human Enhancer Results)**  
  - regLM은 세포주 특이적 인핸서(HepG2, K562, SK-N-SH)를 성공적으로 생성.
  - 생성된 인핸서들은 cell type-specific TF 모티프를 높은 빈도로 포함하고 있음 (예: K562-specific enhancer에는 GATA2 모티프가 많이 나타남).

---


- **Test Data**  
  - **Yeast promoters**:  
    - Approximately 80bp promoter sequences with measured activities in complex and defined media.
    - 50,000 sequences used for the test set.
  - **Human enhancers**:  
    - Approximately 200bp enhancer sequences with activity measured in HepG2, K562, and SK-N-SH cell lines.
    - 2,456 sequences (from chromosome 13) used for the test set.

- **Comparative Models**  
  - **Directed Evolution (DE)**:  
    - Iteratively introduces mutations, selecting the variant with the highest predicted activity at each step.
  - **Ledidi**:  
    - A gradient-based method that maximizes activity while minimizing the number of edits.

- **Evaluation Metrics**  
  - **Predicted Activity Agreement**:  
    - Checking whether an independently trained regression model predicts the desired activity for generated sequences.
  - **GC Content Similarity**:  
    - Measuring whether the GC content of synthetic sequences matches that of real experimental data.
  - **K-mer Frequency Analysis**:  
    - Comparing 4-mer frequencies between synthetic and real sequences (using Mann-Whitney U-tests).
  - **Transcription Factor (TF) Motif Similarity**:  
    - Evaluating the presence and frequency of TF motifs.
  - **Nearest Neighbor Analysis**:  
    - Assessing if synthetic sequences are most similar to real sequences based on k-mer or motif content.

- **Main Results**  
  - regLM-generated sequences:
    - **Activity**:  
      ➔ Match the intended activity labels accurately.
    - **Biological Realism**:  
      ➔ Higher similarity to real data in GC content, 4-mer distribution, and TF motif frequency compared to Directed Evolution and Ledidi.
      ➔ For example, Directed Evolution showed 47 differentially abundant 4-mers, while regLM showed none.
    - **Diversity**:  
      ➔ Maintains motif diversity comparable to real sequences (Directed Evolution showed reduced diversity).
    - **Species-Specific Learning**:  
      ➔ The model captures associations between specific TF motifs and promoter activity levels.

- **Human Enhancer Results**  
  - regLM successfully generated cell line-specific human enhancers.
  - The synthetic enhancers showed a high frequency of appropriate cell type-specific TF motifs (e.g., GATA2 motif enriched in K562-specific enhancers).





<br/>
# 예제  






- **트레이닝 데이터 예시**  
  - **Yeast promoters** (효모 프로모터 데이터)
    - 시퀀스 예:  
      ```
      ATCGGATCGAATTCGAGCTCGAGCTAGC... (80 bp)
      ```
    - 앞에 붙는 레이블(프리픽스):
      ```
      3 2 ATCGGATCGAATTCGAGCTCGAGCTAGC...
      ```
      → 여기서 `3 2`는 complex medium에서는 높은 활동성(3), defined medium에서는 중간 활동성(2)을 의미.

  - **Human enhancers** (인간 인핸서 데이터)
    - 시퀀스 예:  
      ```
      TGCACGTAGTCGTACGATCGTACGTAGCT... (200 bp)
      ```
    - 앞에 붙는 레이블(프리픽스):
      ```
      0 4 1 TGCACGTAGTCGTACGATCGTACGTAGCT...
      ```
      → 여기서 `0 4 1`은 HepG2에서는 낮은 활동성(0), K562에서는 높은 활동성(4), SK-N-SH에서는 약한 활동성(1)을 의미.

- **테스크 인풋-아웃풋 예시 (Task Input-Output Example)**  
  - **Input (Prompt)**:  
    ➔ "3 3" (복잡/정의 배지 모두에서 중간 활성도를 가진 프로모터를 생성하라)  
  - **Model Processing**:  
    ➔ HyenaDNA 기반 regLM이 "3 3" 레이블을 읽고, 이어지는 DNA 시퀀스를 **하나씩** 생성합니다.
  - **Output (Generated DNA Sequence)**:  
    ➔
    ```
    ATGCTGACGTTAGCTAGCTAGGCTAACGCTGACGATCAGT...
    (총 80개 염기 생성 완료)
    ```

- **생성된 시퀀스 평가**  
  - 독립적으로 학습한 **회귀 모델**이 이 시퀀스의 활성도를 예측.
  - 예측된 활성도가 "중간" 범위(훈련 데이터의 3번 구간) 안에 들지 않으면 버림(discard).
  - 통과한 시퀀스만 최종 결과로 사용.

- **특이사항**  
  - 모델은 무작위로 생성하지 않고, 프리픽스에 주어진 조건(레이블)에 맞춰, **생물학적 규칙(TF motif 배치 등)** 을 최대한 지키면서 시퀀스를 생성하려고 합니다.
  - 훈련 데이터에 존재하지 않는 시퀀스를 만들되, 기존 생물학적 분포와 통계적 특성(GC content, k-mer 등)은 유지합니다.

---



- **Training Data Example**  
  - **Yeast promoters**
    - Sequence example:
      ```
      ATCGGATCGAATTCGAGCTCGAGCTAGC... (80 bp)
      ```
    - Prefixed with labels:
      ```
      3 2 ATCGGATCGAATTCGAGCTCGAGCTAGC...
      ```
      → Here, `3 2` indicates high activity (3) in complex medium and medium activity (2) in defined medium.

  - **Human enhancers**
    - Sequence example:
      ```
      TGCACGTAGTCGTACGATCGTACGTAGCT... (200 bp)
      ```
    - Prefixed with labels:
      ```
      0 4 1 TGCACGTAGTCGTACGATCGTACGTAGCT...
      ```
      → Here, `0 4 1` indicates low activity in HepG2, high activity in K562, and weak activity in SK-N-SH.

- **Task Input-Output Example**  
  - **Input (Prompt)**:  
    ➔ "3 3" (Generate a yeast promoter with medium activity in both complex and defined media)
  - **Model Processing**:  
    ➔ The HyenaDNA-based regLM reads the "3 3" prefix and generates the DNA sequence **nucleotide by nucleotide**.
  - **Output (Generated DNA Sequence)**:  
    ➔
    ```
    ATGCTGACGTTAGCTAGCTAGGCTAACGCTGACGATCAGT...
    (80 nucleotides generated)
    ```

- **Evaluation of Generated Sequences**  
  - An independent **regression model** predicts the activity of the generated sequence.
  - If the predicted activity falls outside the expected "medium" activity range (corresponding to label 3), the sequence is discarded.
  - Only sequences matching the desired activity range are retained.

- **Important Details**  
  - The model does not generate sequences randomly; it conditions on the prefix label to generate biologically plausible sequences, respecting regulatory rules such as TF motif placements.
  - While generating novel sequences, the model preserves statistical characteristics like GC content and k-mer distribution to match real biological data.

---




<br/>  
# 요약   





regLM은 HyenaDNA 기반의 자동회귀 언어 모델을 사용해, 프리픽스된 활성도 레이블을 조건으로 DNA 시퀀스를 생성하고, 독립적인 회귀 모델로 필터링하여 원하는 기능을 갖춘 조절 DNA를 설계합니다.  
결과적으로 regLM은 기존 방법(Directed Evolution, Ledidi)보다 현실적이고 다양한 시퀀스를 생성했으며, 실제 생물학적 분포와 통계 특성을 잘 유지했습니다.  
예를 들어, 특정 레이블(예: "3 3")을 프리픽스로 주면, 중간 활성도를 갖는 80bp 길이의 효모 프로모터 시퀀스를 하나씩 생성할 수 있습니다.

---


regLM uses a HyenaDNA-based autoregressive language model to generate DNA sequences conditioned on prefixed activity labels, and filters them using an independent regression model to design regulatory DNA with desired properties.  
As a result, regLM produced more realistic and diverse sequences than traditional methods (Directed Evolution, Ledidi), while preserving biological distributions and statistical characteristics.  
For example, by prompting with a specific label (e.g., "3 3"), regLM can generate an 80bp yeast promoter sequence with medium activity in both media conditions.

---





<br/>  
# 기타  



- **구체적인 주요 피규어 (Figures)**  
  - **Figure 1 (regLM 개요도)**  
    - A~G 단계로 구성:  
      ➔ 레이블 추가 → HyenaDNA 기반 모델 학습 → 시퀀스 생성 → 회귀 모델로 필터링 → 최종 시퀀스 평가까지 전체 워크플로우를 시각화.
  - **Figure 2 (Yeast promoter 결과 분석)**  
    - 생성된 프로모터의 활동성, TF motif 포함 비율, GC content, 시퀀스 다양성, motif 위치 분포 등을 비교 분석.
    - 주요 항목:
      - B: 생성된 프로모터의 예측 활동성
      - C: TF motif 빈도 (활성화/억제 모티프)
      - E: GC content 비교
      - F: 10개 이웃 간 motif 다양성 거리
      - G: TF motif 위치 분포
  - **Figure 3 (Human enhancer 결과 분석)**  
    - 세포주 특이적 인핸서 생성 성능, nearest neighbor 분석, cell type-specific TF motif 빈도 등을 정리.

- **테이블**  
  - 본문에는 별도 테이블 형식은 없지만, **Supplementary Data (부록)** 에 수치 데이터와 비교 지표가 자세히 제시되어 있음 (예: 4-mer 차이 수, nearest neighbor 매칭 비율 등).

- **어펜딕스 (Appendix / Supplementary Sections)**  
  - **Training Details**:  
    - HyenaDNA, Enformer 기반 회귀 모델의 학습 세팅 (batch size, optimizer, learning rate 등) 명시.
  - **Data Processing**:  
    - Yeast 및 Human enhancer 데이터 처리 방법, 레이블 부여 기준, 데이터 split 방법 등을 설명.
  - **Synthetic Sequence Generation (regLM, DE, Ledidi 비교)**:  
    - 합성 방법별 시퀀스 생성, 필터링, 선택 기준을 모두 정리.
  - **In Silico Evaluation Methods**:  
    - GC content, 4-mer frequency, TF motif presence, diversity 계산 방법 및 통계 검정(예: Mann-Whitney U-test, Kruskal-Wallis test) 방법 상세 설명.
  - **Interpretation of regLM**:  
    - 모티프 삽입 후 모델 likelihood 변화를 분석해 모델이 실제로 biological motif 기능을 학습했는지 검증하는 실험 포함.

---



- **Key Figures**  
  - **Figure 1 (Overview of regLM)**  
    - Shows the workflow from adding label tokens, training HyenaDNA, generating sequences, filtering with regression models, to evaluating the final sequences.
  - **Figure 2 (Yeast promoter analysis)**  
    - Detailed comparison of generated promoters in terms of predicted activity, TF motif presence, GC content, sequence diversity, and motif positional distributions.
    - Highlights:
      - B: Predicted activity levels of generated promoters
      - C: Frequency of activating and repressing TF motifs
      - E: GC content comparison
      - F: Diversity distance among 10 nearest neighbors
      - G: Distribution of TF motif positions
  - **Figure 3 (Human enhancer analysis)**  
    - Evaluation of generated cell type-specific enhancers, nearest neighbor analysis based on motif and k-mer content, and frequency of cell type-specific TF motifs.

- **Tables**  
  - No major tables are present in the main text, but **supplementary data** include detailed numerical comparisons such as differential 4-mer counts and nearest neighbor matching rates.

- **Appendix / Supplementary Sections**  
  - **Training Details**:  
    - Provides training settings for HyenaDNA and Enformer-based regression models (batch size, optimizer, learning rate, etc.).
  - **Data Processing**:  
    - Describes processing steps for yeast promoters and human enhancers, label assignment, and dataset splitting.
  - **Synthetic Sequence Generation (regLM, DE, Ledidi comparison)**:  
    - Detailed generation, filtering, and selection procedures for each method.
  - **In Silico Evaluation Methods**:  
    - Methods for calculating GC content, 4-mer frequency, TF motif presence, diversity measures, and statistical tests (e.g., Mann-Whitney U-test, Kruskal-Wallis test).
  - **Interpretation of regLM**:  
    - Tests to validate whether regLM truly learned biological motif functions by inserting motifs and analyzing model likelihood shifts.




<br/>
# refer format:     


@inproceedings{lal2023reglm,
  title={regLM: Designing realistic regulatory DNA with autoregressive language models},
  author={Lal, Avantika and Biancalani, Tommaso and Eraslan, Gokcen},
  booktitle={Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023},
  organization={NeurIPS},
  url={https://github.com/Genentech/regLM}
}




Avantika Lal, Tommaso Biancalani, and Gokcen Eraslan.
"regLM: Designing Realistic Regulatory DNA with Autoregressive Language Models."
In Proceedings of the 37th Conference on Neural Information Processing Systems (NeurIPS), 2023. https://github.com/Genentech/regLM.

