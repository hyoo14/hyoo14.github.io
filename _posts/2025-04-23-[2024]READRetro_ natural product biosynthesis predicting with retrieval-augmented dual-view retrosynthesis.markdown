---
layout: post
title:  "[2024]READRetro: natural product biosynthesis predicting with retrieval-augmented dual-view retrosynthesis"  
date:   2025-04-23 21:56:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


READRetro는 목표 천연물의 SMILES 또는 분자 구조를 입력받아, 단일 반응 예측과 대사 반응 회수를 결합해 해당 분자의 생합성 경로를 단계적으로 계획하는 시스템이다.  


플래닝 예시:  

입력 (Input)  

SMILES of target compound: CC1=C2[C@@H]3CN4CCC5=CC(=C(C=C5[C@H]4C3=CC=C2OC1)N)OC

또는 그래프 구조

Step 1 (단일 스텝 역반응 예측)

예측: tabersonine ⟶ intermediate A

방식: Retroformer/Graph2SMILES + reaction retriever

선택된 반응: cyclization of strictosidine aglycone

Step 2 (다음 전구체 예측)

intermediate A ⟶ intermediate B

예: tryptamine + secologanin

retriever 또는 모델이 "strictosidine" 반응 기억

Step 3 (출발물질 도달 확인)

intermediate B = tryptamine + secologanin

이들은 모두 predefined building blocks로 포함됨

전체 경로 출력 (Multi-step pathway)

tryptamine + secologanin → strictosidine → aglycone → tabersonine




----> 잘은 모르겠지만 NLP Planning을 적용하기 좋은 것 같음  


짧은 요약(Abstract) :    


---


식물은 환경과의 상호작용을 위해 다양한 이차대사산물을 생성하는데, 이들의 생합성 경로를 분자 수준에서 예측하는 일은 여전히 도전적입니다. 이 논문에서는 식물 천연물의 생합성 경로를 예측하기 위한 **READRetro**라는 새로운 바이오-레트로합성(bio-retrosynthesis) 도구를 제안합니다. READRetro는 SMILES 문자열과 그래프 정보를 모두 활용하는 **듀얼 뷰(single-step) 모델**과, 다단계 생합성 경로를 계획할 수 있는 **Retro* 알고리즘 기반의 탐색 모델**, 그리고 **대규모 반응 데이터베이스로부터 정보를 회수하는 리트리버(retriever)**를 결합한 구조입니다. 평가 결과, READRetro는 기존 방법들보다 단일 및 다단계 경로 예측에서 더 우수한 성능을 보였으며, 알려진 생합성 경로는 물론, 새로운 경로도 성공적으로 제시함으로써 실제 식물 생합성 연구에 효과적인 도구임을 입증하였습니다.

---


Plants produce diverse secondary metabolites to interact with their environment, but predicting their complete biosynthetic pathways remains a major challenge. In this paper, the authors introduce **READRetro**, a practical bio-retrosynthesis tool designed to predict the biosynthetic pathways of plant natural products. READRetro combines dual-view single-step retrosynthesis models that leverage both SMILES and molecular graphs, a multi-step pathway planner based on the Retro* algorithm, and retrievers that incorporate knowledge from known reaction databases. The system significantly outperforms conventional models in both single- and multi-step retrosynthesis tasks. READRetro successfully recapitulates known pathways and suggests plausible novel pathways, demonstrating its value as a powerful tool for studying plant biosynthesis. A user-friendly web platform and open-source code are provided for broader use in the research community.

---




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




#### 1.  **트레이닝 데이터 구성**
- 주요 데이터셋은 **BioChem**, **USPTO_NPL**, 그리고 이 둘을 합친 **BioChem+USPTO_NPL**.
- **BioChem**은 METACYC, KEGG, METANETX에서 추출된 생화학 반응 기반 데이터셋이고, 자연물 중심 반응이 포함됨.
- **Atom-mapped 데이터셋**을 만들기 위해 RXNMapper를 사용함 → Graph 기반 모델 학습에 필수.
- 모델 일반화 성능을 평가하기 위해 **훈련셋과 겹치는 반응을 제거한 clean 버전**도 별도로 구축함.

#### 2.  **모델 구조**
- **단일 스텝(single-step) 예측 모델**:
  - **Retroformer**: SMILES + graph 정보 기반의 시퀀스-투-시퀀스 모델. 전역+지역 self-attention 구조.
  - **Graph2SMILES**: 그래프 인코더(GNN) + 트랜스포머 디코더 구조.
  - **Retriever**: 룰 기반 모듈로, 훈련 데이터에 존재하는 반응을 그대로 기억하고 꺼내옴.

- **다단계(multi-step) 레트로합성**:
  - **Retro***: A* 유사 트리 탐색 알고리즘을 사용해 단일 스텝 예측을 반복하여 전체 경로 생성.
  - **Pathway retriever**: KEGG에서 얻은 경로 정보를 기반으로 중간 단계에서 관련 대사경로를 회수해 탐색 종료를 유도함.

#### 3. **모델 학습 및 평가**
- PyTorch로 구현되었으며, Retroformer는 160만 step, Graph2SMILES는 약 8만 step 이상 학습.
- 성능 평가는 단일 스텝(top-k accuracy)과 다단계 성공률/정답 경로 일치율을 기준으로 수행함.
- 최적화를 위해 여러 layout, batch size, learning rate 실험이 수행됨.

---


#### 1.  **Training Data Construction**
- The main datasets include **BioChem**, **USPTO_NPL**, and their merged version **BioChem+USPTO_NPL**.
- **BioChem** was curated from METACYC, KEGG, and METANETX and contains metabolic reactions with a focus on natural products.
- To enable training for graph-based models, atom-mapped datasets were generated using **RXNMapper**.
- A cleaned version excluding overlapping reactions was also built to evaluate model generalizability.

#### 2.  **Model Architecture**
- **Single-step retrosynthesis models**:
  - **Retroformer**: A dual-representation model using both SMILES and molecular graphs, incorporating global and local self-attention.
  - **Graph2SMILES**: Uses a GNN-based encoder followed by a transformer decoder to predict reactions.
  - **Reaction retriever**: A rule-based module that memorizes and retrieves exact reactions from the training set.

- **Multi-step retrosynthesis model**:
  - **Retro\***: A tree search algorithm akin to A* that expands pathways by recursively calling the single-step models.
  - **Pathway retriever**: Fetches reference pathways from KEGG to efficiently terminate branches during search with known intermediate compounds.

#### 3.  **Training and Evaluation**
- Models were implemented in **PyTorch**. Retroformer was trained for 1.6 million steps; Graph2SMILES was trained for over 80,000 steps.
- Evaluation was done using **top-k accuracy** (single-step) and **success rate / hit rate** (multi-step).
- Several hyperparameters like layout, batch size, and learning rate were tuned for optimal performance.




   
 
<br/>
# Results  




#### 1.  **경쟁 모델**
- **BioNavi-NP**, **RetroPathRL**, **MEGAN**, **Graph2SMILES**, **Retroformer** 등이 비교 대상.
- 이 중 **BioNavi-NP**는 기존 SOTA로, vanilla Transformer 기반의 앙상블 구조를 사용함.
- **MEGAN**, **GraphRetro** 등은 단일 단계 예측에서는 경쟁력이 있었지만, 다단계에서는 성능 저조.

#### 2.  **평가 메트릭**
- **단일 스텝 예측 (Single-step retrosynthesis)**: Top-1, Top-3, Top-5, Top-10 accuracy.
- **다단계 경로 예측 (Multi-step retrosynthesis)**:
  - **Success rate**: 목표 분자로부터 빌딩 블록까지 도달한 비율.
  - **Hit rate of building blocks**: 정답 빌딩 블록을 정확히 예측한 비율.
  - **Hit rate of pathways**: 전체 생합성 경로가 ground truth와 일치하는 비율.

#### 3.  **READRetro 성능 요약**
- **단일 스텝**:  
  - Ensemble(Retroformer + Graph2SMILES)은 top-1 기준 최대 23.4%, top-10 기준 59.3% (clean set 기준 22.7%, 58.1%).
  - BioNavi-NP 대비 전 구간에서 정확도 향상.
- **다단계** (BioChem+USPTO_NPL 기준):
  - Success rate: **93.2%**
  - Hit rate of building blocks: **87.2%**
  - Hit rate of pathways: **66.6%**
  - BioNavi-NP 대비 pathway 정확도는 약 **3배 이상 향상**됨 (24.7% → 66.6%)

#### 4.  **실행 시간**
- RetroPathRL: 2시간, BioNavi-NP: 18시간.
- READRetro: 약 5.7시간 (RTX 3090 사용) → 실용성 측면에서도 우수.

---



#### 1.  **Baseline Models**
- READRetro was compared against **BioNavi-NP**, **RetroPathRL**, **MEGAN**, **Graph2SMILES**, and **Retroformer**.
- **BioNavi-NP** served as the prior state-of-the-art model, utilizing a vanilla Transformer-based ensemble.
- Some models like MEGAN and GraphRetro performed well in single-step prediction but failed in multi-step planning.

#### 2.  **Evaluation Metrics**
- **Single-step retrosynthesis**: Top-1, Top-3, Top-5, Top-10 accuracy.
- **Multi-step retrosynthesis**:
  - **Success rate**: Whether a pathway from the target molecule to building blocks was successfully found.
  - **Hit rate of building blocks**: Accuracy in recovering the correct building blocks.
  - **Hit rate of pathways**: Whether the predicted pathway exactly matches the ground truth.

#### 3.  **READRetro Performance Highlights**
- **Single-step**:
  - Ensemble of Retroformer + Graph2SMILES achieved up to **23.4% top-1** and **59.3% top-10 accuracy** (22.7% / 58.1% for clean set).
  - Outperformed BioNavi-NP across all metrics.
- **Multi-step (on BioChem+USPTO_NPL)**:
  - Success rate: **93.2%**
  - Hit rate of building blocks: **87.2%**
  - Hit rate of pathways: **66.6%**
  - READRetro showed more than **3x improvement** in pathway accuracy over BioNavi-NP (66.6% vs. 24.7%).

#### 4.  **Runtime Comparison**
- RetroPathRL: 2 hours, BioNavi-NP: 18 hours.
- READRetro: **5.7 hours** (on NVIDIA RTX 3090), showing strong efficiency for real-world use.

---



<br/>
# 예제  




#### 1.  **트레이닝 데이터**
- **BioChem**: KEGG, METACYC, MetaNetX 기반 생화학 반응 데이터.
- **USPTO_NPL**: 자연물 유사 분자 반응만 추출한 버전.
- **BioChem+USPTO_NPL**: 둘을 합친 메인 훈련 데이터셋 (약 94,000개 반응).
- 그래프 기반 모델 훈련을 위해 **RXNMapper로 atom-mapping** 수행.
- **Clean 버전**은 일반화 성능 평가용으로 훈련셋에 포함된 반응 제거한 데이터셋.

#### 2.  **테스트 데이터**
- **단일 스텝 평가용**:  
  - `single-step 1000 reactions test set`: 훈련셋과 겹치지 않도록 설계됨.
- **다단계 평가용**:  
  - `multi-step 368 test set`: 각 타겟 분자에 대해 ground-truth 경로 존재.
  - 대부분의 반응이 훈련셋에 포함됨 (92.9%), 따라서 **기억 기반 성능 검증**에 적합.
  - Clean 버전에서는 동일한 테스트셋을 훈련셋에서 제거하고 사용 → **일반화 테스트**.

#### 3.  **테스트 태스크**
- **단일 스텝 태스크**:
  - 주어진 제품 분자(SMILES or 그래프)로부터 가능한 전구체 반응을 예측.
  - top-k accuracy로 평가.

- **다단계 태스크**:
  - 목표 분자로부터 시작하여 트리 탐색(Retro*)으로 **전체 생합성 경로 생성**.
  - `성공률`, `빌딩 블록 정답률`, `전체 경로 정답률`로 평가.
  - 테스트 예시:  
    - **Catharanthine / Tabersonine**: 항암제 전구체로 실제 경로 재현 성공.  
    - **Menisdaurilide**: 알려지지 않은 경로를 생화학적으로 타당한 중간체 포함하여 예측.  
    - **Cannabichromenic acid**: 다중 경로(MVA + AA/MA) 통합 성공.  
    - **Glucotropaeolin**: 반응 retriever + pathway retriever의 상호작용으로 정답 경로 완성.

---


#### 1. **Training Data**
- **BioChem**: Curated from KEGG, METACYC, and MetaNetX, focused on biosynthetic reactions.
- **USPTO_NPL**: Extracted from USPTO dataset to include only natural product-like reactions.
- **BioChem+USPTO_NPL**: Main training dataset (~94,000 reactions), also used in atom-mapped form via **RXNMapper**.
- **Clean version**: Excludes training-set reactions from the test set for generalization testing.

#### 2.  **Test Data**
- **Single-step evaluation**:
  - `single-step 1000 reactions test set`: Designed to avoid overlaps with training data.
- **Multi-step evaluation**:
  - `multi-step 368 test set`: Each target molecule is paired with a known biosynthetic pathway.
  - Most reactions (~92.9%) are present in the training set, ideal for evaluating memorability.
  - In the clean version, these reactions are excluded from training to evaluate **generalization**.

#### 3.  **Tasks Performed**
- **Single-step task**:
  - Predict reactants from a given product using SMILES or graph input.
  - Evaluated using top-k accuracy.

- **Multi-step task**:
  - Plan full biosynthetic routes using Retro* search algorithm.
  - Evaluated using: **success rate**, **building block hit rate**, and **pathway hit rate**.
  - Case examples:
    - **Catharanthine / Tabersonine**: READRetro accurately reproduced real-world pathways.
    - **Menisdaurilide**: Predicted novel but chemically valid intermediates.
    - **Cannabichromenic acid**: Merged routes from MVA and AA/MA pathways.
    - **Glucotropaeolin**: Successfully combined outputs from both retrievers to reconstruct the known biosynthesis route.





<br/>  
# 요약   





READRetro는 SMILES와 그래프 기반 표현을 함께 사용하는 두 가지 단일 스텝 모델(Retroformer, Graph2SMILES)과, 기억 기반의 반응/경로 retriever를 통합한 다단계 생합성 경로 예측 시스템이다.  
BioChem과 USPTO_NPL을 기반으로 학습된 READRetro는 기존 BioNavi-NP보다 단일 스텝 정확도와 다단계 경로 재현률 모두에서 뛰어난 성능(예: pathway hit rate 66.6%)을 보였다.  
실제 생합성 경로가 알려진 tabersonine, catharanthine, glucotropaeolin 등을 정확히 재현하거나, menisdaurilide처럼 알려지지 않은 경로도 생화학적으로 타당한 방식으로 제안할 수 있었다.

---



READRetro is a multi-step biosynthetic pathway prediction framework that integrates two single-step models (Retroformer and Graph2SMILES) using both SMILES and graph representations, along with retrievers for recalling known reactions and pathways.  
Trained on BioChem and USPTO_NPL datasets, READRetro outperformed prior models like BioNavi-NP in both single-step accuracy and multi-step pathway recovery, achieving a pathway hit rate of 66.6%.  
It successfully reproduced known biosynthetic pathways of compounds like tabersonine, catharanthine, and glucotropaeolin, and proposed chemically plausible novel routes for compounds like menisdaurilide.




<br/>  
# 기타  




#### 1.  **Figure**
- **Fig. 1**: READRetro의 전체 구조도.  
  - 단일 스텝 예측(빨간 박스)과 다단계 탐색(검정 박스)이 분리되어 있으며, Retroformer, Graph2SMILES, Retriever가 상호작용하여 반응을 생성하고 점수화함.
- **Fig. 2**: 하이퍼파라미터 튜닝 결과와 화학 클래스별 성능 비교.
  - 레이아웃, 배치사이즈, 러닝레이트에 따른 top-k 정확도 변화.
  - AA/MA, MVA/MEP 등 대사경로별 success rate, hit rate 분석.
- **Fig. 3**: 실제 생합성 경로 예시 (tabersonine, menisdaurilide, cannabichromenic acid 등).
  - 생성된 반응(검정), retriever 기반 반응(빨강), pathway retriever 기반 반응(파랑)으로 색상 구분.
- **Fig. 4**: 웹 인터페이스와 결과 시각화 화면 예시.

#### 2.  **Table**
- **Table 1**: 단일 스텝 예측의 top-k 정확도 비교표.  
  - Retroformer, Graph2SMILES, Ensemble, BioNavi-NP 간 성능 비교.
- **Table 2**: 다단계 예측에서의 성능 비교표.  
  - success rate, building block hit rate, pathway hit rate 비교.
  - READRetro가 거의 모든 항목에서 최고 성능 기록.

#### 3.  **어펜딕스 / 보조 자료**
- **Fig. S1–S12**:  
  - S1: 분자 표현(SMILES vs. Graph)의 차이 시각화  
  - S2: Train/test 분포 시각화 (t-SNE 등)  
  - S3: Retro* 트리 탐색 구조  
  - S4: 평가 예시(benzyl acetone pathway)  
  - S5–S9: 다양한 분자의 경로 예측 결과 (glucotropaeolin, dolabellane 등)  
  - S10: Predefined building blocks 목록  
  - S11–S12: 기존 도구들과 READRetro의 비교 실패 사례
- **Dataset S1a/b**: 모델 학습 세부 설정, inference 방법, score 계산 방식 등 포함.
- **Table S1–S6**: ablation, optimizer 성능, 하이퍼파라미터 조합별 top-k 성능 등 비교.

---

#### 1.  **Figures**
- **Fig. 1**: Overview of READRetro architecture.  
  - Highlights the dual-module design: single-step prediction (red box) and multi-step planning (black box), combining Retroformer, Graph2SMILES, and retriever outputs.
- **Fig. 2**: Hyperparameter tuning and performance by chemical class.  
  - Shows how batch size, learning rate, and layer layout affect accuracy.
  - Breakdown of pathway hit rates across classes like AA/MA, MVA/MEP, etc.
- **Fig. 3**: Case studies of biosynthetic pathways (tabersonine, menisdaurilide, CBC acid).  
  - Arrows are color-coded: black (generated), red (reaction retriever), blue (pathway retriever).
- **Fig. 4**: Screenshots of the web interface and pathway output visualization.

#### 2.  **Tables**
- **Table 1**: Top-k accuracies for single-step prediction across models (Retroformer, Graph2SMILES, Ensemble, BioNavi-NP).
- **Table 2**: Performance metrics for multi-step retrosynthesis.  
  - Success rate, building block hit rate, and pathway hit rate comparison.  
  - READRetro achieved the highest scores in nearly all categories.

#### 3.  **Appendix / Supplementary Info**
- **Figures S1–S12**:
  - S1: Illustration of SMILES vs. graph input representations.
  - S2: Visualization of chemical space (t-SNE).
  - S3: Tree structure of Retro* search.
  - S4–S9: Pathway predictions for various compounds (benzyl acetone, glucotropaeolin, etc.).
  - S10: List of predefined biosynthetic building blocks.
  - S11–S12: Failure cases of baseline tools vs. READRetro.
- **Datasets S1a/b**: Training configurations, inference mechanics, scoring definitions.
- **Tables S1–S6**: Ablation study results, optimizer comparisons, top-k accuracy by ensemble settings.

---



<br/>
# refer format:     


@article{kim2024readretro,
  title     = {READRetro: natural product biosynthesis predicting with retrieval-augmented dual-view retrosynthesis},
  author    = {Kim, Taein and Lee, Seul and Kwak, Yejin and Choi, Min-Soo and Park, Jeongbin and Hwang, Sung Ju and Kim, Sang-Gyu},
  journal   = {New Phytologist},
  volume    = {243},
  number    = {6},
  pages     = {2512--2527},
  year      = {2024},
  publisher = {Wiley},
  doi       = {10.1111/nph.20012},
  url       = {https://doi.org/10.1111/nph.20012}
}




Taein Kim, Seul Lee, Yejin Kwak, Min-Soo Choi, Jeongbin Park, Sung Ju Hwang, and Sang-Gyu Kim.
“READRetro: Natural Product Biosynthesis Predicting with Retrieval-Augmented Dual-View Retrosynthesis.”
New Phytologist 243, no. 6 (2024): 2512–2527. https://doi.org/10.1111/nph.20012.





