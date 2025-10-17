---
layout: post
title:  "[2025]Vision Transformer Supported Kolmogorov–Arnold Networks for Survival Prediction in Lung Cancer"
date:   2025-10-17 16:28:21 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: - 메서드: VITKAN은 WSI를 224×224 패치로 나눠 사전학습된 ViT로 특징을 추출하고, 임상·오믹스는 Tabular KAN으로 학습한 뒤 Fusion KAN으로 융합하여 TCGA LUAD/LUSC에서 생존예측(C-index) 성능을 평가했다.


짧은 요약(Abstract) :
한글 설명(초록 요약)
- 문제: 폐암은 사망률이 높고, 정확한 생존 예측이 치료 계획에 중요하지만 기존 방법은 조직학적·표형(임상·오믹스) 특성의 복잡성을 충분히 포착하지 못하고, 멀티모달 융합이 비효율적이며 임상 데이터 통합도 부족했습니다.
- 제안: 병리 Whole Slide Image(WSI)를 위한 비전 트랜스포머(ViT)와, 표형 데이터 처리 및 멀티모달 융합을 위한 Kolmogorov–Arnold Networks(KAN)를 결합한 VITKAN을 제안합니다.
- 구성: (1) 오믹스·임상 데이터를 위한 탭형 KAN, (2) WSI 특징 추출을 위한 고도화된 비전 모델, (3) 이미지-표형 표현을 효과적으로 결합하는 새로운 Fusion KAN.
- 결과: VITKAN은 기존 단일모달 예측기보다 폐암 생존 시간 추정 정확도가 유의하게 높았습니다.
- 설명가능성: 생존 예측 기여도 상위 20개 유전자 중 10개가 실제로 암 생존/진행과 관련됨이 문헌으로 검증되어, 모델 해석 가능성을 입증했습니다.
- 의의: 복잡한 조직·표형 신호를 효과적으로 학습하고 융합하여 정확도와 임상적 활용성을 동시에 향상합니다.

English explanation (abstract summary)
- Problem: Lung cancer survival prediction is clinically crucial, yet prior deep learning approaches that fuse WSIs with tabular data often miss the complexity of histologic and tabular features, struggle with effective fusion, and underuse clinical information.
- Proposal: VITKAN combines Vision Transformers for histology with Kolmogorov–Arnold Networks for both tabular processing and multimodal fusion.
- Components: (1) a Tabular KAN for omics and clinical data, (2) advanced vision backbones for WSI feature extraction, and (3) a novel Fusion KAN to integrate image and tabular representations.
- Results: VITKAN outperforms existing unimodal predictors in estimating lung cancer survival.
- Explainability: 10 of the top 20 genes contributing to predictions are validated in the literature as related to cancer survival/progression.
- Significance: The method achieves higher accuracy and improved interpretability by capturing complex cross-modal relationships.


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
아래 설명은 제공된 논문 본문(Methods, Experimental results 등)에 근거하여, 제안 방법(VITKAN)의 모델 구조, 아키텍처적 특징, 데이터/전처리, 학습·평가 및 해석 가능성 기법을 체계적으로 정리한 것입니다.



1) 개요: VITKAN(Vision Transformer supported Kolmogorov–Arnold Networks)
- 목적: TCGA-LUAD/LUSC의 병리 WSIs, 임상, 오믹스(miRNA, CNV) 데이터를 통합해 폐암 생존을 정밀 예측.
- 핵심 아이디어:
  - 병리 이미지는 사전학습된 비전 트랜스포머(ViT) 백본으로 고수준 특징을 추출.
  - 임상/오믹스 탭울러 데이터는 Kolmogorov–Arnold Networks(KAN)를 활용해 유연하고 해석 가능한 방식으로 표현 학습.
  - 서로 다른 표현(이미지+탭울러)을 다시 KAN 기반의 새로운 Fusion KAN으로 융합하여 고차 상호작용을 포착.
- 구성 모듈:
  1) Histology Network(사전학습 ViT 백본)
  2) Tabular KAN(TKAN)
  3) Fusion KAN(FKAN) + 최종 생존 예측 헤드

2) 데이터와 전처리
- 데이터 소스: TCGA-LUAD, TCGA-LUSC의 H&E WSIs, 임상, miRNA, CNV. 원래 1,100명, 누락값 등의 기준으로 927명 최종 사용.
- 모달리티 구성:
  - WSIs: 각 환자당 진단 슬라이드 1장 사용.
  - 임상: 9개 변수(연속형 1: 진단 시 나이; 범주형 8: 성별, 인종, 병기, 과거암, 동시암, 과거치료, 약물치료, 방사선치료).
  - 오믹스: miRNA + CNV 총 3,883개 특성.
- 이미지 패칭: 20× 배율에서 224×224 픽셀 타일로 분할.
- 특징 선택/스케일링:
  - 오믹스 각 특성(예: CNV, miRNA)은 환자 간 분산이 큰 순으로 정렬하여 예측력 향상을 도모.
  - 연속형 mRNA 유사 특성은 [0,1] 정규화.
  - CNV는 범주값을 0부터 시작하는 정수로 표현.
  - 임상 결측: 나이는 중앙값 대치, 나머지 범주형은 “missing” 카테고리 신설(최빈값 대치는 성능 개선 없음). 나이는 [0,1] 정규화.
  - 범주형 임상 변수는 카테고리별 고유 정수로 인코딩.
- 모달리티 불균형 완화:
  - 3,883개 오믹스 vs 9개 임상 특성의 불균형을 완화하기 위해, 동일 환자-슬라이드 수를 유지하면서 두 변형 데이터셋을 만듦:
    1) Omics+Histology
    2) Clinical+Histology
  - LUAD/LUSC 각각에 대해 위 두 변형을 구성하여 총 네 데이터셋(LUAD-Omics, LUAD-Clinical, LUSC-Omics, LUSC-Clinical).
- 학습/검증 분할: 각 데이터셋을 80/20로 분할, 5-fold 교차검증으로 성능 평가.

3) Histology Network: 사전학습 ViT 백본 기반의 병리 특징 추출
- 사용 백본: CTransPath, CONCH, UNI, HistoSSL 등, 병리 전용 사전학습 비전 트랜스포머.
- 입력: 224×224 타일 Xi.
- 출력 임베딩: ViT 최종 레이어에서 D차원 임베딩 zi.
- 차원 축소: 완전연결층(가중치 W∈R(N×D), 바이어스 b∈RN)으로 N차원 임베딩 hi = W·zi + b 생성.
- 역할: hi는 병리 이미지에서 추출된 고수준 표현으로, 이후 Fusion KAN의 입력으로 사용(WSI 히트맵 생성도 가능).

4) Tabular KAN(TKAN): 임상/오믹스 탭울러 데이터 표현 학습
- 배경: MLP는 고차원·소표본 환경에서 과적합과 고정 활성화함수의 제약으로 한계가 있음.
- KAN 원리: Kolmogorov–Arnold 표현정리에 따라, 다변수 연속함수를 단변수 연속함수들의 합성과 합으로 표현하는 구조를 신경망에 도입.
  - 가중치 행렬 대신 에지에 배치된 학습 가능한 단변수 스플라인 함수(φq,p)를 사용.
  - 각 레이어는 단변수 함수들의 행렬 Φ={φq,p}로 구성되며, 레이어 간 합성과 합을 통해 비선형 변환을 누적.
- 장점:
  - 고차원/저차원 탭울러 모두에서 유연하고 해석 가능한 표현 학습.
  - 고차 상호작용을 함수적 형태로 포착.
- 구현:
  - 임상 및 오믹스 입력을 TKAN에 투입하여 탭울러 표현 hg를 생성(차원 N으로 정렬).

5) Fusion KAN(FKAN): 멀티모달 융합
- 입력 결합: 이미지 임베딩 hi ∈ RN과 탭울러 임베딩 hg ∈ RN을 연결(concatenate)하여 hcat ∈ R(2N).
- 네트워크: 여러 층(L layers)의 KAN 변환(Φ0, …, ΦL−1)을 연쇄 합성해 h* = (ΦL−1 ∘ … ∘ Φ0)(hcat)을 생성.
- 목적: 이미지–탭울러 간의 복잡하고 고차원적인 상호작용을 계층적으로 정교화하여 보다 풍부한 융합 표현 획득.
- 출력/헤드: h*를 생존 예측 전용 출력층으로 전달하여 환자별 생존 시간 예측값을 산출.

6) 학습 설정과 평가
- 학습 설정:
  - 프레임워크: PyTorch
  - 하드웨어: NVIDIA RTX 4090
  - 에폭: 최대 100, 조기종료(early stopping) 사용
  - 최적화기: Adam, AdamW 비교
  - 배치 크기: {8, 16, 32}
  - 학습률: [0.0005, 0.0009]
  - KAN 은닉층 수: {1, 2, 6, 10}
  - 하이퍼파라미터 탐색: Weights & Biases로 체계적 서치
- 평가 지표: Concordance index(C-index). 검열을 고려한 쌍대 순서 일치 비율로 예측 생존시간 순위의 정확도를 평가.

7) 해석 가능성 분석
- 기법: Integrated Gradients(IG)
- 대상: 유전자(오믹스) 특성의 기여도(Attribution) 산출
- 목적: 모델이 예측에 크게 사용한 유전자를 식별하고, 양/음 기여를 분리해 생물학적 해석과 임상적 타당성을 높임.
- 결과 개념: 상위 기여 유전자 Top-20을 도출해 문헌(PubMed) 증거와 교차검증.

8) 방법적 요약의 장점
- 병리 전용 ViT 백본으로 이미지 표현력 강화.
- KAN 기반 TKAN/ FKAN으로 탭울러 학습과 멀티모달 융합을 동시에 해석 가능하고 유연하게 수행.
- 데이터 불균형을 고려한 Clinical+Histology, Omics+Histology 이원 평가 설계로 공정 비교 및 강건성 확보.
- IG 기반 해석으로 모델의 생물학적 신뢰성 강화.



1) Overview: VITKAN (Vision Transformer supported Kolmogorov–Arnold Networks)
- Goal: Predict lung cancer survival by fusing WSIs, clinical, and omics (miRNA, CNV) data from TCGA-LUAD/LUSC.
- Key idea:
  - Use pathology-specific pretrained Vision Transformers (ViTs) to extract high-level histology features.
  - Use Kolmogorov–Arnold Networks (KANs) to model clinical/omics tabular data with flexible, interpretable function-based layers.
  - Fuse image and tabular representations with a novel KAN-based fusion network (Fusion KAN) to capture high-order cross-modal interactions.
- Components:
  1) Histology Network (pretrained ViT backbone)
  2) Tabular KAN (TKAN)
  3) Fusion KAN (FKAN) + survival prediction head

2) Data and preprocessing
- Source: TCGA-LUAD and TCGA-LUSC H&E WSIs, clinical variables, miRNA, and CNV. Final cohort of 927 patients after exclusions (from 1,100).
- Modalities:
  - WSIs: one diagnostic slide per patient.
  - Clinical: 9 variables (1 continuous: age at diagnosis; 8 categorical: gender, race, stage, prior malignancy, synchronous malignancy, prior treatment, pharmaceutical treatment, radiation).
  - Omics: 3,883 features (miRNA + CNV).
- WSI patching: 224×224 tiles at 20× magnification.
- Feature selection/scaling:
  - Sort omics features by variance across patients (descending) to strengthen predictive signal.
  - Normalize continuous mRNA-like features to [0,1].
  - Represent CNV categories as integers starting at 0.
  - Handle missing clinical data via median imputation for age and “missing” category for other categorical variables; normalize age to [0,1]; encode categories as integers.
- Balancing modalities:
  - To mitigate 3,883 (omics) vs 9 (clinical) imbalance, create two matched dataset variants per cancer type:
    1) Omics+Histology
    2) Clinical+Histology
  - This yields four datasets: LUAD-Omics, LUAD-Clinical, LUSC-Omics, LUSC-Clinical.
- Splits: 80/20 train/test, 5-fold cross-validation for robust evaluation.

3) Histology Network: pretrained ViT backbones for pathology
- Backbones: CTransPath, CONCH, UNI, HistoSSL (pathology-specific pretrained vision transformers).
- Input: 224×224 tile Xi.
- ViT output: D-dimensional embedding zi from final layer.
- Dimensionality reduction: a fully connected layer (W∈R(N×D), b∈RN) yields N-dimensional hi = W·zi + b.
- Role: hi is the high-level histology representation; it feeds the Fusion KAN (WSI heatmaps can also be produced).

4) Tabular KAN (TKAN): learning from clinical/omics
- Motivation: MLPs struggle with high-dimensional tabular data and limited samples due to fixed activations and dense connections.
- KAN principle: Based on the Kolmogorov–Arnold representation theorem, replace weight matrices with learnable univariate spline functions (φq,p) on edges; each layer is a matrix of univariate functions Φ, and layers compose nonlinearly.
- Advantages:
  - Flexible and interpretable modeling for both high- and low-dimensional tabular inputs.
  - Naturally captures complex and high-order interactions.
- Implementation:
  - Feed clinical and omics inputs to TKAN to obtain an N-dimensional tabular representation hg.

5) Fusion KAN (FKAN): multimodal fusion
- Input composition: Concatenate hi ∈ RN and hg ∈ RN to form hcat ∈ R(2N).
- Network: Apply a stack of L KAN layers (Φ0, …, ΦL−1) to compute h* = (ΦL−1 ∘ … ∘ Φ0)(hcat).
- Purpose: Iteratively refine and enrich cross-modal representations, capturing high-order interactions between histology and tabular features.
- Output/head: Pass h* to the survival prediction head to estimate patient-specific survival times.

6) Training protocol and evaluation
- Implementation:
  - Framework: PyTorch
  - Hardware: NVIDIA RTX 4090
  - Epochs: up to 100, with early stopping
  - Optimizers: Adam and AdamW
  - Batch sizes: {8, 16, 32}
  - Learning rates: [0.0005, 0.0009]
  - KAN hidden layers: {1, 2, 6, 10}
  - Hyperparameter search: systematic sweeps via Weights & Biases
- Metric: Concordance index (C-index), measuring pairwise ordering agreement of predicted vs. true survival times under censoring.

7) Interpretability
- Method: Integrated Gradients (IG)
- Target: Attribution scores for each gene to quantify their positive/negative contributions to survival prediction.
- Outcome: Identify top-contributing genes (Top-20) and corroborate with the literature (PubMed), thereby increasing biological and clinical trustworthiness.

8) Methodological strengths (summary)
- Strong histology representations via pathology-specific ViT backbones.
- KAN-based TKAN/FKAN provide flexible, interpretable learning for tabular data and fusion, capturing high-order interactions.
- Balanced evaluation across Clinical+Histology and Omics+Histology variants enhances fairness and robustness.
- IG-based analysis offers transparent gene-level insights aligned with published evidence.


<br/>
# Results
아래 내용은 논문 본문에 제시된 실험 설정과 결과(경쟁모델, 테스트 데이터, 메트릭, 비교)만을 정리·해설한 것입니다.

1) 테스트 데이터와 설정
- 프로젝트/질환: TCGA-LUAD(폐선암), TCGA-LUSC(편평상피암)
- 모달리티
  - 병리 WSIs(진단 슬라이드, H&E), 임상(9개 변수), 오믹스(총 3,883개 특징: miRNA + CNV)
- 환자 수: 원자료 1,100명 중 결측된 환자를 제외하여 최종 927명 사용
- 데이터셋 변형(모달 조합)
  - Clinical+Histology: 임상 + WSI
  - Omics+Histology: 오믹스 + WSI
  - 두 질환(LUAD, LUSC) 각각에 대해 위 2종 구성 → 총 4개 데이터셋
- 전처리(요지)
  - WSI: 20× 배율에서 224×224 패치 추출
  - 임상: 나이(연속)는 중앙값 대치 후 [0,1] 정규화, 범주형은 ‘missing’ 포함해 정수 인코딩
  - 오믹스: 환자 간 분산이 큰 순으로 정렬, mRNA 계열 [0,1] 정규화, CNV 범주형은 정수로 매핑
- 분할/학습
  - Train/Test = 80%/20%
  - 5-fold 교차검증, 최대 100 epoch, Early stopping 사용
  - 하이퍼파라미터 탐색: 배치{8,16,32}, 학습률[5e-4, 9e-4], KAN 은닉층{1,2,6,10}, 옵티마이저 Adam/AdamW
  - 구현: PyTorch, NVIDIA RTX 4090, 코드 공개(https://github.com/mertgokpinar/VITKAN)

2) 평가 메트릭
- Concordance index(C-index)
  - 검열되지 않은 샘플 쌍에서 실제 생존시간의 순서를 모델이 올바르게 예측한 비율
  - 값이 높을수록 순서 예측(위험도 추정)이 정확함

3) 경쟁 모델(비교 대상)
- 다중모달 생존예측 SOTA 2종과 비교
  - Porpoise (Chen et al., 2022)
  - Pathomic Fusion (Chen et al., 2020)
- 추가 내부 비교(모듈 단위)
  - Histology 백본(WSI 특징추출): CTransPath, CONCH, UNI, HistoSSL
  - Tabular 모델: Tabular KAN vs FC+MHA(완전연결+멀티헤드어텐션)
  - Fusion 방법: Fusion KAN vs Self-Attention 기반 vs Kronecker product

4) 정량 결과 요약(핵심 수치 포함)
A. VITKAN vs Porpoise/Pathomic Fusion(전체 성능)
- LUAD, Clinical+Histology
  - VITKAN 0.6473 vs Porpoise 0.5245 vs Pathomic 0.5179
- LUSC, Clinical+Histology
  - VITKAN 0.6122 vs Porpoise 0.5333 vs Pathomic 0.5146
- LUAD, Omics+Histology
  - VITKAN 0.6357 vs Porpoise 0.6021 vs Pathomic 0.5236
- LUSC, Omics+Histology
  - VITKAN 0.5954 vs Porpoise 0.5239 vs Pathomic 0.5021
해설
- VITKAN이 두 질환과 두 데이터셋 변형 모두에서 일관되게 최고 C-index를 기록.
- LUAD에서 경쟁모델 대비 격차가 더 큼(유전적 요인 연관이 상대적으로 큰 질환 특성에 기인한다고 논문은 해석).

B. Histology 백본 비교(WSI 특징추출)
- LUAD
  - Clinical+Histology: CTransPath 0.6473(최고), CONCH 0.6412, UNI 0.6272, HistoSSL 0.5850
  - Omics+Histology: CTransPath 0.6357(최고), CONCH 0.6166, UNI 0.6030, HistoSSL 0.5789
- LUSC
  - Clinical+Histology: CONCH 0.6122(최고), UNI 0.5892, HistoSSL 0.5888, CTransPath 0.5692
  - Omics+Histology: CTransPath 0.5954(최고), CONCH 0.5911, HistoSSL 0.5865, UNI 0.5236
해설
- CTransPath와 CONCH가 전반적으로 상위권. 
- LUAD에선 CTransPath가 양 변형 모두 최상.
- LUSC에선 Clinical+Histology에서 CONCH가 최고, Omics+Histology에서는 CTransPath가 최고.
- CONCH는 4개 실험 전반에서 안정적으로 상위 2위를 유지, 가장 “견고”한 백본으로 평가.

C. Tabular 모델 비교(임상/오믹스 처리)
- Tabular KAN vs FC+MHA
  - LUAD Clinical: 0.647 vs 0.600
  - LUAD Omics: 0.635 vs 0.581
  - LUSC Clinical: 0.612 vs 0.588
  - LUSC Omics: 0.595 vs 0.586
해설
- Tabular KAN이 모든 조건에서 FC+MHA보다 우수.
- 고차원/복잡한 탭울라 데이터를 다루는 데 KAN의 가변 활성함수(스플라인 기반)가 효과적임을 시사.

D. Fusion 방법 비교(모달 융합)
- Fusion KAN vs Self-Attention vs Kronecker
  - LUAD Clinical: 0.647 vs 0.559 vs 0.600
  - LUAD Omics: 0.635 vs 0.569 vs 0.581
  - LUSC Clinical: 0.612 vs 0.574 vs 0.588
  - LUSC Omics: 0.595 vs 0.557 vs 0.586
해설
- Fusion KAN이 모든 셋업에서 최고 성능.
- Kronecker는 일관된 2위, Self-Attention은 가장 낮음.
- KAN 기반 융합이 임상/오믹스와 조직 이미지 간의 고차 상호작용을 더 잘 포착함.

5) 추가 관찰(논문 내 해석)
- Clinical+Histology가 Omics+Histology보다 대체로 더 높은 C-index를 보임. 논문은 오믹스의 초고차원성 때문에 심층모델이 의미 있는 신호를 안정적으로 학습하기 어렵고, 임상은 단순하고 직접적인 예후 신호가 많다는 점을 원인으로 제시.
- LUAD가 유전적 요인 연관성이 높고, LUSC는 흡연 등 외부요인 영향이 커서(자료 상) 예후 추정 난도가 다르다는 해석을 제시.
- 정성 분석(Integrated Gradients)에서 상위 20개 기여 유전자 중 10개는 문헌적으로 암/폐암 관련성이 확인됨(테이블 V). 이 부분은 모델 해석가능성(설명가능성) 측면의 결과.

정리
- 경쟁모델 대비: VITKAN이 LUAD/LUSC, Clinical/Omics 전 실험에서 최고 C-index.
- 핵심 기여 모듈: Tabular KAN(탭울라 처리), Fusion KAN(모달 융합)이 성능 향상을 주도.
- Histology 백본: CTransPath와 CONCH가 특히 강력하며, 질환·데이터셋에 따라 최적 백본이 다를 수 있음.
- 실험 설계: 공정한 분할(5-fold CV, 80/20), 조기종료, 체계적 하이퍼파라미터 탐색으로 재현 가능성 제고.



Below is a structured summary limited to what is reported in the paper (competitors, test data, metric, and comparisons).

1) Test data and setup
- Projects/diseases: TCGA-LUAD (adenocarcinoma), TCGA-LUSC (squamous cell carcinoma)
- Modalities
  - Histology: diagnostic H&E WSIs
  - Clinical: 9 variables
  - Omics: 3,883 features (miRNA + CNV)
- Cohort: 927 patients after excluding samples with missing censoring/clinical data (from an initial 1,100)
- Dataset variants (per disease)
  - Clinical+Histology
  - Omics+Histology
  - Total of 4 datasets (LUAD/LUSC × Clinical/Omics)
- Preprocessing (high level)
  - WSIs tiled at 20× into 224×224 patches
  - Clinical: median imputation for age + [0,1] scaling; categorical with “missing” category then integer-encoded
  - Omics: features sorted by variance across patients; mRNA-like features scaled to [0,1]; CNV as integer-coded categories
- Splits/training
  - Train/Test = 80%/20%, 5-fold cross-validation, up to 100 epochs, early stopping
  - Hyperparameters: batch {8,16,32}, LR [5e-4, 9e-4], KAN hidden layers {1,2,6,10}, Adam/AdamW
  - Implementation: PyTorch, NVIDIA RTX 4090, code at https://github.com/mertgokpinar/VITKAN

2) Evaluation metric
- Concordance index (C-index): fraction of correctly ordered uncensored pairs by predicted survival/risk; higher is better.

3) Competitors
- Multimodal SOTA baselines
  - Porpoise (Chen et al., 2022)
  - Pathomic Fusion (Chen et al., 2020)
- Additional internal ablations
  - Histology backbones: CTransPath, CONCH, UNI, HistoSSL
  - Tabular models: Tabular KAN vs FC+MHA
  - Fusion: Fusion KAN vs Self-Attention vs Kronecker product

4) Key quantitative results (with numbers)
A. Overall comparison: VITKAN vs Porpoise/Pathomic Fusion
- LUAD, Clinical+Histology
  - VITKAN 0.6473 vs Porpoise 0.5245 vs Pathomic 0.5179
- LUSC, Clinical+Histology
  - VITKAN 0.6122 vs Porpoise 0.5333 vs Pathomic 0.5146
- LUAD, Omics+Histology
  - VITKAN 0.6357 vs Porpoise 0.6021 vs Pathomic 0.5236
- LUSC, Omics+Histology
  - VITKAN 0.5954 vs Porpoise 0.5239 vs Pathomic 0.5021
Interpretation
- VITKAN consistently achieves the best C-index across both cancers and both dataset variants.
- The performance gap is larger for LUAD, which the paper attributes to stronger genetic associations in LUAD.

B. Histology backbone comparison (WSI feature extraction)
- LUAD
  - Clinical+Histology: CTransPath 0.6473 (best), followed by CONCH 0.6412, UNI 0.6272, HistoSSL 0.5850
  - Omics+Histology: CTransPath 0.6357 (best), followed by CONCH 0.6166, UNI 0.6030, HistoSSL 0.5789
- LUSC
  - Clinical+Histology: CONCH 0.6122 (best), followed by UNI 0.5892, HistoSSL 0.5888, CTransPath 0.5692
  - Omics+Histology: CTransPath 0.5954 (best), followed by CONCH 0.5911, HistoSSL 0.5865, UNI 0.5236
Interpretation
- CTransPath and CONCH are the top performers overall.
- CTransPath excels for LUAD; for LUSC, CONCH leads on Clinical+Histology while CTransPath leads on Omics+Histology.
- CONCH is the most robust backbone (top-2 across all four experiments).

C. Tabular modeling (clinical/omics)
- Tabular KAN vs FC+MHA
  - LUAD Clinical: 0.647 vs 0.600
  - LUAD Omics: 0.635 vs 0.581
  - LUSC Clinical: 0.612 vs 0.588
  - LUSC Omics: 0.595 vs 0.586
Interpretation
- Tabular KAN outperforms FC+MHA in all settings, indicating strong suitability of KANs for high-dimensional tabular data.

D. Fusion mechanism (modality integration)
- Fusion KAN vs Self-Attention vs Kronecker
  - LUAD Clinical: 0.647 vs 0.559 vs 0.600
  - LUAD Omics: 0.635 vs 0.569 vs 0.581
  - LUSC Clinical: 0.612 vs 0.574 vs 0.588
  - LUSC Omics: 0.595 vs 0.557 vs 0.586
Interpretation
- Fusion KAN is best in all cases; Kronecker is second; Self-Attention is lowest.
- KAN-based fusion more effectively captures high-order cross-modal interactions.

5) Additional observations (from the paper)
- Clinical+Histology tends to outperform Omics+Histology; the paper attributes this to the extreme dimensionality of omics and the more direct prognostic signals in clinical variables.
- Differences between LUAD and LUSC likely reflect underlying biology (genetic vs environmental drivers).
- Qualitative analysis via Integrated Gradients finds that 10 of the top-20 genes (per cancer) have literature support for roles in cancer, indicating explainability of VITKAN’s predictions.

In summary
- Against strong multimodal baselines, VITKAN achieves the highest C-index across cancers and modality combinations.
- The Tabular KAN and Fusion KAN modules are the main contributors to the gains.
- Among histology backbones, CTransPath and CONCH are particularly effective; the best choice can be dataset-specific.
- The experimental design (5-fold CV, early stopping, hyperparameter search) supports robust evaluation and reproducibility.


<br/>
# 예제
아래 예시는 본 논문(VITKAN: Vision Transformer supported Kolmogorov–Arnold Networks)을 그대로 따르는 데이터 구성, 전처리, 태스크 정의, 입출력 형식, 학습/평가 절차를 환자 단위로 구체화한 것입니다. 실제 환자 식별자는 가상의 예를 사용합니다.

1) 전체 개요
- 목표: 폐암(LUAD, LUSC) 환자의 생존시간을 정확히 예측
- 입력: 병리 Whole Slide Image(WSI) + 임상 데이터(9개 변수) + 오믹스 데이터(miRNA 연속형, CNV 범주형; 총 3,883 특징)
- 모델: 사전학습 ViT 기반 병리 특징 추출기 + Tabular KAN(임상/오믹스) + Fusion KAN(융합)
- 출력: 환자 단위의 예측 생존시간(실수값). 평가 지표는 C-index
- 데이터 분할: 프로젝트-모달리티별 4개 세트(LUAD-Clinical, LUAD-Omics, LUSC-Clinical, LUSC-Omics), 각 80/20 학습·테스트 분할 + 5-fold CV
- 하이퍼파라미터 탐색: 배치 {8,16,32}, lr ∈ [5e-4, 9e-4], KAN 은닉층 {1,2,6,10}, 옵티마이저 Adam/AdamW, 최대 100 epoch + early stopping

2) 입력 데이터 구체 형식과 전처리
2-1) Histology(WSI)
- 원본: TCGA LUAD/LUSC의 진단용 H&E WSI(환자당 1장)
- 패치 생성: 배율 20×, 타일 크기 224×224 RGB
- 예: 환자 1장의 슬라이드 → 유효 타일 1,200개 (환자마다 200~2,000개 수준 가변)
- ViT 백본(사전학습 병리 모델 중 선택): CTransPath, CONCH, UNI, HistoSSL
- ViT 최종 레이어 임베딩: z_i ∈ R^D (패치 i)
- 선형 축소: h_i = W z_i + b, h_i ∈ R^N
- 슬라이드 단일 벡터로 집약: 모든 패치의 {h_i}를 평균 또는 어텐션 가중합으로 풀링하여 h_img ∈ R^N 생성
  참고: 본문은 패치 임베딩 계산을 수식으로 명시하고, 그림 1에는 히트맵/어텐션 개념이 표현됩니다. 구현에서는 평균/어텐션 풀링 중 하나로 슬라이드 단일 벡터를 만듭니다.

2-2) 임상 데이터(Clinical, 9개 변수)
- 연속형: age_at_diagnosis → 결측은 중앙값 대치 후 [0,1] 정규화
- 범주형(8개): Gender, Race, Tumor Stage, Prior Malignancy, Synchronous Malignancy, Prior Treatment, Pharmaceutical Treatment, Radiation Treatment
  - 결측 처리: “missing”이라는 새로운 카테고리 추가(최빈값 대치는 성능 향상 없음)
  - 인코딩: 각 카테고리를 고유 정수로 매핑(원-핫 대신 정수 매핑 사용)
- 하나의 환자 임상 벡터 예: x_clin ∈ R^9 (실제로는 연속 1 + 범주 8을 정수/정규화 값으로 포함)

2-3) 오믹스 데이터(Omics)
- miRNA: 연속형, [0,1] 정규화
- CNV: 범주형, 카테고리 레벨을 0부터 시작하는 정수로 표현
- 특징 선택: 환자 전체 분산 기준 내림차순 정렬(본 연구는 정렬을 사용하며, 총 3,883개의 특징 사용)
- 하나의 환자 오믹스 벡터 예: x_omics ∈ R^3883

2-4) 레이블(생존)
- OS_time: 관찰 생존시간(예: 일 단위)
- OS_event: 이벤트 발생 여부(1=사망, 0=우측 검열)
- 학습 출력은 예측 생존시간 \hat{T} (실수값). 평가(C-index)는 비검열 샘플 쌍을 기준으로 산출

3) 태스크별 트레이닝/테스트 입력·출력 예시
3-1) 태스크 A: LUAD-Clinical (WSI + 임상)
- 입력(환자 단위)
  - Histology: 슬라이드 타일 텐서 [P, 3, 224, 224], P=유효 타일 수(예: 1,200)
  - Clinical: x_clin ∈ R^9
  - 레이블: OS_time=965, OS_event=1
- 출력
  - \hat{T} ∈ R (예: \hat{T}=1,120일)
- 학습 배치 예(배치=16)
  - 타일: 16개 환자 × (각 P_k 타일) → ViT → {h_i} → 슬라이드 풀링 → h_img(k) ∈ R^N
  - 임상: 16×9 텐서 → Tabular KAN → h_tab(k) ∈ R^N
  - 융합: h_cat = [h_img || h_tab] ∈ R^{2N} → Fusion KAN → h* ∈ R^N → MLP → \hat{T}
- 손실/최적화
  - lr=7e-4, AdamW, 최대 100 epoch, early stopping(검증 C-index 모니터링)
- 테스트
  - 고정된 전처리/모델로 \hat{T} 추론 → C-index 계산

3-2) 태스크 B: LUAD-Omics (WSI + 오믹스)
- 입력(환자 단위)
  - Histology: [P, 3, 224, 224]
  - Omics: x_omics ∈ R^3883 (miRNA[0,1], CNV=정수 카테고리)
  - 레이블: OS_time, OS_event
- 출력: \hat{T} ∈ R
- 파이프라인: 임상 대신 오믹스가 Tabular KAN 입력으로 들어감. 나머지는 동일

3-3) 태스크 C: LUSC-Clinical / LUSC-Omics
- LUAD과 동일한 구성, 단 프로젝트가 LUSC로 변경

3-4) 태스크 D(어블레이션): 단일모달 학습
- Histology only: h_img → 예측
- Tabular only(Clinical 또는 Omics): h_tab → 예측
- 목적: 융합 이점 및 KAN/백본 효과 확인

4) 예시 레코드(가상)
- 환자 ID: TCGA-LUAD-001
- 슬라이드: /slides/TCGA-LUAD-001.svs → 타일 1,048개
- 임상(전처리 후)
  - age_at_diagnosis: 67세 → 0.74
  - Gender: male → 1
  - Race: white → 2
  - Tumor Stage: stage II → 2
  - Prior Malignancy: missing → 0
  - Synchronous Malignancy: no → 0
  - Prior Treatment: yes → 1
  - Pharmaceutical Treatment: yes → 1
  - Radiation Treatment: no → 0
- 오믹스(일부만 예시)
  - miRNA: hsa-miR-21=0.84, hsa-miR-155=0.31, …
  - CNV: EGFR=3, KRAS=2, TP53=1, … (정수 카테고리)
- 레이블
  - OS_time_days: 965
  - OS_event: 1

5) 미니배치 포워드 예시(한 환자)
- 타일 → ViT(CTransPath 권장) → z_i ∈ R^D
- 선형 축소: h_i ∈ R^N
- 슬라이드 풀링: h_img = mean_i(h_i) ∈ R^N
- Tabular KAN(임상 또는 오믹스): h_tab ∈ R^N
- 융합: h_cat=[h_img||h_tab] ∈ R^{2N} → Fusion KAN → h* ∈ R^N
- 예측: \hat{T}=MLP(h*) ∈ R

6) 평가 지표(C-index) 계산 예시(소규모 예)
- 실제(OS_time): [965, 700, 500], 이벤트=[1,1,0]
- 예측(\hat{T}): [1120, 650, 800]
- 비검열 쌍만 고려 → (1번,2번) 한 쌍
  - 실제 순서: 700 < 965 → ψ(T2,T1)=1
  - 예측 순서: 650 < 1120 → ψ(\hat{T2},\hat{T1})=1
  - 일치 → C-index=1.0 (예시)

7) 성능 기대치(논문 결과 요약)
- 멀티모달(WSI+Clinical)
  - LUAD: C-index ≈ 0.6473
  - LUSC: C-index ≈ 0.6122
- 멀티모달(WSI+Omics)
  - LUAD: C-index ≈ 0.6357
  - LUSC: C-index ≈ 0.5954
- 히스토로지 백본 비교(평균 C-index)
  - CTransPath, CONCH가 대체로 상위권
- 탭ular 모델 비교
  - Tabular KAN > FC+MHA (임상·오믹스 모두)
- 융합 전략 비교
  - Fusion KAN > Kronecker > Self-Attention

8) 해석 가능성(선택적 분석 예시)
- IG(Integrated Gradients)로 유전자 기여도 산출
- 베이스라인(0 벡터)→실제 입력까지 적분 경사
- 상위 20개 유전자 리스트업, 문헌 교차검증(논문 표 V에서는 LUAD/LUSC 각각 상위 20 중 10개가 암 연관 근거 확인)

9) 학습 설정 샘플
- 백본: CTransPath(또는 CONCH)
- N(축소 차원): 예: 256
- 배치: 16
- 옵티마이저: AdamW, lr=7e-4, weight decay=1e-4
- 에폭: 최대 100, early stopping(patience=10, 모니터=val C-index)
- 크로스밸리데이션: 5-fold, 환자 단위 분할 유지
- 구현: PyTorch, GPU: RTX 4090

10) 테스트 단계 입출력 요약
- 입력(환자):
  - [P,3,224,224] 타일 텐서
  - 임상 9개 또는 오믹스 3,883개 벡터
- 출력(환자):
  - \hat{T} (실수, 예측 생존시간)
- 리포트:
  - 테스트 집합 C-index, 백본/탭ular/융합별 비교
  - 필요시 상위 기여 유전자 IG 분석 결과





The following examples concretize how to prepare inputs/outputs, define tasks, and run training/testing in the exact spirit of the VITKAN paper.

1) Overview
- Goal: Predict time-to-event survival for lung cancer (LUAD, LUSC) patients.
- Inputs: Pathology WSIs + clinical tabular data (9 vars) + omics (miRNA continuous, CNV categorical; total 3,883 features).
- Model: Pretrained ViT histology backbone + Tabular KAN (for clinical/omics) + Fusion KAN (for multimodal integration).
- Output: Patient-level predicted survival time (a real-valued scalar). Evaluation via C-index.
- Data split: Four datasets (LUAD-Clinical, LUAD-Omics, LUSC-Clinical, LUSC-Omics). 80/20 train/test and 5-fold CV.
- Hyperparameters: batch {8,16,32}, lr ∈ [5e-4, 9e-4], KAN hidden layers {1,2,6,10}, Adam/AdamW, up to 100 epochs with early stopping.

2) Inputs and preprocessing
2-1) Histology (WSI)
- Source: TCGA LUAD/LUSC diagnostic H&E WSIs (one per patient).
- Patching: 20× magnification, 224×224 RGB tiles.
- Example: one slide → ~1,200 valid tiles (range ~200–2,000).
- ViT backbones (pretrained for pathology): CTransPath, CONCH, UNI, HistoSSL.
- ViT final embedding per patch: z_i ∈ R^D.
- Linear reduction: h_i = W z_i + b, h_i ∈ R^N.
- Slide-level pooling: aggregate {h_i} via mean or attention-weighted pooling to get h_img ∈ R^N.
  Note: The paper gives the patch-level formulation and Figure 1 hints attention/heatmaps. In practice, you pool to a single slide vector.

2-2) Clinical (9 variables)
- Continuous: age_at_diagnosis → median imputation, min-max to [0,1].
- Categorical (8): Gender, Race, Tumor Stage, Prior Malignancy, Synchronous Malignancy, Prior Treatment, Pharmaceutical Treatment, Radiation Treatment.
  - Missing handling: add a “missing” category (mode imputation did not help).
  - Encoding: map each category to a unique integer (no one-hot per paper).
- A patient clinical vector: x_clin ∈ R^9 (1 normalized continuous + 8 integer-coded categorical).

2-3) Omics
- miRNA: continuous, min-max to [0,1].
- CNV: categorical, integer-coded starting at 0.
- Feature ordering: sort by variance across patients (paper uses sorting; the full 3,883 features are used).
- A patient omics vector: x_omics ∈ R^3883.

2-4) Labels (survival)
- OS_time: observed time-to-event (e.g., days).
- OS_event: event indicator (1=death, 0=censored).
- The model predicts \hat{T} (a real-valued time). C-index is computed over permissible uncensored pairs.

3) Task-specific train/test I/O examples
3-1) Task A: LUAD-Clinical (WSI + Clinical)
- Input (per patient)
  - Histology: tiles tensor [P, 3, 224, 224] (e.g., P=1,200).
  - Clinical: x_clin ∈ R^9.
  - Label: OS_time=965, OS_event=1.
- Output
  - \hat{T} ∈ R (e.g., \hat{T}=1,120 days).
- Training batch example (batch=16)
  - Tiles: 16 patients × (their P_k tiles) → ViT → {h_i} → slide pooling → h_img(k) ∈ R^N.
  - Clinical: 16×9 → Tabular KAN → h_tab(k) ∈ R^N.
  - Fusion: h_cat=[h_img||h_tab] ∈ R^{2N} → Fusion KAN → h* ∈ R^N → MLP → \hat{T}.
- Optimization
  - lr=7e-4, AdamW, up to 100 epochs, early stopping on validation C-index.
- Testing
  - Infer \hat{T} for test patients → compute C-index.

3-2) Task B: LUAD-Omics (WSI + Omics)
- Input (per patient)
  - Histology: [P, 3, 224, 224].
  - Omics: x_omics ∈ R^3883 (miRNA in [0,1], CNV integers).
  - Label: OS_time, OS_event.
- Output: \hat{T} ∈ R.
- Same pipeline as Task A, but Tabular KAN consumes omics instead of clinical.

3-3) Task C: LUSC-Clinical / LUSC-Omics
- Same as LUAD tasks but using the LUSC cohort.

3-4) Task D (ablation): Unimodal training
- Histology only: h_img → predictor.
- Tabular only (Clinical or Omics): h_tab → predictor.
- Purpose: quantify gains from fusion and the contribution of KAN/backbones.

4) Example record (mock)
- patient_id: TCGA-LUAD-001
- slide: /slides/TCGA-LUAD-001.svs → 1,048 tiles
- clinical (post-processed)
  - age_at_diagnosis: 67 → 0.74
  - Gender: male → 1
  - Race: white → 2
  - Tumor Stage: stage II → 2
  - Prior Malignancy: missing → 0
  - Synchronous Malignancy: no → 0
  - Prior Treatment: yes → 1
  - Pharmaceutical Treatment: yes → 1
  - Radiation Treatment: no → 0
- omics (partial illustration)
  - miRNA: hsa-miR-21=0.84, hsa-miR-155=0.31, …
  - CNV: EGFR=3, KRAS=2, TP53=1, …
- labels
  - OS_time_days: 965
  - OS_event: 1

5) Mini-batch forward (single patient)
- Tiles → ViT (e.g., CTransPath) → z_i ∈ R^D
- Linear reduction: h_i ∈ R^N
- Slide pooling: h_img = mean_i(h_i) ∈ R^N
- Tabular KAN (clinical or omics): h_tab ∈ R^N
- Fusion: h_cat=[h_img||h_tab] ∈ R^{2N} → Fusion KAN → h* ∈ R^N
- Prediction: \hat{T} = MLP(h*) ∈ R

6) C-index toy calculation
- True times: [965, 700, 500], events=[1,1,0]
- Predictions: [1120, 650, 800]
- Consider uncensored pair (1,2) only:
  - True order: 700 < 965 → ψ(T2,T1)=1
  - Pred order: 650 < 1120 → ψ(\hat{T2},\hat{T1})=1
  - Match → C-index=1.0 (toy)

7) Expected performance (from paper)
- Multimodal (WSI+Clinical)
  - LUAD: C-index ≈ 0.6473
  - LUSC: C-index ≈ 0.6122
- Multimodal (WSI+Omics)
  - LUAD: C-index ≈ 0.6357
  - LUSC: C-index ≈ 0.5954
- Histology backbones
  - CTransPath and CONCH are generally strongest.
- Tabular models
  - Tabular KAN > FC+MHA.
- Fusion strategies
  - Fusion KAN > Kronecker > Self-Attention.

8) Interpretability (optional gene attribution)
- Use Integrated Gradients with a zero baseline.
- Integrate gradients from baseline to input for all gene features.
- Rank genes by absolute attribution; top-20 genes per cohort; cross-check with literature.

9) Training setup sample
- Backbone: CTransPath (or CONCH)
- N (reduced dim): e.g., 256
- Batch: 16
- Optimizer: AdamW, lr=7e-4, weight decay=1e-4
- Epochs: up to 100, early stopping (patience=10, monitor val C-index)
- CV: 5-fold, patient-level split
- Framework: PyTorch, GPU: RTX 4090

10) Test-time I/O summary
- Input (per patient):
  - [P,3,224,224] tiles
  - 9-D clinical or 3,883-D omics vector
- Output (per patient):
  - \hat{T} (real-valued predicted survival time)
- Report:
  - Test-set C-index and comparisons by backbone/tabular/fusion choice
  - Optional IG-based top-gene analysis

<br/>
# 요약
- 메서드: VITKAN은 WSI를 224×224 패치로 나눠 사전학습된 ViT로 특징을 추출하고, 임상·오믹스는 Tabular KAN으로 학습한 뒤 Fusion KAN으로 융합하여 TCGA LUAD/LUSC에서 생존예측(C-index) 성능을 평가했다.
- 결과: VITKAN은 Porpoise·Pathomic Fusion 대비 더 높은 C-index를 보였으며(예: LUAD 임상+조직 0.6473 vs 0.5245/0.5179, 오믹스+조직 0.6357 vs 0.6021/0.5236), 조직 백본은 CTransPath/CONCH가 우수했고 Tabular KAN은 FC+MHA보다, Fusion KAN은 Self-Attention·Kronecker보다 일관되게 우수했다.
- 예시: IG 기반 설명가능성 분석에서 상위 20개 유전자 중 10개가 문헌으로 암 관련성이 확인되었고(LUAD: CD160, PARP2, ENSA 등; LUSC: GPR87, SMCP, UBR5 등), LUAD와 LUSC의 기여 분포 차이도 관찰됐다.

- Methods: VITKAN splits WSIs into 224×224 patches for ViT-based histology features, models clinical/omics with a Tabular KAN, and fuses them via a Fusion KAN, evaluated on TCGA LUAD/LUSC with C-index.
- Results: VITKAN outperformed Porpoise and Pathomic Fusion (e.g., LUAD clinical+histology 0.6473 vs 0.5245/0.5179; omics+histology 0.6357 vs 0.6021/0.5236); CTransPath/CONCH were the best histology backbones, Tabular KAN beat FC+MHA, and Fusion KAN surpassed self-attention and Kronecker fusion.
- Examples: Integrated Gradients highlighted top-20 genes with 10 validated by literature (LUAD: CD160, PARP2, ENSA; LUSC: GPR87, SMCP, UBR5), and revealed distinct attribution patterns between LUAD and LUSC, supporting explainability.

<br/>
# 기타
‘기타(다이어그램, 피규어, 테이블, 어펜딕스 등)’의 결과·인사이트 중심 요약입니다.

피규어(다이어그램)
- Figure 1: VITKAN 워크플로우
  - 결과: WSI를 224×224(20×) 패치로 분할 → 병리 전용 사전학습 ViT로 히스토리 임베딩 추출 → 임상·오믹스는 Tabular KAN(TKAN)로 처리 → 이미지·탭형 임베딩을 Fusion KAN(FKAN)으로 통합 → 생존예측 및 IG 기반 유전자 기여도 해석.
  - 인사이트: 병리 특화 ViT와 KAN 기반 탭형/융합 모듈의 조합으로 모달리티별 복잡한 패턴과 고차 상호작용을 동시에 포착. 설계적으로 해석 가능성을 내장(유전자 중요도). 모듈러 구조라 백본과 탭형/융합 모듈을 독립적으로 최적화 가능.

- Figure 2: 유전자 기여도(Integrated Gradients) 분포
  - 결과:
    - 대부분 유전자는 기여도가 0에 근접. 극소수만 생존예측에 크게 기여.
    - LUAD: 양/음 기여가 거의 대칭, 상위 유전자 기여도 절대값이 LUSC보다 큼, 상위 20개 완만한 감소.
    - LUSC: 양의 기여가 약간 우세, 상위 몇 개 유전자에서 급격히 떨어진 뒤 완만.
    - LUAD·LUSC 간 상위 20개 유전자 집합은 상호 배타적.
  - 인사이트: 아형별로 예측에 관여하는 유전자 집합과 조절 방향(상향/하향)이 다름. LUAD는 유전적 요인의 영향이 더 크고(LUSC 대비 상위 기여도 더 큼), LUSC는 다수 유전자가 비교적 고르게 기여. 모델이 상·하향 조절의 방향성도 분리해 학습.

테이블
- Table I: 멀티모달 기법 비교(VITKAN vs Porpoise vs Pathomic Fusion)
  - 결과: LUAD/LUSC의 Clinical+Histology, Omics+Histology 모든 설정에서 VITKAN C-Index 최고(예: LUAD-Clinical 0.6473). 특히 LUAD에서 격차가 큼.
  - 인사이트: VITKAN의 융합과 표현학습이 기존 멀티모달 SOTA 대비 일관되게 우수. 아형과 데이터 구성(임상/오믹스) 변화에도 성능 강건.

- Table II: 병리 히스토리 백본 비교(CTransPath, CONCH, UNI, HistoSSL)
  - 결과: 병리 특화 ViT가 일관되게 우수. CTransPath가 LUAD(Clinical/Omic)와 LUSC(Omic)에서 최고, CONCH가 LUSC(Clinical)에서 최고. UNI/HistoSSL은 전반적으로 열세.
  - 인사이트: 백본 선택이 성능에 큰 영향. 일반 도메인(ImageNet) 대비 병리 도메인 사전학습이 생존예측 특성 추출에 유리. CTransPath와 CONCH가 가장 견고.

- Table III: 탭형 네트워크 비교(Tabular KAN vs FC+MHA)
  - 결과: 모든 설정에서 Tabular KAN이 FC+MHA 대비 C-Index 우위(LUAD에서 격차 더 큼).
  - 인사이트: 가변적 1D 스플라인 활성(learnable activation on edges)을 쓰는 KAN이 고차원·희소/잡음 많은 탭형 데이터(오믹스·임상)에서 일반 MLP/주의기반보다 고급 패턴 포착에 유리.

- Table IV: 융합 전략 비교(Fusion KAN vs Self-Attention vs Kronecker)
  - 결과: Fusion KAN이 일관된 최고 성능, Kronecker가 그 다음, Self-Attention이 최하.
  - 인사이트: KAN 기반 융합이 모달 간 비선형 고차 상호작용을 더 풍부하게 모델링. 병리·오믹스/임상 간 이질성에서도 손실 최소화하며 결합 표현을 안정적으로 학습.

- Table V: 상위 유전자 문헌 근거
  - 결과: 아형별 상위 20개 중 10개는 암 관련 문헌 근거 확인. 예) LUAD: CD160(면역화학요법 반응), PARP2(SCLC에서 PARP 억제 치료 타깃), ENSA(세포주기/생존). LUSC: GPR87(흉부암 치료 타깃), SMCP(종양 개시·CSC 표지), UBR5(암 전반).
  - 인사이트: 모델이 생물학적으로 설득력 있는 표지자를 재발견/발굴하며, 아형별로 상위 유전자가 달라 임상·생물학적 차이를 반영. 해석 가능성 측면의 신뢰도 강화.

어펜딕스
- 제공된 본문 범위에는 별도의 어펜딕스가 명시되어 있지 않습니다.



Figures (Diagrams)
- Figure 1: VITKAN workflow
  - Results: WSIs are patched into 224×224 tiles (20×) → histology embeddings from pathology-pretrained ViTs → clinical/omics processed via Tabular KAN (TKAN) → modalities fused by Fusion KAN (FKAN) → survival prediction and IG-based gene attribution.
  - Insights: The combination of pathology-specific ViTs and KAN-based tabular/fusion modules captures complex modality-specific patterns and high-order cross-modal interactions. The design is inherently interpretable (gene importance) and modular, enabling independent optimization of backbones and KAN modules.

- Figure 2: Gene attribution (Integrated Gradients) distributions
  - Results:
    - Most genes have near-zero attributions; a small subset dominates prediction.
    - LUAD: near-symmetric positive/negative attributions; higher top-gene magnitudes than LUSC; smooth decay across top-20.
    - LUSC: slight positive skew; sharp early drop then flatter tail.
    - Top-20 gene sets are mutually exclusive between LUAD and LUSC.
  - Insights: Subtypes rely on distinct gene sets and regulation directions. LUAD appears more genetics-driven (larger top attributions); LUSC shows a broader set with moderate contributions. The model disentangles up- vs down-regulated drivers.

Tables
- Table I: Multimodal baselines vs VITKAN (Porpoise, Pathomic Fusion)
  - Results: VITKAN achieves the highest C-Index across all four settings (LUAD/LUSC × Clinical+Histology / Omics+Histology), with the largest margins in LUAD (e.g., LUAD-Clinical 0.6473).
  - Insights: VITKAN provides consistently superior and robust multimodal integration across subtypes and feature sets.

- Table II: Histology backbone comparison (CTransPath, CONCH, UNI, HistoSSL)
  - Results: Pathology-pretrained ViTs dominate. CTransPath is best on LUAD (Clinical/Omic) and LUSC (Omic), while CONCH is best on LUSC (Clinical). UNI/HistoSSL lag overall.
  - Insights: Backbone choice is critical. Domain-specific pretraining on pathology data materially boosts survival-relevant feature extraction.

- Table III: Tabular networks (Tabular KAN vs FC+MHA)
  - Results: Tabular KAN outperforms FC+MHA in all settings, with larger gains in LUAD.
  - Insights: Learnable spline-based activations in KANs better model high-dimensional, noisy/sparse tabular data (omics/clinical) than standard MLP/attention-based approaches.

- Table IV: Fusion strategies (Fusion KAN vs Self-Attention vs Kronecker)
  - Results: Fusion KAN consistently ranks first, followed by Kronecker; self-attention is last.
  - Insights: KAN-based fusion captures rich, nonlinear, high-order cross-modal interactions more effectively, handling heterogeneity with lower information loss.

- Table V: Literature support for top genes
  - Results: 10 of the top 20 genes (per subtype) have published cancer-related evidence. Examples: LUAD—CD160 (immunochemotherapy response), PARP2 (PARP inhibition in SCLC), ENSA (cell cycle/survival); LUSC—GPR87 (thoracic cancer target), SMCP (tumor initiation/CSC marker), UBR5 (broad cancer relevance).
  - Insights: The model identifies biologically plausible markers and subtype-specific signatures, strengthening interpretability and clinical relevance.

Appendix
- No appendix was included in the provided excerpt.

<br/>
# refer format:

BibTeX (임시)
@misc{vitkan_lung_cancer_2025,
  title        = {Vision Transformer Supported Kolmogorov–Arnold Networks for Survival Prediction in Lung Cancer},
  author       = {Mert Gökpınar and Yasin Almalioglu and Mustafa Taha Kocyigit and Tamer Kahveci and Derya Demir and Kayhan Başak and Mehmet Turan},
  year         = {2025},
  howpublished = {ACM},
  note         = {ACM BCB},
  url          = {https://github.com/mertgokpinar/VITKAN},

}

Chicago 스타일 참고(Author–Date, 서지항목)
Mert Gökpınar, Yasin Almalioglu, Mustafa Taha Kocyigit, Tamer Kahveci, Derya Demir, Kayhan Başak,  and Mehmet Turan. 2025. “Vision Transformer Supported Kolmogorov–Arnold Networks for Survival Prediction in Lung Cancer.” ACM BCB. https://github.com/mertgokpinar/VITKAN.

