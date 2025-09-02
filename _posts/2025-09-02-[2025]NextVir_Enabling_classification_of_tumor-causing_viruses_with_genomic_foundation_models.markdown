---
layout: post
title:  "[2025]NextVir: Enabling classification of tumor-causing viruses with genomic foundation models"
date:   2025-09-02 15:35:06 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 메써드: NextVir는 DNABERT‑S, Nucleotide Transformer, HyenaDNA를 LoRA(rank=4)로 효율 미세조정하고 mean pooling+2층 어댑터로 150bp Illumina MiSeq sequencing data(인간+7개 온코바이러스; iCAV, ART 시뮬레이션)를 다중·이진 분류하며, 컨텍스트 분리(비중첩 구간), 돌연변이/인델, 오염 내성 실험을 설계  


짧은 요약(Abstract) :

- 배경: 종양을 유발하거나 위험을 높이는 온코바이러스는 종이 다양하고 돌연변이가 빠르며, 한 샘플에 여러 바이러스가 섞여 있을 수 있어 단순히 “바이러스 존재 여부”만 판별하는 이진 분류로는 부족합니다.
- 제안: NextVir는 유전체 기반 대형 기초모델(DNABERT-S, Nucleotide Transformer, HyenaDNA)을 과제 특화로 미세조정해, 시퀀싱 리드가 어떤 온코바이러스 계통(패밀리)에서 왔는지 다중 분류하는 프레임워크입니다.
- 방법: 각 기초모델이 산출한 임베딩을 효율적으로 미세조정하여, 짧은 리드 수준에서 바이러스 기원을 식별하도록 학습합니다.
- 결과: NextVir는 기존 딥러닝 방법들을 능가하는 성능을 보였고, 유전체 기초모델을 바이러스 분류 같은 특화 문제에 적응시키는 접근의 잠재력을 시사합니다.


Motivation: Oncoviruses, pathogens known to cause or increase the risk of cancer, include both common viruses such as human papillomaviruses and rarer pathogens such as human T-lymphotropic viruses. Computational methods for detecting viral DNA from data acquired by modern DNA sequencing technologies have enabled studies of the association between oncoviruses and cancers. Those studies are rendered particularly challenging when multiple species of oncovirus are present in a tumor sample. In such scenarios, merely detecting the presence of a sequencing read of viral origin is insufficiently informative—instead, a more precise characterization of the viral content in the sample is required.
Results: We address this need with NextVir, to our knowledge the first multi-class viral classification framework that adapts genomic foundation models to detecting and classifying sequencing reads of oncoviral origin. Specifically, NextVir explores several foundation models—DNABERT-S, Nucelotide Transformer, and HyenaDNA—and efficiently fine-tunes them to enable accurate identification of the sequencing reads’ origin. The results demonstrate superior performance of the proposed framework over existing deep learning methods and suggest downstream potential for foundational models in genomics.


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




1) 문제 정의
- 목표: 각 시퀀싱 리드 r_i가 인간(비바이러스)인지, 아니면 7개 온코바이러스 계통 중 어느 계통에 속하는지 멀티클래스 분류.
- 레이블: y_i ∈ {0, 1, …, n}. 여기서 0은 인간(비바이러스), 1..n은 서로 다른 온코바이러스 계통.
- 비교를 위해 동일 프레임워크를 바이너리(바이러스 vs 비바이러스)로도 변형하여 학습/평가.

2) 기반(파운데이션) 모델 개요
NextVir는 다음 3개 유전체 파운데이션 모델을 백본으로 채택하고, 그 위에 가벼운 어댑터를 올려 미세조정한다.
- DNABERT-S (NextVir-D)
  - 토크나이저: SentencePiece 기반 BPE(바이트-쌍 인코딩). 사전 정의된 k-mer 없이 빈도 기반 서브워드 어휘를 학습.
  - 사전학습/정련: DNABERT-2를 바탕으로 C2LR(Curriculum-Contrastive) 전략. 1) Weighted SimCLR로 임베딩 학습, 2) MI-Mix(Manifold Instance Mixup)로 점진적으로 어려운 양/음성 쌍을 구성해 종 구분 능력을 강화.
  - 입력 길이: 사전학습은 700bp, 정련은 10kb 비중첩 쌍 시퀀스. 바이러스 데이터는 정련 셋에서 소량(약 2.34%)만 포함.
- Nucleotide Transformer 500M (NextVir-N)
  - 구조: BERT 스타일 인코더, MLM(마스킹 언어모델링).
  - 토크나이저: 비중첩 k-mer(6-mer와 1-mer 혼합).
  - 학습 데이터: 다종(plant와 virus 제외) 대규모 유전체. 파라미터 약 5억.
- HyenaDNA (NextVir-H)
  - 구조: 디코더 전용, Hyena 오퍼레이터로 초장문맥(최대 100만 토큰) 처리.
  - 토크나이저: 1-mer(염기 단위 4문자) 최소 어휘.
  - 사전학습: 인간 참조 유전체. 파라미터 약 650만.

3) NextVir 아키텍처
- 입력/전처리
  - 리드 길이: Illumina 환경을 반영해 150bp 고정.
  - 토크나이징: 백본 모델별 스킴 적용.
  - 패딩/어텐션 마스크: 특히 DNABERT-S 기반의 경우 토큰 길이가 배치마다 달라질 수 있어, 배치 내 최대 길이에 맞춰 패딩하고 패딩 토큰이 어텐션에 영향을 주지 않도록 마스크 적용. 변길이 배치의 병렬 처리를 가능하게 함.
- 임베딩과 풀링
  - 각 백본은 토큰 단위 임베딩(크기 L)을 산출: L은 NextVir-D=768, NextVir-N=1024, NextVir-H=256.
  - 토큰 차원에 대해 평균 풀링을 적용하여 입력 리드마다 고정 길이(L차원) 임베딩을 획득. 간단하고 파라미터가 없으며 계산량이 작아 분류 성능과 효율을 균형 있게 달성.
- 어댑터(분류기)
  - 구조: 2층 완전연결(FC) + ReLU. 첫 FC로 64차원 잠재공간에 투영 후, 두 번째 FC가 로짓 산출.
  - 출력: 멀티클래스에서는 (바이러스 계통 수 + 1)의 로짓을, 바이너리에서는 1개의 로짓을 출력. 소프트맥스(멀티) 또는 시그모이드(바이너리)로 확률화.

4) 미세조정(LoRA) 및 학습 설정
- 파라미터 효율 미세조정(PEFT)
  - LoRA를 이용해 주의층의 Q/K/VO 및 FFN(dense) 층에 저랭크 업데이트를 적용. 내재 랭크 r=4(연산/메모리 효율과 성능의 균형). 학습 후, LoRA 업데이트를 베이스 가중치에 합성해 미세조정.
- 손실함수
  - 클래스 불균형 보정을 위한 가중 크로스엔트로피. 클래스 표본수가 적을수록 더 큰 가중을 부여하여 소수계통의 정확한 판별을 유도.
- 최적화 하이퍼파라미터
  - 옵티마이저: Schedule-free AdamW.
  - 기본 설정: 학습률 0.001, β1=0.85, weight decay λ=0.005.
  - 그리드서치: 학습률(0.0005–0.005), β1(0.85–0.95), λ(0.001–0.01)에서 검증셋으로 탐색.
  - 에폭/워밍업: 총 15에폭, 1에폭의 40% 구간을 워밍업. 최종 모델 선택은 검증 손실 기준.
- 계산 환경 및 소프트웨어
  - 하드웨어: AMD EPYC 7642(48코어), RAM 512GB, AMD Vega 20 32GB ROCm GPU 다중(최대 8장).
  - 시간/자원: 멀티클래스 모델(15에폭) 기준 4 GPU에서 약 12시간. 전체 실험 약 15 GPU-일.
  - 구현: Python, PyTorch-ROCm 분산 학습. 사전학습 가중치와 라이선스(DNABERT-S Apache 2.0, NT CC, HyenaDNA BSD-3)를 준수.

5) 학습 데이터 구성
- 온코바이러스 레퍼토리
  - iCAV에서 18,680개 온코바이러스 게놈 수집. 7개 계통: HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV.
- 인간 레퍼런스
  - GRCh38.p14 프라이머리 어셈블리.
- 리드 생성
  - ART 시뮬레이터로 Illumina MiSeq 150bp 리드 생성.
  - 데이터 분할: 무작위 80/10/10(학습/검증/테스트).
- 문맥 분리(Context-supported) 분할
  - 각 리드를 원 게놈에 매핑해 위치를 얻고, 위치 순서대로 정렬한 뒤 앞 80%/다음 10%/마지막 10%로 분할해 학습/검증/테스트가 서로 비중첩 구간에서 유래하게 구성. PCR 치우침 등으로 특정 구간만 커버되는 실제 상황을 모사.
- 강건성 평가용 추가 세트
  - 돌연변이 강건성: 훈련 데이터는 그대로 두고, 테스트에만 (i) 치환 5%, 10%, (ii) 치환 10% + 인델 5%를 주입한 변이 리드를 생성해 성능 평가.
  - 오염(컨타미네이션) 강건성: 테스트 세트에 인간 컨타미놈(박테리아/진균)에서 생성한 리드를 0.5%, 1%, 5% 비율로 스파이크. 이 리드들은 레이블상 “Human(비바이러스)”로 간주. 실제 시퀀싱에서 흔한 저수준 오염을 모사.

6) 학습/평가 프로토콜
- 멀티클래스 분류: 위 아키텍처와 손실로 7개 바이러스 계통 + 인간(비바이러스) 총 8클래스 분류.
- 바이너리 확장 및 벤치마크
  - 출력층을 1 로짓으로 바꾸고 바이너리 크로스엔트로피로 학습.
  - 비교 대상: DeepVirFinder(DVF, CNN), Virtifier(LSTM+어텐션), XVir(Transformers).
  - 공정성 확보: 동일 데이터로 각 베이스라인을 재학습(저자 프로토콜 준수).
  - 지표: 정확도, AUC-ROC. 랜덤 분할과 문맥 분리 분할 모두 보고.

핵심 설계 선택의 의미
- 짧은 리드(150bp) 학습: 종양 시퀀싱에서 가장 보편적인 리드 길이를 직접 타겟팅.
- 평균 풀링: 파라미터 없는 단순 전략으로 계산량과 성능의 균형. 어텐션 풀링 등 고급 기법은 향후 대안.
- LoRA(r=4): 대형 백본의 효율적 미세조정으로, 학습 속도/메모리 사용을 통제하면서도 작업 특이 표현을 획득.
- 가중 크로스엔트로피: 심하게 불균형한 계통(특히 HHV-8, MCV 등)의 성능 저하를 완화.
- 문맥 분리/돌연변이/오염 실험: 실제 시퀀싱 파이프라인에서 빈번한 도전과제를 반영한 강건성 검증.



This section systematically summarizes the Methods of pcbi.1013360, covering the backbone models, overall architecture, training data and preprocessing, optimization and fine-tuning, robustness protocols (context-split, mutations, contamination), and the binary benchmarking setup.

1) Problem formulation
- Goal: For each sequencing read r_i, perform multiclass attribution to Human (non-viral) or one of seven oncoviral families.
- Labels: y_i ∈ {0, 1, …, n}, where 0 denotes Human and 1..n denote distinct oncoviral families.
- For comparison, the framework is also instantiated as a binary classifier (viral vs non-viral).

2) Genomic foundation models (GFMs)
NextVir uses one of three GFMs as a backbone and adds a lightweight adapter on top:
- DNABERT-S (NextVir-D)
  - Tokenizer: SentencePiece with Byte Pair Encoding (BPE), i.e., learned subwords without fixed k-mers.
  - Training strategy: Curriculum-Contrastive (C2LR). Step 1 uses Weighted SimCLR; step 2 uses Manifold Instance Mixup (MI-Mix) to strengthen species-discriminative embeddings.
  - Input lengths during pretraining/refinement: 700 bp (pretraining) and 10 kbp non-overlapping pairs (refinement). Viral sequences are scarce in the refinement set (~2.34%).
- Nucleotide Transformer 500M (NextVir-N)
  - Architecture: BERT-style encoder with masked language modeling.
  - Tokenization: non-overlapping k-mers (a mix of 6-mers and 1-mers).
  - Pretraining corpus: multi-species genomes excluding plants and viruses. ~500M parameters.
- HyenaDNA (NextVir-H)
  - Architecture: decoder-only with the Hyena operator, enabling up to 1M-token context.
  - Tokenization: 1-mers (A/C/G/T).
  - Pretraining: human reference genome. ~6.5M parameters.

3) NextVir architecture
- Input and preprocessing
  - Read length: 150 bp (typical for Illumina short reads in tumor sequencing).
  - Tokenization: per-backbone scheme.
  - Padding and attention masks: sequences in a batch are padded to the max token length, and masks prevent padding from influencing attention. This enables efficient batching of variable-length tokenized inputs (notably for DNABERT-S).
- Embeddings and pooling
  - The backbone outputs per-token embeddings of size L: NextVir-D=768, NextVir-N=1024, NextVir-H=256.
  - Mean pooling across tokens yields a fixed-length L-dimensional sequence embedding per read. This parameter-free choice balances accuracy and efficiency; more elaborate pooling is left as future work.
- Adapter (classifier head)
  - Two fully connected layers with a ReLU in between; the hidden size is 64.
  - Outputs class logits: C logits for multiclass (number of viral families + 1 for Human) or a single logit for the binary variant. Softmax or sigmoid convert logits to probabilities.

4) Fine-tuning with LoRA and training setup
- Parameter-efficient fine-tuning (PEFT)
  - LoRA updates are applied to attention Q/K/V and feedforward (dense) layers. Intrinsic rank r=4 balances efficiency and performance; LoRA updates are merged with base weights after training.
- Loss
  - Class-weighted cross-entropy to counter class imbalance, up-weighting rare classes so the model learns to treat all classes fairly.
- Optimization hyperparameters
  - Optimizer: schedule-free AdamW.
  - Defaults: learning rate 0.001, β1=0.85, weight decay λ=0.005.
  - Grid search: learning rate in [0.0005, 0.005], β1 in [0.85, 0.95], λ in [0.001, 0.01], selected on the validation set.
  - Training: 15 epochs with 40% of the first epoch as warmup; model selection by validation loss.
- Compute and software
  - Hardware: AMD EPYC 7642 (48 cores), 512 GB RAM, up to 8× AMD Vega 20 32 GB GPUs (ROCm).
  - Time/footprint: multiclass training (15 epochs) ~12 hours on 4 GPUs; total ~15 GPU-days across experiments.
  - Stack: Python with PyTorch-ROCm distributed training. Pretrained weights and licenses observed (DNABERT-S Apache 2.0, NT CC, HyenaDNA BSD-3).

5) Training data construction
- Oncoviral genomes
  - 18,680 genomes from iCAV spanning seven families: HBV, HPV, HCV, EBV, HTLV, HHV-8, and MCV.
- Human reference
  - GRCh38.p14 primary assemblies.
- Read simulation
  - ART simulator for Illumina MiSeq 150 bp reads.
  - Random split: 80/10/10 into train/validation/test.
- Context-supported split
  - Reads are aligned to their source genomes to obtain positions, sorted by genomic coordinate, and then split 80/10/10 so that train/val/test originate from disjoint, non-overlapping regions. This mimics PCR bias and uneven coverage scenarios.
- Robustness test sets
  - Mutation robustness: training data unchanged; mutated test sets include (i) 5% and 10% substitutions and (ii) 10% substitutions + 5% indels.
  - Contamination robustness: test sets spiked with human “contaminome” reads (bacterial/fungal) at 0.5%, 1%, and 5%; these reads are labeled as “Human” (non-viral), simulating realistic low-level contamination.

6) Training/evaluation protocol
- Multiclass classification: end-to-end with LoRA fine-tuning and the weighted cross-entropy loss over 8 classes (7 viral families + Human).
- Binary extension and benchmarking
  - Modify the head to output a single logit and train with binary cross-entropy.
  - Baselines: DeepVirFinder (CNN), Virtifier (LSTM with attention), and XVir (Transformer).
  - Fair comparison: retrain baselines on the same dataset under their published training procedures.
  - Metrics: accuracy and AUC-ROC, reported for both random and context-supported splits.

Rationale behind key design choices
- Short-read focus (150 bp): directly targets the dominant read type in tumor sequencing.
- Mean pooling: simple, parameter-free, fast; provides strong baselines without extra compute.
- LoRA (rank 4): efficient adaptation of large backbones with controlled memory and time budgets.
- Class-weighted loss: mitigates severe class imbalance, especially for underrepresented families (e.g., HHV-8, MCV).
- Context-split/mutation/contamination protocols: reflect practical challenges in real sequencing workflows and assess robustness accordingly.


<br/>
# Results



1) 모델/경쟁모델
- NextVir 구성
  - NextVir-D: DNABERT-S 기반
  - NextVir-N: Nucleotide Transformer(500M) 기반
  - NextVir-H: HyenaDNA 기반
  - 학습 전(사전학습 임베딩 고정) vs LoRA(랭크 4) 기반 미세조정(fine-tuning) 두 설정 비교
- 외부 비교(이진 분류 벤치마크)
  - DeepVirFinder(DVF), XVir, Virtifier
  - 세 방법 모두 본 논문 데이터로 재학습해 공정 비교

2) 테스트 데이터(세팅)
- 기본(반무실험) 데이터
  - iCAV에서 7개 발암성 바이러스 군(가족): HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV의 18,680 게놈 + 인간 GRCh38.p14
  - ART로 150bp Illumina MiSeq 리드 생성, 80/10/10 학습/검증/테스트 분할
  - 리드 수(테이블 1): Human 464,016; HBV 229,057; HPV 149,184; HCV 34,086; EBV 25,975; HTLV 5,296; HHV-8 1,959; MCV 1,709
- Context-supported(문맥 분리) 세팅
  - 리드를 원 게놈 좌표에 정렬해 각 클래스별로 위치순 정렬
  - 앞 80% 학습, 다음 10% 검증, 마지막 10% 테스트로 “서로 겹치지 않는 구간”으로 분할
  - PCR 바이어스 등 비균일 커버리지를 모사
- 돌연변이 강건성 테스트
  - 훈련 데이터는 동일
  - 테스트 집합에서 바이러스 리드에 치환(substitution) 5%, 10%와 “10% 치환 + 5% indel” 시나리오 부여
- 오염 강건성 테스트
  - 인간 “contaminome”(세균/진균) 리드를 0.5%, 1%, 5% 비율로 테스트셋에 스파이크
  - 오염 리드는 “Human(비온코바이러스)” 클래스로 라벨

3) 평가 지표(메트릭)
- 다중분류
  - 전체(top-1) 정확도, 클래스별 정확도/재현율(동일 개념으로 사용)
  - 미세조정 모델은 5개 시드 평균에 기반한 95% 신뢰구간 표기
- 이진분류(바이러스 vs 비바이러스)
  - 정확도, AUC-ROC
  - ROC 곡선 제시(그림 3)
- 강건성
  - 돌연변이/오염 수준에 따른 정확도 변화량(Δaccuracy)

4) 핵심 결과 및 비교
A. 사전학습 임베딩만 사용(어댑터만 학습)
- 전체 정확도
  - NextVir-D: 77.40%
  - NextVir-N: 81.51%
  - NextVir-H: 58.00%
- 클래스별
  - 짧은 게놈(HBV, HPV)에서 높고, 긴 게놈(EBV, HHV-8)에서 낮음
  - 예: EBV(NextVir-D 49.59%), HHV-8(NextVir-D 38.60%)
- 해석: 사전학습 중 바이러스 노출이 거의 없었음에도 어댑터만으로 상당 수준 분별 가능. 다만 커버리지(리드 수)가 낮은/게놈이 긴 군에서 성능 저하.

B. LoRA 미세조정 후(표 2)
- 전체(top-1) 정확도(±95% CI)
  - NextVir-D: 94.69% ±0.34
  - NextVir-N: 95.02% ±0.95
  - NextVir-H: 89.88% ±0.24
- 클래스별(일부)
  - EBV: NextVir-D 95.13% ±0.56(사전학습 49.59% → 대폭 향상)
  - HHV-8: NextVir-D 62.33% ±1.41, NextVir-N 57.30% ±2.47, NextVir-H 54.70% ±2.46(개선되지만 여전히 최난이도)
  - HBV: 99%대(모든 모델)
  - HPV/HCV/HTLV/MCV: 전반적으로 90%대(일부 95~100% 근접)
- 해석: 미세조정이 전 군에서 큰 폭의 향상을 유도. 특히 긴 게놈·저커버리지 클래스(EBV, HHV-8)에서 향상이 두드러지나, HHV-8은 여전히 데이터 희소성으로 난이도가 높음.

C. Context-supported(문맥 분리) 다중분류(표 3)
- 전체(top-1)
  - NextVir-D: 86.85%
  - NextVir-N: 89.38%
  - NextVir-H: 79.89%
- 클래스별 특징
  - HHV-8 급락(NextVir-D 15.82%, NextVir-N 21.43%, NextVir-H 13.27%)
  - MCV는 여전히 매우 높음(≥98%)
  - NextVir-N이 전반적으로 가장 견조
- 해석: 훈련/테스트 구간 분리로 쉬운 “문맥 암기”를 차단하자 전체 성능이 하락. 그러나 사전학습-only보다 여전히 높아, 미세조정이 “일반화 가능한 패턴”을 학습했음을 시사. HHV-8은 극저커버리지+긴 게놈으로 특히 어렵고, MCV는 짧은 게놈+매우 낮은 변이/구조적 단순성 영향으로 추정.

D. 돌연변이 강건성(표 4)
- 기준(무변이) 대비 전체 정확도 변화
  - 5% 치환: 전체 정확도 감소 <4%p(모든 모델). 대부분 클래스에서 <7%p 감소.
  - 10% 치환: 중등도 하락(예: NextVir-D 85.04%, -9.7%p)
  - 10% 치환 + 5% indel: 큰 하락(NextVir-D 76.16%, -18.5%p; NextVir-N 78.66%, -15.2%p; NextVir-H 74.96%, -15.0%p)
- 클래스 취약성
  - EBV, HHV-8, MCV에서 변이에 더 민감
  - NextVir-N은 비중첩 k-mer 토크나이즈로 indel 프레임시프트에 취약(예: HPV에서 타 모델 대비 하락 폭 큼)
  - NextVir-D(BPE)도 indel에 민감(시프트로 토큰 경계 붕괴)
  - NextVir-H(1-mer)는 기대 대비 indel 이점이 일관되게 나타나진 않음
- 해석: 현실적 오차·변이(치환 5%)엔 비교적 강건. indel 비중이 커지면 토크나이즈 특성상 급격히 취약해짐.

E. 오염 강건성(그림 2)
- 세균/진균 오염 0.5%, 1%, 5% 스파이크 시에도 전체 정확도 소폭 하락에 그침
- NextVir-D가 가장 안정, NextVir-N이 상대적으로 하락 폭 큼
- 일반적 오염 수준(~1%)을 고려하면 실사용 파이프라인에 충분히 견조
- 오염 리드를 “Human(비온코바이러스)”로 올바르게 분류하는 경향 확인

F. 이진 분류 벤치마크(표 5, 그림 3)
- 랜덤 분할
  - 정확도 / AUC-ROC
  - DVF: 98.87% / 0.999(정확도 최고)
  - NextVir-N: 98.69% / 0.999
  - NextVir-D: 98.22% / 0.999
  - NextVir-H: 97.71% / 0.997
  - XVir: 98.00% / 0.998
  - Virtifier: 93.71% / 0.984
- Context-supported(문맥 분리)
  - NextVir-N: 91.57% / 0.989
  - NextVir-D: 91.55% / 0.985
  - NextVir-H: 88.70% / 0.969
  - DVF: 89.90% / 0.968
  - XVir: 88.50% / 0.968
  - Virtifier: 71.70% / 0.922
- 해석
  - 랜덤 분할에선 DVF가 근소하게 최고 정확도지만, AUC는 NextVir와 동급
  - 문맥 분리(더 어려운)에서는 NextVir-D/N이 DVF 등 기존법을 상회
  - ROC 곡선에서도 NextVir의 일관된 우수성 확인

5) 종합 해석
- 다중분류 최초 프레임워크로서 NextVir는 사전학습 기반 모델을 LoRA로 효율적으로 적응시켜 전반적 최고 성능 달성(특히 EBV 등 난도 높은 군에서 대폭 향상).
- 커버리지(리드 수/게놈 길이)가 성능과 강건성(돌연변이, 문맥 분리)에 큰 영향을 미침. HHV-8은 극저커버리지로 지속적 난점.
- 이진 분류 벤치마크에서도 특히 “문맥 분리” 조건에서 차별적 우수성 확인 → 실제 분석 파이프라인에서의 일반화 잠재력 높음.
- 오염에 대한 견조성은 실제 임상/현장 데이터 적용성을 뒷받침.



1) Models and baselines
- NextVir variants
  - NextVir-D (DNABERT-S), NextVir-N (Nucleotide Transformer 500M), NextVir-H (HyenaDNA)
  - Two regimes: frozen pretrained embeddings with adapter-only training vs LoRA-based fine-tuning (rank 4)
- External baselines for binary detection
  - DeepVirFinder (DVF), XVir, Virtifier
  - All retrained on this study’s dataset for fair comparison

2) Test data (settings)
- Semi-experimental main dataset
  - 7 oncoviral families from iCAV: HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV + human GRCh38.p14
  - 150 bp Illumina MiSeq reads via ART; 80/10/10 train/val/test
  - Read counts (Table 1): Human 464,016; HBV 229,057; HPV 149,184; HCV 34,086; EBV 25,975; HTLV 5,296; HHV-8 1,959; MCV 1,709
- Context-supported split
  - Reads aligned to genomes; per class, ordered by position and split into disjoint genomic segments (80/10/10)
  - Mimics PCR biases and uneven coverage
- Mutation robustness test
  - Training unchanged; test reads mutated
  - Substitutions at 5% and 10%; “10% substitution + 5% indel” stress test
- Contamination robustness test
  - Spike bacterial/fungal “human contaminome” reads at 0.5%, 1%, 5% into the test set
  - Contaminant reads labeled as “Human” (non-oncoviral)

3) Metrics
- Multiclass: overall top-1 accuracy, per-class accuracy/recall; for fine-tuned models, 95% CI across 5 seeds
- Binary: accuracy and AUC-ROC; ROC curves
- Robustness: change in accuracy under mutation/contamination

4) Key results and comparisons
A. Adapter-only (frozen) models
- Overall accuracy
  - NextVir-D: 77.40%; NextVir-N: 81.51%; NextVir-H: 58.00%
- Per-class pattern: higher on short genomes (HBV/HPV), lower on long genomes (EBV/HHV-8)
- Takeaway: even without fine-tuning, adapters leverage pretrained representations; however, low coverage/long genomes remain challenging

B. With LoRA fine-tuning (Table 2)
- Overall top-1 (±95% CI)
  - NextVir-D: 94.69% ±0.34; NextVir-N: 95.02% ±0.95; NextVir-H: 89.88% ±0.24
- Highlights
  - EBV nearly doubles: NextVir-D 95.13% ±0.56
  - HHV-8 improves but remains hardest: NextVir-D 62.33% ±1.41; NextVir-N 57.30% ±2.47; NextVir-H 54.70% ±2.46
  - HBV ~99.5–99.8%; other classes mostly in 90s
- Takeaway: fine-tuning yields large gains across all families, especially for EBV; HHV-8 is still constrained by extreme rarity

C. Context-supported multiclass (Table 3)
- Overall top-1: NextVir-D 86.85%; NextVir-N 89.38%; NextVir-H 79.89%
- HHV-8 drops sharply (≤~21% recall), while MCV stays ~99%
- NextVir-N best overall in this harder setting
- Takeaway: disjoint genomic regions lower performance but remain well above frozen baselines, evidencing true generalization; HHV-8 is most affected by low coverage + long genome

D. Mutation robustness (Table 4)
- Global trends
  - At 5% substitutions: overall drop <4 percentage points for all models; most classes <7 points
  - At 10% substitutions: moderate drop (e.g., NextVir-D to 85.04%, −9.7 points)
  - At 10% substitutions + 5% indel: large drop (NextVir-D 76.16%, −18.5; NextVir-N 78.66%, −15.2; NextVir-H 74.96%, −15.0)
- Class vulnerabilities: EBV, HHV-8, MCV more sensitive
- Tokenization effects
  - NextVir-N (non-overlapping k-mers) is indel-sensitive due to frameshifts (e.g., HPV shows larger drops)
  - NextVir-D (BPE) also suffers from indel-induced token boundary shifts
  - NextVir-H (1-mer) does not show consistent advantage under indels
- Takeaway: realistic error rates (small substitutions) are well-tolerated; indels remain a key failure mode

E. Contamination robustness (Fig. 2)
- Accuracy degrades only marginally up to 5% spiked bacterial/fungal contamination
- NextVir-D has smallest drop; NextVir-N largest among the three
- Given typical ~1% contamination, robustness is adequate for practice

F. Binary benchmarks (Table 5; Fig. 3)
- Random split (Accuracy / AUC-ROC)
  - DVF: 98.87% / 0.999 (best accuracy)
  - NextVir-N: 98.69% / 0.999; NextVir-D: 98.22% / 0.999
  - NextVir-H: 97.71% / 0.997; XVir: 98.00% / 0.998; Virtifier: 93.71% / 0.984
- Context-supported
  - NextVir-N: 91.57% / 0.989; NextVir-D: 91.55% / 0.985
  - DVF: 89.90% / 0.968; XVir: 88.50% / 0.968; NextVir-H: 88.70% / 0.969; Virtifier: 71.70% / 0.922
- Takeaway: DVF slightly leads in random-split accuracy, but NextVir matches AUC and clearly surpasses all baselines in the harder context-supported setting

5) Overall interpretation
- NextVir is the first multiclass oncoviral read classifier leveraging genomic foundation models and LoRA, delivering strong gains over frozen embeddings and state-of-the-art performance in challenging settings (disjoint genomic segments).
- Coverage (reads per genome length) is a dominant driver of accuracy and robustness; HHV-8 remains the hardest class due to extreme rarity and long genome.
- In binary viral detection, NextVir meets or exceeds leading baselines, especially under context-supported splits, indicating better generalization.
- Robustness to low-level contamination supports practical deployment in real sequencing pipelines.

참고: 모든 수치와 해석은 본문 내 표/그림(표 1–5, 그림 2–3)과 해당 서술을 근거로 정리했습니다.


<br/>
# 예제





1) 개요
- 목적: 종양 샘플의 NGS(short-read, 150bp)에서 유래한 염기서열 단편이 어느 온코바이러스 계통(가족)에서 왔는지를 다중 분류하거나, 최소한 바이러스/비바이러스(사람)인지 이진 분류하는 모델을 학습/평가.
- 기반: 세 가지 게놈 파운데이션 모델(DNABERT-S, Nucleotide Transformer, HyenaDNA)을 LoRA 미세조정과 얇은 어댑터로 적응. 입력은 150bp 리드, 출력은 클래스 로짓(다중 분류 시 8개: 사람+7 바이러스 계통; 이진 시 1개).

2) 데이터셋 구성(트레이닝/검증/테스트 생성)
- 바이러스 유전체: iCAV 데이터베이스에서 7개 온코바이러스 계통의 18,680개 유전체
  - HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV
- 비바이러스(사람): GRCh38.p14 프라이머리 어셈블리
- 리드 생성: ART로 Illumina MiSeq 150bp 리드 시뮬레이션
- 분할: 전체 리드를 80:10:10(학습:검증:테스트) 랜덤 분할
- 클래스별 리드 수/유전체 길이(요지)
  - 사람: 4.64e8 bp 기준 다수 리드
  - EBV/HHV-8처럼 유전체가 긴 바이러스는 커버리지가 낮아 리드 수가 상대적으로 적음
  - HHV-8, MCV는 전체 분포에서 특히 희소

3) 입력 표현과 전처리(모델별 토크나이즈, 패딩, 풀링)
- 공통 입력: A/C/G/T로 구성된 150bp 문자열 한 개가 “한 샘플”
- 토큰화(모델별)
  - NextVir-D(DNABERT-S): SentencePiece+BPE 기반 서브워드 토큰(가변 길이 시퀀스)
  - NextVir-N(Nucleotide Transformer): 비중첩 k-mer(주로 6-mer, 필요 시 1-mer 혼합)
  - NextVir-H(HyenaDNA): 1-mer(개별 염기) 토큰
- 배치 처리: 최대 길이에 맞춰 패딩 및 어텐션 마스크 적용(패딩 토큰은 어텐션에서 무시)
- 임베딩: 토큰 시퀀스를 변환한 후 토큰 축 평균(mean pooling) → 고정 길이 벡터
  - 임베딩 차원 L: D=768, N=1024, H=256
- 어댑터: 2층 FFN(중간 차원 64, ReLU) → 출력 로짓 생성

4) 라벨과 출력(다중/이진)
- 다중 분류
  - 라벨 집합: {사람(0), HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV} 총 8개
  - 출력: 길이 8 로짓 → 소프트맥스 확률 → argmax가 예측 클래스
  - 손실: 클래스 불균형을 보정하는 가중 크로스엔트로피(클래스 j의 가중치 ∝ 1/Mj)
- 이진 분류(벤치마크용)
  - 라벨: 1=바이러스(7개 계통 중 하나), 0=사람
  - 출력: 스칼라 로짓 → 시그모이드 확률 → 0.5 임계로 판정
  - 손실: 바이너리 크로스엔트로피

5) 학습 설정(핵심)
- 미세조정: LoRA(rank=4)를 자기어텐션(Q/K/V)과 dense층에 적용 + 어댑터 학습
- 최적화: Schedule-free AdamW(lr=0.001, β1=0.85, weight decay λ=0.005), 15 epoch, 첫 epoch의 40% 워ーム업, 검증 손실로 모델 선택
- 배치 입력은 패딩/마스크로 병렬 처리

6) 수행 태스크 정의(실험 프로토콜)
A. 표준 다중 분류(랜덤 분할)
- 목적: 8클래스 중 어디에 속하는지 리드 단위로 예측
- 인풋: 150bp 리드 문자열
- 아웃풋: 8차원 확률(사람/7바이러스 계통)
- 지표: 전체 정확도, 클래스별 리콜/정확도 등

B. 컨텍스트-서포티드(문맥 분리) 다중 분류
- 동기: PCR 등으로 특정 영역만 과증폭되어 학습/테스트 구간이 겹치지 않을 수 있음
- 방법: 각 클래스 내에서 리드를 유전체 좌표로 정렬 후
  - 앞 80%→학습, 다음 10%→검증, 마지막 10%→테스트(서로 비중첩)
- 인풋/아웃풋: 표준 다중 분류와 동일
- 지표: 표준과 동일(정확도 등). 일반적으로 표준보다 정확도 하락

C. 돌연변이 강건성 테스트(테스트셋 변형)
- 동기: 바이러스의 높은 변이율(치환/indel)에 대한 내성 평가
- 방법: 학습 데이터는 그대로, 테스트 리드에 변이 주입
  - 치환률: 5%, 10%
  - indel: 10% 치환 + 5% indel 조합
- 인풋: 변이가 주입된 150bp 리드
- 아웃풋: 표준 다중 분류 출력
- 관찰: 변이↑ → 정확도↓, indel이 특히 큰 영향. 희소·저커버리지 클래스(EBV, HHV-8, MCV)가 더 취약

D. 오염 강건성 테스트(테스트셋 스파이크인)
- 동기: 실제 실험에서 박테리아/진균 등 비표적 DNA가 소량 섞임
- 방법: 테스트셋에 사람 컨타미놈(박테리아/진균 유래 리드)을 0.5%, 1%, 5% 비율로 스파이크인
  - 이 추가 리드는 “사람(비바이러스)” 클래스로 라벨
- 인풋: 오염 리드가 혼합된 테스트 리드
- 아웃풋: 표준 다중 분류 출력
- 관찰: 낮은 오염률에서 성능 저하 미미, 5%에서도 평균 정확도 유지

E. 이진 분류 벤치마킹(외부 방법과 비교)
- 목적: Viral vs Non-viral 판별 태스크로 단순화하여 기존 모델(DVF, Virtifier, XVir)과 비교
- 인풋: 150bp 리드
- 아웃풋: 바이러스일 확률(스칼라)
- 지표: 정확도, AUC-ROC
- 분할: 랜덤 분할과 컨텍스트-서포티드 분할 둘 다 평가

7) 구체적 예시

예시 A: 다중 분류 학습 배치(랜덤 분할)
- 입력(배치 3개)
  - r1: “ACGT…(총 150bp)” → 실제 출처: HPV
  - r2: “TTGA…(150bp)” → 실제 출처: 사람
  - r3: “CCCA…(150bp)” → 실제 출처: EBV
- 토큰화
  - NextVir-D: [T37, T592, …, PAD, PAD] + 어텐션 마스크
  - NextVir-N: [“ACGTGC”, “TGCAAA”, …] 6-mer 비중첩
  - NextVir-H: [A, C, G, T, …] 150개의 1-mer
- 어댑터 출력(로짓 → 소프트맥스)
  - r1 예: [사람:0.01, HBV:0.02, HPV:0.95, …] → 예측: HPV
  - r2 예: [사람:0.98, HBV:0.01, HPV:0.00, …] → 예측: 사람
  - r3 예: [사람:0.05, EBV:0.90, …] → 예측: EBV
- 라벨: r1=HPV, r2=사람, r3=EBV
- 손실: 가중 크로스엔트로피(희소 클래스 가중↑)

예시 B: 표준 테스트 예측
- 입력: “GATC…(150bp)”(실제 HCV 유래)
- 출력 확률(예): 사람 0.03, HBV 0.01, HPV 0.02, HCV 0.92, … → 예측 HCV

예시 C: 컨텍스트-서포티드 분할 설정(EBV 예)
- EBV 유전체(평균 ~168kb) 좌표 기준 정렬
  - 0–80% 구간 리드 → 학습
  - 80–90% → 검증
  - 90–100% → 테스트
- 테스트 리드는 학습과 유전체 상 문맥이 겹치지 않음

예시 D: 돌연변이 주입
- 원본 HPV 리드: “AGCT…(150bp)”
- 10% 치환 + 5% indel
  - 대략 15개 위치 치환, 7~8개 위치 삽입/삭제로 프레임 시프트
- 입력: 변이 주입된 시퀀스
- 기대: 치환만 있을 때보다 성능 하락이 크며, k-mer 기반 토큰화(NextVir-N)는 프레임 시프트에 특히 민감

예시 E: 오염 스파이크인
- 테스트셋 10만 리드에 대해 1% 오염률 → 1,000개 박테리아/진균 리드 추가
- 이 리드는 모두 “사람(비바이러스)” 라벨로 평가
- 기대: 평균 정확도 소폭 하락(대개 유지), 모델은 오염 리드를 비바이러스로 올바르게 분류하는 경향

예시 F: 이진 분류 태스크(벤치마크)
- 입력: “ATGC…(150bp)”
- 출력: 바이러스일 확률 p(예: 0.97) → 예측=바이러스
- 지표: AUC-ROC, 정확도
- 결과 경향: 랜덤 분할에서는 SOTA와 비슷하거나 근접, 컨텍스트-서포티드에서는 NextVir가 베이스라인 대비 우수

8) 추가 구현 포인트(간단 요약)
- LoRA rank=4로 파라미터 효율적 업데이트
- 평균 풀링으로 토큰별 임베딩을 시퀀스 임베딩으로 압축(단순·안정)
- 15 epoch, schedule-free AdamW, warmup 40% of 1st epoch
- 가중 크로스엔트로피로 희소 클래스(예: HHV-8) 보정 시도




1) Overview
- Goal: Given 150bp NGS reads from tumor samples, classify each read into one of the seven oncovirus families or “Human” (non-viral), or perform binary viral vs non-viral detection.
- Approach: Start from genomic foundation models (DNABERT-S, Nucleotide Transformer, HyenaDNA), apply LoRA-based fine-tuning and a lightweight adapter. Input is a 150bp DNA string; output is logits over 8 classes (multi-class) or a single logit (binary).

2) Dataset construction (train/val/test)
- Viral genomes: 18,680 genomes from iCAV across 7 oncoviral families
  - HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV
- Non-viral: Human GRCh38.p14 primary assemblies
- Read simulation: Illumina MiSeq 150bp reads via ART
- Split: 80:10:10 random split into train/val/test
- Notes: Long-genome viruses (e.g., EBV, HHV-8) tend to have lower coverage; HHV-8 and MCV are particularly sparse in the dataset.

3) Input representation and preprocessing
- Common input: One 150bp A/C/G/T string per sample
- Tokenization (per base model)
  - NextVir-D (DNABERT-S): SentencePiece+BPE subwords (variable-length)
  - NextVir-N (Nucleotide Transformer): non-overlapping k-mers (mostly 6-mers, mixed with 1-mers)
  - NextVir-H (HyenaDNA): 1-mer tokens
- Batching: pad to the longest sequence in batch with attention masks
- Embeddings: mean pooling over token dimension → fixed-length vector
  - Embedding sizes L: D=768, N=1024, H=256
- Adapter: 2-layer FFN (hidden=64, ReLU) → output logits

4) Labels and outputs
- Multi-class
  - Label set: {Human(0), HBV, HPV, HCV, EBV, HTLV, HHV-8, MCV} (8 classes)
  - Output: 8 logits → softmax probabilities → argmax as prediction
  - Loss: class-weighted cross-entropy (weight ∝ 1/Mj for class j)
- Binary (for benchmarking)
  - Labels: 1=viral, 0=non-viral (Human)
  - Output: single logit → sigmoid probability → threshold at 0.5
  - Loss: binary cross-entropy

5) Training setup (key points)
- Fine-tuning: LoRA (rank=4) on self-attention (Q/K/V) and dense layers + train the adapter
- Optimizer: schedule-free AdamW (lr=0.001, β1=0.85, weight decay λ=0.005), 15 epochs, 40% warmup of the first epoch, model selection by validation loss
- Batching with padding/masking

6) Tasks (experimental protocols)
A. Standard multi-class classification (random split)
- Input: 150bp read
- Output: probabilities over 8 classes
- Metrics: overall accuracy, per-class accuracy/recall, etc.

B. Context-supported multi-class classification
- Motivation: PCR and other workflows can create non-uniform coverage; test reads may come from genome regions unseen during training.
- Method: Within each class, order reads by genome coordinates; split 0–80% (train), 80–90% (val), 90–100% (test), ensuring non-overlapping regions across splits.
- Same inputs/outputs as (A), expect lower accuracy vs random split.

C. Robustness to mutations (mutated test only)
- Motivation: High viral mutation rates (substitutions and indels)
- Method: Keep training unchanged; inject mutations into test reads
  - Substitution rates: 5%, 10%
  - Indels: 10% substitutions + 5% indels
- Input: mutated 150bp reads
- Output: multi-class predictions
- Observation: Accuracy drops as mutation rate increases; indels cause larger degradation, especially for sparse/low-coverage classes (EBV, HHV-8, MCV).

D. Robustness to contamination (spike-in test)
- Motivation: Real sequencing contains non-target DNA (bacteria/fungi)
- Method: Spike test set with human contaminome reads at 0.5%, 1%, 5% proportions
  - Label all spiked reads as “Human” (non-viral)
- Input: mixed test set
- Output: multi-class predictions
- Observation: Only marginal drop at low contamination; even at 5% overall accuracy remains high.

E. Binary detection benchmarking (vs. external baselines)
- Simplify to viral vs non-viral; compare with DVF, Virtifier, XVir
- Input: 150bp read; Output: viral probability
- Metrics: accuracy, AUC-ROC
- Splits: both random and context-supported
- Finding: NextVir matches SOTA in random split and outperforms baselines in context-supported split.

7) Concrete examples

Example A: Multi-class training batch (random split)
- Inputs (batch of 3)
  - r1: “ACGT…(150bp)” → true label: HPV
  - r2: “TTGA…(150bp)” → true label: Human
  - r3: “CCCA…(150bp)” → true label: EBV
- Tokenization
  - NextVir-D: [T37, T592, …, PAD] + attention mask
  - NextVir-N: [“ACGTGC”, “TGCAAA”, …] 6-mers
  - NextVir-H: [A, C, G, T, …] 150 1-mers
- Adapter outputs (logits→softmax)
  - r1: [Human 0.01, HBV 0.02, HPV 0.95, …] → predict HPV
  - r2: [Human 0.98, HBV 0.01, …] → predict Human
  - r3: [Human 0.05, EBV 0.90, …] → predict EBV
- Loss: weighted cross-entropy

Example B: Standard test prediction
- Input: “GATC…(150bp)” (true HCV)
- Output probs: Human 0.03, HBV 0.01, HPV 0.02, HCV 0.92, … → predict HCV

Example C: Context-supported split (EBV)
- Sort EBV reads by coordinate on its ~168kb genome
  - 0–80% → train, 80–90% → val, 90–100% → test
- Test reads share no genomic context with training reads

Example D: Mutation injection
- Original HPV read: “AGCT…(150bp)”
- 10% substitutions + 5% indels
  - ~15 substitutions, ~7–8 indel events causing frame shifts
- Expect larger performance drop than substitutions alone; k-mer tokenizers are especially sensitive to frame shifts.

Example E: Contamination spike-in
- For a 100k-read test set, add 1% bacterial/fungal reads → +1,000 reads
- Label all added reads as Human
- Expect small average accuracy drop; model tends to classify contaminants as non-viral correctly.

Example F: Binary task (benchmark)
- Input: “ATGC…(150bp)”
- Output: viral probability p (e.g., 0.97) → predict viral
- Metrics: AUC-ROC, accuracy
- Trend: Comparable to SOTA on random split; superior under context-supported split.

8) Implementation notes (brief)
- LoRA rank=4 for parameter-efficient fine-tuning
- Mean pooling for stable, parameter-free sequence embedding
- 15 epochs, schedule-free AdamW, 40% warmup of first epoch
- Class-weighted CE to compensate for rare classes (e.g., HHV-8)

참고: 위 예시의 토큰 표기(T37 등), 확률 값 등은 설명을 위한 예시입니다. 실제 SentencePiece/BPE 어휘나 확률 분포는 학습 결과에 따라 달라집니다.

<br/>
# 요약

메써드: NextVir는 DNABERT‑S, Nucleotide Transformer, HyenaDNA를 LoRA(rank=4)로 효율 미세조정하고 mean pooling+2층 어댑터로 150bp Illumina 읽기(인간+7개 온코바이러스; iCAV, ART 시뮬레이션)를 다중·이진 분류하며, 컨텍스트 분리(비중첩 구간), 돌연변이/인델, 오염 내성 실험을 설계했다. 
결과: 미세조정 후 전체 Top‑1은 90%+ (NextVir‑N 95.0%, NextVir‑D 94.7%, NextVir‑H 89.9%); EBV는 NextVir‑D에서 ~95%로 거의 두 배 향상되었고 HHV‑8은 가장 어려웠으며, 컨텍스트 분리에서도 79.9–89.4%를 유지, 5% 치환 시 전체 정확도 하락은 <4%, 오염 5%까지도 영향이 미미했다. 
예시: 다중 온코바이러스 혼합 시료, PCR 치우침으로 인한 비균일 피복, 치환·인델 변이, 박테리아/진균 오염 시나리오에서 견고함을 보였고, 이진 검출 벤치마크에선 무작위 분할에서 DVF와 동급(정확도 98.2–98.7%, AUC 0.997–0.999), 컨텍스트 분리에서는 기존법을 상회했다.

Methods: NextVir adapts DNABERT‑S, Nucleotide Transformer, and HyenaDNA with LoRA (rank 4) and a 2‑layer adapter on mean‑pooled embeddings to classify 150‑bp Illumina reads (human plus 7 oncoviral families from iCAV, ART‑simulated), and evaluates context‑supported splits (non‑overlapping regions), mutation/indel robustness, and contamination tolerance. 
Results: After fine‑tuning, overall top‑1 exceeded 90% (NextVir‑N 95.0%, NextVir‑D 94.7%, NextVir‑H 89.9%); EBV nearly doubled to ~95% with NextVir‑D while HHV‑8 remained hardest; accuracy under context‑supported splits stayed at 79.9–89.4%, 5% substitutions reduced accuracy by <4%, and up to 5% contamination had only marginal impact. 
Examples: The method was robust on mixed oncoviral samples, PCR‑biased coverage, substitution/indel variation, and bacterial/fungal contamination, and in binary detection matched DVF on random splits (accuracy 98.2–98.7%, AUC 0.997–0.999) while outperforming baselines on context‑supported splits.

<br/>
# 기타



피규어(다이어그램)
- Fig 1. NextVir 프레임워크 개요
  - 결과/설명: 세 가지 기반 모델(DNABERT-S, Nucleotide Transformer, HyenaDNA)의 시퀀스 임베딩을 평균 풀링하고, 2층 FFN 어댑터가 로짓을 출력. LoRA로 주의(쿼리/키/밸류)와 dense 층만 저랭크 업데이트. 배치 패딩+마스크로 가변 길이 입력 처리. 캡션의 점선 화살표는 훈련 레짐(프리트레인 고정 vs LoRA 파인튜닝)에서 갱신되는 층을 표시.
  - 인사이트: 어댑터만 학습해도 어느 정도 분류가 가능하지만(특히 짧은 바이러스), LoRA 미세튜닝으로 길고 커버리지가 낮은 바이러스(EBV, HHV-8)에서 성능이 크게 향상됨. 구조가 단순하여 효율적이며, 다양한 토크나이저(서브워드, k-mer, 1-mer)에 공통으로 적용 가능.

- Fig 2. 오염(컨타미네이션) 내성
  - 결과: 테스트셋에 0.5–5%의 박테리아/진균 읽기를 스파이크해도 전체 정확도 하락은 매우 작음. 보통 실제 오염이 ~1% 수준임을 감안하면 현장 적용에 충분히 견고. 세 모델 중 NextVir-D가 하락폭이 가장 작고, NextVir-N이 가장 큼.
  - 인사이트: NextVir는 비온코바이러스 혼입 환경에서도 “비온코바이러스(=Human) 클래스”로 안전하게 흡수해 분류 파이프라인 전단 필터로 활용 가능.

- Fig 3. 이진(바이러스 vs 비바이러스) ROC 곡선
  - 결과: 랜덤 분할에서는 DVF와 동급의 최고 AUC, 컨텍스트-서포티드(유전체 비중첩 구간 분할)에서는 NextVir가 DVF, XVir, Virtifier보다 일관되게 우수.
  - 인사이트: 단순 랜덤 분할을 넘어, 실제 실험에서 흔한 비균일 커버리지/구간 편향에도 일반화가 잘 됨. 파운데이션 모델 기반 임베딩이 ‘구간 암기’가 아닌 ‘일반 패턴’ 학습에 유리함을 시사.

테이블
- Table 1. 데이터셋 구성(유전체 길이, 150bp 읽기 수)
  - 결과: EBV(∼168kb), HHV-8(∼135kb)은 길이가 길고 읽기 수가 적어(커버리지 낮음), 분류가 어렵다. HBV(∼3.2kb), HPV(∼7.7kb)는 짧아 커버리지가 높고 분류가 쉬움. MCV는 읽기 수가 매우 적음.
  - 인사이트: 커버리지(읽기 밀도)가 클래스 난이도를 좌우. 길고 희귀한 바이러스는 추가 데이터/전략 필요.

- Table 2. 프리트레인 고정 vs LoRA 파인튜닝 성능(멀티클래스)
  - 결과: 어댑터만 학습해도 다수 클래스에서 70%+ 정확도이나 EBV/HHV-8에서 약함. LoRA 파인튜닝 후 전체 정확도 90%+ (NextVir-D/N ≈95%). EBV는 거의 2배 향상, HHV-8도 큰 폭 개선(여전히 최난).
  - 인사이트: 미세튜닝의 이득이 가장 큰 곳은 긴 유전체·저커버리지 클래스. 파운데이션 임베딩의 전이 가능성은 확인되나, 과제 특화 미세튜닝이 결정적.

- Table 3. 컨텍스트-서포티드 분류(훈련/검증/테스트가 유전체 비중첩 구간)
  - 결과: 전체 정확도 하락(예: NextVir-D 94.8%→86.9%)하지만 프리트레인 고정 대비 여전히 크게 우수. NextVir-N이 상대적으로 가장 견고. HHV-8은 급락, 반면 MCV는 거의 유지.
  - 인사이트: 구간별 바이어스가 있어도 일반화가 유지됨. HHV-8의 급락은 “희귀+긴 유전체 → 극저커버리지”의 이중 난이도 때문. 커버리지 보완이 핵심.

- Table 4. 돌연변이(치환/인델) 강건성
  - 결과: 5–10% 치환에서는 전반적 정확도 하락이 작음(모델별 전체 -2~10%p, 다수 클래스 -7%p 내). 인델(5%)이 섞이면 급격히 하락. DNABERT-S(BPE)는 프레임시프트에 취약. Nucleotide Transformer(비중첩 k-mer)는 인델에 특히 민감(HPV에서 타 모델 대비 하락 큼). HyenaDNA(1-mer)는 기대만큼 인델 내성이 일관되게 강하지 않음. 저커버리지/희귀 클래스(EBV, HHV-8, MCV)가 가장 큰 타격.
  - 인사이트: 치환에는 비교적 강하나, 인델 내성은 토크나이저/인코딩 설계의 한계가 큼. 실제 임상 샘플에서 인델·프레임시프트가 존재할 수 있어, 인델 친화적 전처리/토큰화가 향후 과제.

- Table 5. 이진 분류(바이러스 탐지) 벤치마크
  - 결과: 랜덤 분할에서 NextVir는 DVF와 동급 최고 정확도/AUC. 컨텍스트-서포티드에서는 모든 베이스라인(DVF/XVir/Virtifier) 대비 NextVir가 상회(NextVir-D/N 정확도 ≈91.6%, AUC 0.985–0.989).
  - 인사이트: 파운데이션 모델 기반 접근이 이진 탐지에서도 SOTA와 경쟁/우위. 특히 어려운 분할에서 장점이 큼.

어펜딕스(S1 Appendix)
- 수록 내용(논문 본문에 언급된 범위)
  - LoRA 랭크 선택: 랭크를 키우면 성능이 약간 더 좋아질 수 있으나 계산비용/수렴시간이 크게 증가(본문에서도 r=4 채택 근거로 인용).
  - 컨텍스트-서포티드 분할: 정렬 기반 위치 정렬, 80/10/10 비중첩 스플릿 절차 및 시각화 제공.
  - 최적화/그리드서치: 러닝레이트, β1, λ 범위 탐색 세부 공유.
  - 추가 벤치마크: 단일 종 이진 탐지, 균일 샘플링 데이터, 시뮬레이티드 바이러스 발견, HPV 실험 데이터 검출, HHV-8 난이도 분석 등 보강 결과 수록.
- 인사이트: 본문 결론을 지지/보강하는 부가 실험과 구현·튜닝 세부가 제공됨. 특히 HHV-8 난점(희소성+긴 유전체+저커버리지)과 LoRA 랭크–효율 트레이드오프의 정량적 근거가 어펜딕스에 정리됨.




Figures/Diagrams
- Fig 1. NextVir framework
  - Findings: Three base models (DNABERT-S, Nucleotide Transformer, HyenaDNA) produce sequence embeddings, mean-pooled and fed to a 2-layer FFN adapter. LoRA updates only attention (Q/K/V) and dense layers. Batch padding+attention masks handle variable-length inputs. Dashed arrows denote which layers are trained under different regimens (frozen vs LoRA fine-tuning).
  - Insights: Adapter-only can discriminate short/abundant viruses, but LoRA fine-tuning is crucial for long, low-coverage viruses (EBV, HHV-8). The design is simple, efficient, and tokenizer-agnostic.

- Fig 2. Robustness to contamination
  - Findings: Spiking 0.5–5% bacterial/fungal reads causes only marginal drops in overall accuracy. Given real contamination is typically ~1%, robustness is strong. NextVir-D shows the smallest decline; NextVir-N the largest.
  - Insights: NextVir can act as a front-end filter in mixed-DNA pipelines, safely assigning contaminants to the “non-oncoviral/Human” bucket.

- Fig 3. Binary ROC curves
  - Findings: Under random splits, NextVir matches the best AUCs (on par with DVF). Under context-supported splits (non-overlapping genome segments), NextVir consistently outperforms DVF, XVir, and Virtifier.
  - Insights: Beyond random splits, NextVir generalizes under uneven coverage and locus bias—indicating pattern learning rather than region memorization.

Tables
- Table 1. Dataset composition (genome length, number of 150bp reads)
  - Findings: EBV (~168 kb) and HHV-8 (~135 kb) are long and have few reads (low coverage), thus harder to classify. HBV (~3.2 kb) and HPV (~7.7 kb) are short with high coverage. MCV has very few reads.
  - Insights: Per-class difficulty is driven by coverage. Long and rare viruses need more data or tailored strategies.

- Table 2. Pretrained (frozen) vs LoRA fine-tuned (multiclass)
  - Findings: Adapter-only reaches 70%+ on many classes but struggles on EBV/HHV-8. With LoRA fine-tuning, overall accuracy exceeds 90% (NextVir-D/N ≈95%). EBV nearly doubles; HHV-8 improves markedly but remains the hardest.
  - Insights: Fine-tuning yields the largest gains for long/low-coverage classes. Foundation embeddings transfer reasonably, but task-specific adaptation is pivotal.

- Table 3. Context-supported classification (disjoint genome regions)
  - Findings: Overall accuracy drops (e.g., NextVir-D 94.8%→86.9%) but still far above adapter-only baselines. NextVir-N is relatively strongest. HHV-8 collapses; MCV is largely unaffected.
  - Insights: Generalization holds under locus bias. HHV-8’s collapse reflects “rare + long genome → ultra-low coverage.” Improving coverage is key.

- Table 4. Robustness to mutations (substitutions/indels)
  - Findings: 5–10% substitutions cause modest declines (overall −2–10% points; most classes <−7%p). Adding 5% indels triggers sharp drops. DNABERT-S (BPE) is sensitive to frame shifts; Nucleotide Transformer (non-overlapping k-mers) is particularly vulnerable to indels (HPV worst). HyenaDNA (1-mer) is not consistently more indel-robust than others. Low-coverage/rare classes (EBV, HHV-8, MCV) degrade the most.
  - Insights: Substitution robustness is decent, but indels expose tokenization limits. Indel-aware preprocessing/tokenization will be important for real-world samples.

- Table 5. Binary detection benchmarks
  - Findings: Under random splits, NextVir ≈ DVF in top-tier accuracy/AUC. Under context-supported splits, NextVir (D/N) surpasses all baselines (accuracy ≈91.6%, AUC 0.985–0.989).
  - Insights: Foundation-model-based NextVir is competitive or superior in viral detection, especially in the more realistic, harder split.

Appendix (S1 Appendix)
- Contents (as referenced in the main text)
  - LoRA rank choice: Higher ranks yield marginal gains but much higher compute/time (supports using rank 4).
  - Context-supported split: Alignment-based positional binning and 80/10/10 non-overlapping split procedure with visualization.
  - Optimization/grid search: Details for LR, β1, λ tuning.
  - Additional benchmarks/analyses: Single-species binary detection, uniformly sampled data, simulated viral discovery, HPV detection on experimental data, HHV-8 difficulty analysis, etc.
- Insights: Appendices provide quantitative backing for main claims (e.g., HHV-8 difficulty from rarity+length+coverage; LoRA rank–efficiency trade-off) and practical details to reproduce the results.

<br/>
# refer format:



BibTeX
@article{RobertsonConsulVikalo2025NextVir,
  author    = {Robertson, John and Consul, Shorya and Vikalo, Haris},
  title     = {NextVir: Enabling classification of tumor-causing viruses with genomic foundation models},
  journal   = {PLOS Computational Biology},
  year      = {2025},
  volume    = {21},
  number    = {8},
  pages     = {e1013360},
  doi       = {10.1371/journal.pcbi.1013360},
  url       = {https://doi.org/10.1371/journal.pcbi.1013360},
  publisher = {Public Library of Science},
  note      = {Published August 21, 2025}
}



Robertson, John, Shorya Consul, and Haris Vikalo. 2025. “NextVir: Enabling Classification of Tumor-Causing Viruses with Genomic Foundation Models.” PLOS Computational Biology 21 (8): e1013360. https://doi.org/10.1371/journal.pcbi.1013360.



Robertson, John, Shorya Consul, and Haris Vikalo. “NextVir: Enabling Classification of Tumor-Causing Viruses with Genomic Foundation Models.” PLOS Computational Biology 21, no. 8 (August 21, 2025): e1013360. https://doi.org/10.1371/journal.pcbi.1013360.
