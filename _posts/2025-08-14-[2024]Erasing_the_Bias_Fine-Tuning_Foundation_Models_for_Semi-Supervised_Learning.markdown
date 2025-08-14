---
layout: post
title:  "[2024]Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning"
date:   2025-08-14 18:46:32 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

분포에서 비롯된 bias를 완화하는 파인튜닝 툴..
구체적으로는: 
FINESSL은 파운데이션 모델을 PEFT(VPT)로 미세튜닝하며, 클래스별 동적 마진을 부여하는 Balanced Margin Softmax로 집적 편향을 지우고, 보조 분류기에 Decoupled Label Smoothing과 전 샘플 가중화를 결합해 인지 편차를 줄이며 버려지던 비신뢰 가짜라벨도 활용  


짧은 요약(Abstract) :

- 문제의식: 최근 반지도학습(SSL)은 다양한 기법이 제안됐지만, 실제 배치 시 기대보다 성능이 낮아 쓰기 어렵다는 문제가 있다.
- 핵심 아이디어: 사전학습된 파운데이션 모델을 SSL에 맞게 적응(fine-tuning)시키는 새 방법 FINESSL을 제안한다.
- 관찰한 한계
  - Aggregated biases: 파운데이션 모델이 사전학습 데이터의 불균형을 내재하고 있어, SSL 훈련 중 확인편향으로 특정 클래스의 의사라벨이 과도하게 쏠리는 현상이 누적·증폭됨.
  - Cognitive deviation: 모델 신뢰도(확률)가 실제 난이도나 정오 여부를 제대로 반영하지 못해 임계치 기반 의사라벨 선택이 실패하기 쉬움(과신, OOD 구분 어려움 등).
- 제안 기법
  - Balanced Margin Softmax: 클래스별로 균형 잡힌 마진을 주어 어려운 클래스를 보상하고, 의사라벨 분포의 쏠림(편향 누적)을 완화.
  - Decoupled Label Smoothing: 주분기와 분리된 보조 분류기에 라벨 스무딩을 적용해 신뢰도 왜곡을 줄이고(과신 완화), 더 신뢰도있는 의사라벨 활용을 유도.
- 결과: 여러 벤치마크에서 SOTA 달성, 학습 비용 6배 이상 감소, 다양한 파라미터 효율 튜닝(PEFT) 및 최신 SSL 알고리즘과 매끄럽게 통합 가능.
- 코드: https://github.com/Gank0078/FineSSL


- Motivation: Despite many advances in semi-supervised learning (SSL), real-world deployment often underperforms.
- Core idea: FINESSL adapts pre-trained foundation models specifically for SSL to close this deployment gap.
- Identified issues
  - Aggregated biases: Pretraining-induced class imbalance is amplified during SSL via confirmation bias, skewing pseudo-labels.
  - Cognitive deviation: Model confidence is poorly aligned with true task difficulty and correctness (overconfidence, OOD confusion), hurting threshold-based pseudo-label selection.
- Proposed methods
  - Balanced Margin Softmax: Adds adaptive, class-specific margins to favor harder classes and debias pseudo-label distributions.
  - Decoupled Label Smoothing: Applies label smoothing to an auxiliary (decoupled) classifier to regularize confidence and mitigate overconfidence without harming representation learning.
- Results: State-of-the-art performance on multiple benchmarks, over 6× reduction in training cost, and seamless integration with various PEFT and modern SSL methods.
- Code: https://github.com/Gank0078/FineSSL


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




핵심 요약
- 무엇을 해결하나: 파운데이션 모델(예: CLIP-ViT)을 반정규지도(SSL)로 미세튜닝할 때, (1) 클래스 간 불균형이 의사라벨에 누적·증폭되는 “집적 편향(aggregated biases)”과 (2) 신뢰도 점수가 과도 또는 오판정되어 임계값 기반 선택이 어려워지는 “인지 편차(cognitive deviation)”를 동시에 해결.
- 무엇을 제안하나: FINESSL. 두 축으로 해결:
  1) Balanced Margin Softmax(BMS): 클래스 학습 속도에 따라 동적으로 마진을 부여해 모델 자체의 클래스 편향을 직접 상쇄.
  2) Decoupled Label Smoothing(DLS): 본 분기에서 분리된 보조 분류기를 라벨 스무딩으로 학습시켜 신뢰도-난이도 정렬을 유도하고, 그 확률로 모든 비라벨 데이터를 가중 재활용.
- 어떻게 쓰나: 파운데이션 모델은 고정(freeze), PEFT(대표적으로 VPT-deep)로 적은 파라미터만 학습. FixMatch/FlexMatch/FreeMatch/SoftMatch/DebiasPL 등 최신 SSL과 매끄럽게 결합.

문제 정의와 동기
- 집적 편향: 파운데이션 모델의 사전학습 데이터 편향이 미세튜닝 중 확인편향(confirmation bias)으로 증폭되어 특정 클래스에 의사라벨이 몰림.
- 인지 편차: (전반) 과소/과대신뢰로 과제 난이도 추정이 왜곡, (개별) 정답/오답/OOD의 신뢰도 분포가 겹쳐 임계값 기반 선택이 부정확.

모델과 아키텍처
- 백본: CLIP의 ViT 이미지 인코더만 사용(고정).
- PEFT: Visual Prompt Tuning(VPT-deep) 기본 채택. 각 ViT 레이어 입력에 학습 가능한 프롬프트 토큰을 전층에 삽입해 적은 파라미터로 표현을 적응. 최종 선형 분류기만 함께 학습.
- 이중 분기 구조:
  - 메인 브랜치: 의사라벨 예측 및 주 손실 최적화(BMS 적용).
  - 보조 분기(auxiliary classifier): 백본 출력 위에 별도의 선형층. 이 분기는 라벨 스무딩으로 학습하며, 역전파를 백본으로 “차단(detach)”해 표현학습을 훼손하지 않음. 보조 분기의 확률은 비라벨 샘플 가중치 산출에만 사용.

학습 데이터와 증강
- 레이블 소량 + 비라벨 대량의 표준 SSL 설정. 약/강 증강(weak ω, strong Ω) 뷰를 사용해 일관성 정규화.
- 벤치마크: CIFAR-10/100, FOOD-101, Semi-Aves(불균형·OOD 포함), ImageNet(라벨 1%, 10%).

핵심 기법 1: Balanced Margin Softmax(BMS)
- 목표: 임계값 조정이 아닌 “모델 자체의 클래스 선호”를 직접 교정하여, 의사라벨 분포를 균형화.
- 클래스 학습 속도 σ_t(k): 시간 t에 클래스 k로 “충분히 자신 있는”(ζ=0.7) 의사라벨 개수. β_t(k)=σ_t(k)/max_k σ_t(k)를 정규화.
- 클래스 마진 Δ_t(y)=1−β_t(y): 덜 학습된 클래스일수록 큰 마진.
- 스케일 α_t: 클래스별 학습속도 격차가 클수록 크게. α_t=(max β_t−min β_t)·α_base.
- 손실: 표준 cross-entropy를 마진이 반영된 softmax로 치환. BMS는 지도 손실(레이블데이터)과 일관성 정규화(비라벨) 모두에 적용해 클래스 균형을 유도.

핵심 기법 2: Decoupled Label Smoothing(DLS) + 전량 재가중 학습
- 보조 분기 학습: 라벨 스무딩(λ)된 감독 손실 및 “부드러운” 의사라벨 기반의 일관성 정규화로 보조 분기만 학습. 백본으로의 기울기 전파는 차단.
- 전량 재가중: 임계값으로 비라벨을 버리지 않고, 보조 분기 확률의 최대값으로 중요도 ψ(u)=γ·max p_aux(y|ω(u))를 계산하여 모든 비라벨 샘플을 가중 학습. 메인 브랜치의 비라벨 손실은 BMS + 가중 합으로 계산.
- 직관: 스무딩으로 과잉 확신을 완화한 보조 분기 확률은 “정/오/OOD”를 더 잘 분리하는 가중치로 기능. 메인 브랜치는 표현을 해치지 않고 안정적으로 지도.

전체 목표함수와 알고리즘 개요
- 총손실 L = (메인) 지도 BMS + (메인) 가중 일관성 BMS + (보조) 지도(스무딩) + (보조) 스무딩 일관성.
- 루프(요약):
  1) 현재 배치에서 클래스별 학습 속도 σ_t 업데이트(ζ 기준).
  2) β_t 정규화, Δ_t와 α_t 계산.
  3) 보조 분기 확률로 비라벨 샘플 가중치 ψ 산출.
  4) 메인 브랜치: BMS로 지도·비지도(가중) 손실 계산.
  5) 보조 분기: 라벨 스무딩으로 지도·비지도 손실 계산(백본 detach).
  6) 합성 손실로 PEFT 파라미터와 분류기만 SGD 업데이트.

학습 설정(대표)
- VPT 길이 50, SGD(lr 0.03, momentum 0.9, wd 5e-4), 배치 32, 에폭 30, 에폭당 500 스텝.
- 기본 하이퍼파라미터: ζ=0.7, λ=0.5, α_base=8.0, γ=3.0, μ(비라벨/라벨 배치비)=1(일부 ImageNet=2).

무엇이 특별한가
- 임계값 조정이 아니라 “마진”으로 모델 선호를 직접 교정해 클래스 균형을 유도(BMS).
- 라벨 스무딩을 본 분기에서 분리·차단(DLS)하여 표현학습 저해 없이 신뢰도-난이도 정렬.
- 비라벨 전량을 보조 분기 확률로 재가중해 활용(낮은 신뢰도도 정보로 사용).
- 파운데이션 모델의 효율적 미세튜닝(PEFT)과 현대 SSL의 플러그앤플레이 통합.

경험적 하이라이트(논문 보고)
- 다양한 벤치마크에서 SOTA 달성. 초저라벨(CIFAR-10, N1)에서 큰 격차.
- WRN을 스크래치로 학습 대비 약 1/6 훈련시간에 수렴(ViT+VPT 기반).
- OOD 존재(OpenSSL, Semi-Aves 혼합)에서도 DLS로 정/오/OOD 분리가 선명해져 성능 향상.
- VPT 외 Lora/Adapter/AdaptFormer, ResNet(BN tuning/BitFit/SSF) 등 다양한 미세튜닝 대안과도 호환.

한계(논문 언급)
- 주로 CLIP의 이미지 인코더와 ViT 기반에 초점. 텍스트 인코더 활용, 다른 사전학습(MoCo/SimCLR)과의 결합은 후속과제.


Core Idea
- Problem: Fine-tuning foundation models for SSL suffers from (1) aggregated biases—class-imbalance in pseudo-labels amplified during training, and (2) cognitive deviation—misaligned confidence both globally (task difficulty) and locally (correct vs. incorrect vs. OOD), making threshold-based selection unreliable.
- Proposal: FINESSL with two key components:
  1) Balanced Margin Softmax (BMS): directly debias the model’s class preference by injecting adaptive, per-class margins derived from class learning pace.
  2) Decoupled Label Smoothing (DLS): train an auxiliary, detached classifier with label smoothing to regularize confidence and produce sample weights, enabling weighted use of all unlabeled data.
- Practicality: Freeze the foundation backbone and adapt only small PEFT modules (default: VPT-deep) and the linear head. FINESSL is plug-and-play with FixMatch, FlexMatch, FreeMatch, SoftMatch, DebiasPL.

Model and Architecture
- Backbone: CLIP ViT image encoder (frozen).
- PEFT: Visual Prompt Tuning (VPT-deep). Learnable prompts are inserted at all Transformer layers; only prompts and the final linear classifier are trained.
- Two branches:
  - Main branch: produces pseudo-labels and is optimized with BMS.
  - Auxiliary branch: a separate linear head on top of the frozen backbone (with its own PEFT params), trained with label smoothing; gradients are detached from the backbone. Its probabilities are used solely to compute sample weights for unlabeled data.

Data and Augmentation
- Standard SSL setting: few labeled, many unlabeled. Use weak/strong augmentations for consistency.
- Benchmarks: CIFAR-10/100, FOOD-101, Semi-Aves (long-tailed and OOD), ImageNet (1%, 10% labels).

Key Technique 1: Balanced Margin Softmax (BMS)
- Goal: Produce class-balanced pseudo-labels by rectifying model preferences, not by tweaking confidence thresholds.
- Class learning pace σ_t(k): number of unlabeled samples assigned to class k with confidence ≥ ζ (ζ=0.7). Normalize β_t(k)=σ_t(k)/max_k σ_t(k).
- Margin per class Δ_t(y)=1−β_t(y): larger for slower-learning classes.
- Dynamic scaling α_t=(max β_t−min β_t)·α_base: strengthen margins when class pace disparity is larger.
- Loss: Replace standard cross-entropy by a margin-augmented softmax. Apply BMS to both supervised loss and consistency loss in the main branch.

Key Technique 2: Decoupled Label Smoothing (DLS) and Full Reweighting
- Auxiliary branch: trained with label smoothing (λ) on both supervised and soft-consistency objectives. Detach gradients to avoid harming representation learning.
- Full reweighting: do not discard low-confidence unlabeled data; weight every unlabeled example by ψ(u)=γ·max p_aux(y|weak(u)) from the auxiliary branch. Main-branch unlabeled loss is a BMS loss weighted by ψ.
- Intuition: Smoothed auxiliary probabilities yield a more discriminative separation among correct/incorrect/OOD, serving as reliable importance weights; main branch focuses on representation and debiased decision boundaries.

Objective and Training Loop
- Total loss L = (main) supervised BMS + (main) weighted consistency BMS + (aux) supervised with smoothing + (aux) consistency with smoothing.
- Algorithm (per iteration):
  1) Update σ_t by counting confident pseudo-labels.
  2) Compute β_t, Δ_t, and α_t.
  3) Compute unlabeled weights ψ from the auxiliary branch.
  4) Compute main-branch losses (BMS).
  5) Compute auxiliary losses (with smoothing; backbone detached).
  6) Update PEFT modules and classifier via SGD.

Training Setup (typical)
- VPT length 50; SGD with lr 0.03, momentum 0.9, weight decay 5e-4; batch size 32; 30 epochs, 500 steps/epoch.
- Defaults: ζ=0.7, λ=0.5, α_base=8.0, γ=3.0, unlabeled/labeled batch ratio μ=1 (ImageNet uses μ=2).

Why It’s Distinctive
- Debiases the model directly via BMS instead of moving thresholds.
- Decouples label smoothing to preserve representation quality while aligning confidence with task difficulty.
- Reuses all unlabeled data through principled weighting (not hard thresholding).
- Works seamlessly with modern SSL methods and various PEFT schemes; efficient and robust.

Empirical Highlights (as reported)
- State-of-the-art or highly competitive on CIFAR-10/100, FOOD-101, Semi-Aves, and ImageNet low-label settings.
- About 6× reduction in training time compared to training WideResNet from scratch.
- Improved pseudo-label balance and confidence quality; strong under OOD presence thanks to DLS.
- Compatible with other PEFT (AdaptFormer, LoRA, Adapter) and with ResNet via BN-tuning, BitFit, SSF.

Limitations (noted by authors)
- Focused mainly on CLIP’s image encoder and ViT backbone; exploring text encoder and other pretraining (e.g., MoCo/SimCLR) is future work.


<br/>
# Results



1) 평가 셋업(Backbone/파인튜닝/기준선/지표/예산)
- 백본과 파인튜닝
  - CLIP의 ViT 이미징 인코더를 백본으로 사용. 백본 파라미터는 고정(frozen).
  - 기본 파인튜닝 전략으로 VPT(Visual Prompt Tuning, deep variant, prompt 길이 50) 채택.
  - 다양한 PEFT 대안(LoRA, Adapter, AdaptFormer)과 ResNet-CLIP에 맞춘 경량 파인튜닝(BN Tuning, BitFit, SSF)도 비교.
- 비교 대상(경쟁 모델)
  - Pseudo-Label (PL), FixMatch, FlexMatch, FreeMatch, SoftMatch, DebiasPL.
  - 모든 경쟁 기법은 동일한 일반화된 파인튜닝 프레임워크 내에서 구현·비교됨(동일한 훈련 예산과 설정).
- 데이터셋과 라벨 레짐
  - 표준: CIFAR-10(N1/N2/N4), CIFAR-100(N4/N25/N100), FOOD-101(N2/N4/N10).
  - 복잡: Semi-Aves(비대칭 클래스/ID+OOD 혼재 설정 포함), ImageNet(1%·10% 라벨).
  - 추가 분석: OpenSSL(절반 OOD 클래스; CIFAR-100 N5/N10), Long-Tailed SSL(CIFAR-100, 불균형 비율 ρ), ViT scratch 학습 비교 등.
- 평가지표
  - Top-1 정확도(테스트셋), 3회 반복 평균±표준편차.
- 학습 예산·환경
  - 30 epochs × 500 steps(이전 scratch 1024×1024 대비 대폭 단축).
  - 공통 옵티마이저·하이퍼파라미터 공유(SGD, lr 0.03, wd 5e-4, momentum 0.9 등).
  - 동일 프레임워크에서의 FixMatch 등은 높은 기준 임계치(0.95) 대신 0.7을 사용해 각 방법의 성능을 최대화.

2) 표준 벤치마크 결과(정량)
- CIFAR-10(N1/N2/N4)
  - FINESSL: 96.15 / 96.87 / 97.39.
  - 초저라벨(N1)에서 기존 방법 대비 약 18%p 내외의 대폭 향상(논문 본문 서술). 간단한 과제(N4)에서는 상위 방법들과 유사한 고정확도 구간에서 근소 차이.
- CIFAR-100(N4/N25/N100)
  - FINESSL: 80.44 / 84.51 / 86.66.
  - 동 세팅에서 FixMatch/FlexMatch/FreeMatch/SoftMatch 대비 전반적 우위. DebiasPL 대비 다수 세팅에서 평균 약 3%p 상회(본문 서술).
- FOOD-101(N2/N4/N10)
  - FINESSL: 87.04 / 89.28 / 89.69.
  - 전반적으로 최신 기법들 대비 우세.

3) 복잡 데이터셋 결과(정량)
- Semi-Aves(불균형·细粒度, OOD 혼입 시나리오 포함)
  - Du=Din(전부 ID): FixMatch 65.52, DebiasPL 66.82, FINESSL 67.25(최고).
  - Du=Din∪Dout(ID+OOD): FixMatch 60.15, DebiasPL 60.85, FINESSL 61.12(최고).
  - OOD 혼입 상황에서도 안정적 우위.
- ImageNet(대규모, 1%/10% 라벨)
  - 1%: FixMatch/FlexMatch/FreeMatch/SoftMatch ≈ 73.19~73.21, DebiasPL 73.58, FINESSL 74.22.
  - 10%: FixMatch/FlexMatch/FreeMatch/SoftMatch ≈ 78.49~78.77, DebiasPL 78.72, FINESSL 79.21.
  - 대규모에서도 소폭이지만 일관된 SOTA.

4) 품질 분석과 추가 실험(정성+정량)
- 의사라벨 품질
  - FINESSL은 학습 전반에 걸쳐 더 높은 의사라벨 정확도와 더 높은 엔트로피(클래스 분포 균형화)를 달성(그림 3a/3b).
- OpenSSL(절반 OOD 클래스)에서의 이점
  - CIFAR-100 N5/N10에서 DLS(Decoupled Label Smoothing) 제거 시 평균 1.87%p 성능 하락 → DLS가 OOD 구분·가중치 할당을 개선(그림 4a).
  - DLS 도입 시 올바른/오라클-오류/OOD 샘플 간 신뢰도 분포 경계가 뚜렷해짐(그림 4b~4d).
- Long-Tailed SSL
  - CIFAR-100 불균형(ρ=10/20)에서 평균 약 +2.95%p(SoftMatch 대비) 성능 향상(표 9). DebiasPL도 강하나 다수 설정에서 FINESSL이 상회.
- ResNet-CLIP(비-Transformer) 기반에서도 유효
  - BN Tuning / BitFit / SSF 기반 파인튜닝 비교에서 FINESSL이 DebiasPL 대비 평균 +2.19%p 우세(표 7). 예: BN Tuning N4 55.21(우리) vs 52.90(DebiasPL), N25 67.06 vs 66.62, N100 70.60 vs 70.34 등.
- 다른 PEFT 전략과의 호환성
  - VPT 외에 AdaptFormer/LoRA/Adapter 모두 결합 가능(표 4). 일부 세팅에서는 AdaptFormer가 VPT를 근소 상회하나 전체적으로 프레임워크 일관 호환·개선.
- 효율성(시간)
  - WideResNet을 scratch로 학습하는 FixMatch 대비 약 1/6 훈련 시간. 총 시간: FixMatch(WideResNet) 55,574.5s vs FixMatch(VPT) 9,240.0s vs FINESSL(VPT) 9,630.0s(표 5).
- 제거/대체 실험(기여 요소)
  - Balanced Margin Softmax(BMS) 제거 시 특히 소라벨(CIFAR-100 N4)에서 −3.74%p(표 6) → 클래스 편향(aggregated biases) 완화가 핵심.
  - 라벨 스무딩을 메인 가지에 직접 적용하면 표현학습 저해. 보조 분류기(aux)로 분리·detach한 DLS가 성능·안정성에 기여(특히 OOD 혼입 시 +1.63%p, 표 6).
  - 동적 스케일 α, 감독손실 쪽의 마진 반영 등도 모두 유의미한 향상에 기여(표 6). 하이퍼파라미터(λ, α0 등) 감도는 낮음(그림 3c/3d).

5) 추가 관찰
- ViT scratch 학습은 SSL에서 긴 학습에도 성능이 크게 낮음(시간·성능 모두 비효율).
- FFT를 잘 쓰는 팁: LP-FT(선형분류기 학습 후 전량 미세조정) 또는 의미기반 초기화(FFT†)가 FFT 성능을 크게 끌어올림. 그럼에도 동일 전략에서 FINESSL이 FixMatch 대비 평균 +4.59%p 우세(표 10).
- ImageNet-21k 사전학습 ViT에서도 FINESSL이 DebiasPL 대비 평균 +0.36%p 우위(표 11).

요약 결론
- FINESSL은 Top-1 정확도 기준으로 CIFAR-10/100, FOOD-101, Semi-Aves, ImageNet 전반에서 SOTA 또는 동급의 최고 성능을 달성.
- 특히 초저라벨(CIFAR-10 N1)·불균형·OOD 혼입(OpenSSL) 상황에서 강인함을 입증.
- VPT 등 PEFT와 결합 시 학습비용을 약 6배 절감하면서도, 균형 마진 소프트맥스(BMS)로 클래스 편향을 완화하고, 분리형 라벨 스무딩(DLS)+가중 샘플링으로 신뢰도 편차(인지 편차)를 줄여 의사라벨 선택·활용을 전반적으로 개선.




1) Evaluation setup (backbone/fine-tuning/baselines/metric/budget)
- Backbone and fine-tuning
  - CLIP vision encoder (ViT) as backbone; all backbone weights frozen.
  - Default PEFT is deep Visual Prompt Tuning (VPT, prompt length 50).
  - Also compare other PEFTs (LoRA, Adapter, AdaptFormer) and lightweight tuning for ResNet-CLIP (BN Tuning, BitFit, SSF).
- Competitors (baselines)
  - Pseudo-Label (PL), FixMatch, FlexMatch, FreeMatch, SoftMatch, DebiasPL.
  - All baselines re-implemented under the same fine-tuning framework with shared training budget.
- Datasets and label regimes
  - Standard: CIFAR-10 (N1/N2/N4), CIFAR-100 (N4/N25/N100), FOOD-101 (N2/N4/N10).
  - Complex: Semi-Aves (class-imbalance, with or without OOD unlabeled), ImageNet (1%/10% labels).
  - Additional analyses: OpenSSL (half OOD classes; CIFAR-100 N5/N10), Long-Tailed SSL (CIFAR-100), ViT-from-scratch comparisons.
- Metric
  - Top-1 accuracy on test sets; mean ± std over 3 runs.
- Training budget and environment
  - 30 epochs × 500 steps (much less than 1024 × 1024 for training from scratch).
  - Shared optimizer/hyperparams across methods (SGD, lr 0.03, wd 5e-4, momentum 0.9).
  - For fairness, threshold 0.7 (instead of 0.95) is used in the unified framework when it yields better baseline performance.

2) Main quantitative results on standard benchmarks
- CIFAR-10 (N1/N2/N4)
  - FINESSL: 96.15 / 96.87 / 97.39.
  - In the extremely low-label N1 setting, FINESSL improves by about 18 percentage points over prior methods (per paper text); for N4 (easier), several methods are close at high accuracy.
- CIFAR-100 (N4/N25/N100)
  - FINESSL: 80.44 / 84.51 / 86.66.
  - Consistently surpasses FixMatch/FlexMatch/FreeMatch/SoftMatch; on average about +3 points over DebiasPL across many settings (per paper text).
- FOOD-101 (N2/N4/N10)
  - FINESSL: 87.04 / 89.28 / 89.69.
  - Outperforms state-of-the-art competitors overall.

3) Results on complex datasets
- Semi-Aves (fine-grained, imbalanced, OOD)
  - Du=Din (all ID unlabeled): FixMatch 65.52, DebiasPL 66.82, FINESSL 67.25 (best).
  - Du=Din∪Dout (with OOD): FixMatch 60.15, DebiasPL 60.85, FINESSL 61.12 (best).
- ImageNet (1%/10% labels)
  - 1%: FixMatch/FlexMatch/FreeMatch/SoftMatch ≈ 73.19–73.21, DebiasPL 73.58, FINESSL 74.22.
  - 10%: FixMatch/FlexMatch/FreeMatch/SoftMatch ≈ 78.49–78.77, DebiasPL 78.72, FINESSL 79.21.
  - Shows consistent SOTA or close-to-SOTA on large-scale data.

4) Quality and auxiliary analyses
- Pseudo-label quality
  - FINESSL yields higher pseudo-label accuracy and higher entropy (i.e., more balanced class distribution) throughout training (Fig. 3a/3b).
- OpenSSL (half OOD classes)
  - Removing DLS degrades accuracy by 1.87% on average in N5/N10 (Fig. 4a). DLS produces clearer confidence separation among correct/wrong/OOD samples (Fig. 4b–4d), leading to better sample reweighting.
- Long-Tailed SSL
  - On CIFAR-100 with imbalance (ρ=10/20), FINESSL outperforms SoftMatch by ≈ +2.95% on average (Tab. 9). DebiasPL is competitive but FINESSL wins in most settings.
- ResNet-CLIP (non-Transformer)
  - With BN Tuning / BitFit / SSF, FINESSL consistently outperforms DebiasPL by +2.19% on average (Tab. 7). Examples: BN Tuning N4 55.21 (ours) vs 52.90 (DebiasPL); N25 67.06 vs 66.62; N100 70.60 vs 70.34.
- Alternative PEFTs
  - AdaptFormer/LoRA/Adapter integrate smoothly; AdaptFormer occasionally edges out VPT in some settings, confirming method generality (Tab. 4).
- Efficiency (time)
  - ≈ 6× reduction in training time vs training FixMatch from scratch with WideResNet. Total time: FixMatch(WideResNet) 55,574.5s vs FixMatch(VPT) 9,240.0s vs FINESSL(VPT) 9,630.0s (Tab. 5).
- Ablations (what matters)
  - Removing Balanced Margin Softmax (BMS) hurts especially with scarce labels (e.g., −3.74% at CIFAR-100 N4; Tab. 6), showing it mitigates aggregated class biases.
  - Decoupled Label Smoothing (DLS) must be on an auxiliary, detached head; applying label smoothing directly to the main branch degrades representation learning. DLS is particularly helpful with OOD unlabeled (e.g., +1.63% on Semi-Aves with OOD; Tab. 6).
  - Dynamic scaling α and applying margins also in supervised loss further improve robustness and balance (Tab. 6). Hyperparameters show low sensitivity (Fig. 3c/3d).

5) Additional observations
- ViT-from-scratch under SSL performs poorly despite long training; inefficient in both accuracy and time.
- For FFT, LP-FT or semantic-aware initialization (FFT†) dramatically improve FFT; even then, FINESSL still surpasses FixMatch by ≈ +4.59 points on average under those strategies (Tab. 10).
- With ImageNet-21k pretraining, FINESSL remains best on average (+0.36 points vs DebiasPL; Tab. 11).

Bottom line
- FINESSL achieves SOTA or near-SOTA Top-1 accuracy across CIFAR-10/100, FOOD-101, Semi-Aves, and ImageNet, with pronounced gains in extremely low-label, imbalanced, and OOD-heavy scenarios.
- With PEFT (e.g., VPT), it cuts training cost by about 6×, while Balanced Margin Softmax reduces aggregated class biases, and Decoupled Label Smoothing plus sample reweighting mitigate confidence misalignment (cognitive deviation) to improve pseudo-label selection and utilization.


<br/>
# 예제




1) 기본 문제 설정과 데이터(입력/출력)
- 과제(테스크): 이미지 분류용 반지도학습(SSL). C개의 클래스로 분류.
- 입력
  - 라벨 있는 데이터 Dl = {(x_i, y_i)}: 이미지 x_i, 정답 레이블 y_i(원-핫 벡터, 길이 C).
  - 라벨 없는 데이터 Du = {u_j}: 이미지 u_j만 존재, 레이블 없음.
  - 각 학습 스텝에서 라벨 배치 B개, 언라벨 배치 μB개 사용(논문 기본은 μ=1, ImageNet만 μ=2).
  - 이미지 전처리: 약(weak) 증강 ω(·), 강(strong) 증강 Ω(·) 두 뷰를 생성(SSL 일관성 규제용). 구체 증강 기법은 프레임워크 호환 수준으로 취급(논문은 뷰 두 개 사용만 명시).
- 출력
  - 모델 f(x; {Θ, θ})의 소프트맥스 확률 p(y|x) 길이 C 벡터(클래스별 확률).
  - 의사라벨(bq_j): 언라벨 이미지 u_j에 대한 약증강 예측 q_j = p(y|ω(u_j))의 argmax 클래스.
  - 최종 테스트 출력: 테스트 이미지당 1개 예측 클래스(Top-1). 성능 지표는 Top-1 Accuracy.

2) 모델과 미세조정(PEFT)
- 백본: CLIP의 이미지 인코더(비전 트랜스포머, ViT). 사전학습 파라미터 Θ는 고정.
- 파라미터 효율 미세조정(PEFT): VPT-deep 기본 사용(프롬프트 길이 50). 학습 파라미터 θ에는 프롬프트와 최종 선형 분류기 포함.
- 보조 분류기(auxiliary classifier): faux(x; {Θ, θ′})를 추가(한 층의 FC 헤드). θ′는 보조 모듈 및 해당 분류기 파라미터. 이 분기에서만 라벨 스무딩을 적용하고, 메인 표현부로 그래디언트가 흘러가지 않도록 분리(detach).

3) 학습 목표와 손실(입력→출력→손실 연결)
- 감독(supervised) 손실(라벨 배치 B개 사용)
  - 표준 교차엔트로피 대신 “Balanced Margin Softmax(BMS)” 적용.
  - 핵심 아이디어: 클래스별 학습 속도(쉬움/어려움)를 추정해, 어려운 클래스에 더 큰 마진을 부여하여 모델의 편향(aggregated biases)을 줄임.
- 비감독(unsupervised) 손실(언라벨 배치 μB개 사용)
  - 메인 분기: 강증강 예측 f(Ω(u_j))에 의사라벨(bq_j)로 일관성 손실을 계산.
  - FINESSL의 차별점: 모든 언라벨 샘플을 사용하되, 보조 분류기의 신뢰도에서 도출한 가중치 ψ(u_j)=γ·max(p_aux(y|ω(u_j)))로 재가중하여 품질-수량 트레이드오프를 동시에 달성(낮은 품질 샘플은 작은 가중치).
- 보조 분기 손실(Decoupled Label Smoothing, DLS)
  - 보조 분기에서만 라벨 스무딩된 연속 의사라벨 e(q_j) 사용(λ∈(0,1), 논문 기본 0.5).
  - 목적: 과신/인지 편차(cognitive deviation)를 완화해 신뢰도(확률)와 실제 난이도 간의 정렬을 개선. 이 보조 분기 손실의 그래디언트는 메인 표현부로 전파되지 않음(detach).

4) FINESSL 고유 구성요소 계산 예시(작동 방식)
- 클래스별 학습 속도 σ_t(k)
  - 시점 t에서 언라벨 예측 q_j 중 “max(q_j) ≥ ζ(논문 기본 0.7)”이면서 argmax=q_j=k인 개수를 카운트.
  - 정규화: β_t(k) = σ_t(k) / max_k σ_t(k)
  - 클래스 마진: Δ_t(k) = 1 − β_t(k) (배정량이 적을수록 Δ가 큼 → 어려운 클래스)
  - 스케일 α_t = (max_k β_t(k) − min_k β_t(k)) · α (논문 기본 α=8.0)
  - 감독/일관성 손실의 목표 클래스에 이 동적 마진을 반영하여, 어려운 클래스를 더 잘 학습하게 유도.
- 보조 분기 라벨 스무딩 의사라벨
  - e(q_j)[k] = (1−λ)+λ/C (k=bq_j일 때), 그 외는 λ/C.
- 언라벨 재가중
  - ψ(u_j) = γ·max(Softmax(faux(ω(u_j)))) (논문 기본 γ=3.0)

5) 하나의 학습 스텝(미니배치) 예
- 입력
  - 라벨 배치 B=32: {(x_i, y_i)}_{i=1..32}
  - 언라벨 배치 μB=32: {u_j}_{j=1..32}
- 처리
  1) 각 u_j에 대해 q_j = p(y|ω(u_j)) 계산, bq_j = argmax(q_j) 산출
  2) σ_t(k) 집계(ζ=0.7 넘는 예측만 카운팅) → β_t(k), Δ_t(k), α_t 계산
  3) 라벨 배치에 대해 BMS 감독 손실 계산
  4) 메인 분기 언라벨 손실: bq_j와 f(Ω(u_j))로 일관성 손실 계산, 각 샘플 가중치 ψ(u_j) 곱해 합산
  5) 보조 분기 언라벨 손실: e(q_j)와 faux(Ω(u_j))로 라벨 스무딩된 일관성 손실 계산(메인 표현부로는 detach)
  6) 총손실 = (메인: 감독+BMS 일관성) + (보조: 감독+스무딩 일관성)
  7) θ, θ′만 최적화(Θ는 동결)
- 출력
  - 이번 스텝의 스칼라 총손실 값
  - 역전파 후 갱신된 θ, θ′
  - 누적 σ_t(k) 업데이트

6) 테스트(추론)
- 입력: 테스트 이미지 x
- 출력: p(y|x)에서 argmax 클래스(Top-1), 지표는 Top-1 Accuracy

7) 구체 시나리오 A: CIFAR-100, N4 설정(논문 표준 벤치마크)
- 데이터
  - 클래스 C=100
  - 라벨: 클래스당 4장 → Dl 크기 400
  - 언라벨: 나머지 학습 이미지를 Du로 사용
- 입력 형태
  - 라벨 배치: 32개 RGB 이미지, y는 길이 100의 원-핫 벡터
  - 언라벨 배치: 32개 RGB 이미지(의사라벨은 학습 중 생성)
- 출력
  - 각 스텝마다 손실
  - 전체 학습 후 테스트셋 Top-1 Accuracy
- 기대 효과
  - BMS로 클래스별 의사라벨 분포의 불균형(aggregated biases) 완화
  - DLS+재가중으로 과신/인지 편차(cognitive deviation)와 OOD 섞임에도 강건

8) 구체 시나리오 B: Semi-Aves(두 가지 언라벨 구성)
- Du = Du_in: 언라벨이 라벨과 동일한 클래스 분포(세분류 조류 데이터, 클래스 불균형 큼)
- Du = Du_in ∪ Du_out: 언라벨에 새로운 클래스(OOD)도 혼재
- 입력/출력
  - 입력은 위와 동일(이미지)
  - 출력은 Top-1 Accuracy
- 기대 효과
  - DLS로 OOD 혼재 상황에서 언라벨 샘플의 가중치가 보수적으로 내려가며 오분류 샘플 영향이 감소

9) 구체 시나리오 C: OpenSSL(CIFAR-100, N5/N10)
- 설정
  - 100개 클래스 중 50개만 “보이는(Seen, ID)” 클래스로 학습(이들에 라벨 N5 또는 N10 제공)
  - 나머지 50개는 OOD로 언라벨에 섞임
- 입력/출력
  - 입력은 동일
  - 출력은 Seen 클래스에 대한 테스트Accuracy
- 기대 효과
  - DLS로 ID/오라벨/OOD 간 신뢰도 경계가 선명해져 재가중 품질이 향상

10) 구체 시나리오 D: 장꼬리(LTSSL)
- 예: CIFAR-100에서 라벨/언라벨 각각 불균형 비율 ρ 설정(예: 10, 20)
- 입력/출력
  - 입력은 동일
  - 출력은 Top-1 Accuracy
- 기대 효과
  - BMS로 클래스 간 학습 속도 정렬 → 장꼬리 환경에서도 의사라벨 분포가 더 균형적으로 개선

11) 학습 설정(논문 기본값)
- 옵티마이저: SGD(lr=0.03, momentum=0.9, weight decay=5e-4), 코사인 스케줄
- 배치: 기본 32(데이터셋 따라 약간 변동), μ=1(대부분), Epoch=30, 각 Epoch 500 스텝
- FINESSL 핵심 하이퍼파라미터: ζ=0.7(학습 속도 카운트 임계값), α=8.0(마진 스케일 기본), λ=0.5(DLS 강도), γ=3.0(언라벨 재가중 스케일)
- 백본/미세조정: CLIP ViT, VPT-deep(프롬프트 길이 50), Θ 동결, θ·θ′만 업데이트

12) 정리: 입·출력 요약
- 트레이닝 입력
  - 라벨 배치: RGB 이미지 텐서, 원-핫 벡터 레이블
  - 언라벨 배치: RGB 이미지 텐서
- 트레이닝 출력
  - 스텝 손실, 갱신된 미세조정 파라미터, 의사라벨/가중치 내부 생성
- 테스트 입력/출력
  - 입력: RGB 이미지
  - 출력: 예측 클래스(Top-1), 평가 지표: Top-1 Accuracy





1) Task and data (inputs/outputs)
- Task: Semi-supervised image classification with C classes.
- Inputs
  - Labeled set Dl = {(x_i, y_i)}: image x_i and one-hot label y_i (length C).
  - Unlabeled set Du = {u_j}: image u_j only (no labels).
  - Each training step uses a labeled mini-batch of size B and an unlabeled mini-batch of size μB (μ=1 by default; μ=2 only for ImageNet in the paper).
  - Two views per image for SSL: a weak augmentation ω(·) and a strong augmentation Ω(·) (for consistency regularization).
- Outputs
  - Model softmax probability p(y|x) of length C.
  - Pseudo-label bq_j for an unlabeled image u_j defined as argmax over q_j = p(y|ω(u_j)).
  - Final test output: Top-1 predicted class per test image; the metric is Top-1 Accuracy.

2) Model and parameter-efficient fine-tuning
- Backbone: CLIP image encoder (ViT). Pretrained parameters Θ are frozen.
- PEFT: VPT-deep by default (prompt length 50). Trainable parameters θ include the prompts and the final linear classifier.
- Auxiliary classifier faux(x; {Θ, θ′}): a single FC head added to the feature extractor, trained with label smoothing. It is decoupled so its gradients do not flow back to the shared representation.

3) Training objectives (how inputs become outputs/losses)
- Supervised loss (on labeled mini-batch)
  - Replace standard cross-entropy with Balanced Margin Softmax (BMS).
  - Idea: estimate per-class learning pace and impose larger margins for harder/slower classes to mitigate aggregated biases.
- Unsupervised loss (on unlabeled mini-batch)
  - Main branch: consistency between bq_j and f(Ω(u_j)) on strong views.
  - FINESSL difference: use all unlabeled samples by reweighting each with ψ(u_j)=γ·max(p_aux(y|ω(u_j))) derived from the auxiliary classifier; this balances quantity and quality without hard thresholding.
- Auxiliary branch loss (Decoupled Label Smoothing, DLS)
  - Only the auxiliary classifier uses smoothed pseudo-labels e(q_j) with λ∈(0,1) (0.5 by default) to reduce cognitive deviation (overconfidence vs actual difficulty). This branch is detached from the main representation.

4) FINESSL key components (how they are computed)
- Class learning pace σ_t(k)
  - At step t, count confident pseudo-labels for each class: number of u_j where max(q_j) ≥ ζ (ζ=0.7 by default) and argmax=q_j=k.
  - Normalize β_t(k) = σ_t(k) / max_k σ_t(k); define class margin Δ_t(k) = 1 − β_t(k).
  - Scale α_t = (max β_t − min β_t) · α (α=8.0 by default).
  - Use these dynamic margins in supervised and consistency losses so slow classes get stronger emphasis.
- Smoothed pseudo-labels for the auxiliary branch
  - e(q_j)[k] = (1−λ)+λ/C if k=bq_j, otherwise λ/C.
- Reweighting unlabeled samples
  - ψ(u_j) = γ·max(Softmax(faux(ω(u_j)))) (γ=3.0 by default).

5) One training step (mini-batch) example
- Inputs
  - Labeled batch B=32: {(x_i, y_i)}_{i=1..32}
  - Unlabeled batch μB=32: {u_j}_{j=1..32}
- Processing
  1) Compute q_j = p(y|ω(u_j)), define bq_j = argmax(q_j).
  2) Aggregate σ_t(k) over unlabeled (count those with max(q_j) ≥ 0.7), then compute β_t(k), Δ_t(k), α_t.
  3) Compute supervised BMS loss on labeled batch.
  4) Main-branch unsupervised consistency on f(Ω(u_j)) with bq_j, weighted by ψ(u_j).
  5) Auxiliary-branch unsupervised loss with e(q_j) and faux(Ω(u_j)); gradients are detached from the backbone.
  6) Total loss = (main: supervised + weighted consistency) + (auxiliary: supervised + smoothed consistency).
  7) Update θ and θ′ (Θ frozen).
- Outputs
  - Scalar total loss for the step.
  - Updated θ, θ′.
  - Updated σ_t(k).

6) Inference (testing)
- Input: test image x
- Output: argmax of p(y|x) (Top-1); measure Top-1 Accuracy.

7) Concrete scenario A: CIFAR-100, N4 setting (standard in the paper)
- Data
  - C=100 classes.
  - Labeled: 4 per class → Dl size 400.
  - Unlabeled: remaining training images.
- Input shapes
  - Labeled batch: 32 RGB images, labels as 100-dim one-hot vectors.
  - Unlabeled batch: 32 RGB images (pseudo-labels generated on-the-fly).
- Output
  - Training: step losses.
  - Final: Top-1 Accuracy on test set.
- Expected behavior
  - BMS reduces aggregated biases in pseudo-label distribution.
  - DLS + reweighting mitigate cognitive deviation and OOD effects.

8) Concrete scenario B: Semi-Aves (two unlabeled configurations)
- Du = Du_in: unlabeled from the same classes (highly imbalanced fine-grained birds).
- Du = Du_in ∪ Du_out: unlabeled mixed with novel (OOD) classes.
- Inputs/outputs
  - Inputs as images, outputs as Top-1 Accuracy.
- Expected behavior
  - DLS makes reweighting more conservative for potential OOD/noisy samples, improving robustness.

9) Concrete scenario C: OpenSSL on CIFAR-100 (N5/N10)
- Setup
  - Use 50 seen (ID) classes with N5 or N10 labels; the other 50 are OOD and appear only in unlabeled data.
- Inputs/outputs
  - Inputs: as above.
  - Output: accuracy on seen classes.
- Expected behavior
  - DLS sharpens the confidence boundary among correct/incorrect/OOD, improving reweighting.

10) Concrete scenario D: Long-tailed SSL (LTSSL)
- Example
  - CIFAR-100 with long-tailed labeled and unlabeled splits, imbalance ratio ρ (e.g., 10, 20).
- Inputs/outputs
  - Inputs as above, outputs as Top-1 Accuracy.
- Expected behavior
  - BMS aligns learning pace across classes, making pseudo-labels more balanced even under long tails.

11) Training configuration (paper defaults)
- Optimizer: SGD (lr=0.03, momentum=0.9, weight decay=5e-4), cosine schedule.
- Batches: 32 by default, μ=1 (except ImageNet), 30 epochs, 500 steps per epoch.
- FINESSL hyperparameters: ζ=0.7, α=8.0, λ=0.5, γ=3.0.
- Backbone/PEFT: CLIP ViT, VPT-deep (prompt length 50), freeze Θ; update θ, θ′.

12) Input/output summary
- Training input
  - Labeled: RGB images + one-hot labels.
  - Unlabeled: RGB images (pseudo-labels and weights created during training).
- Training output
  - Step loss; updated PEFT and auxiliary parameters.
- Test input/output
  - Input: RGB image.
  - Output: predicted class (Top-1); metric: Top-1 Accuracy.

이 설명은 논문 본문에 정의된 데이터 구성, 손실함수, 하이퍼파라미터, 학습/추론 절차를 그대로 따르며, FINESSL의 두 핵심(균형 마진 소프트맥스, 디커플드 라벨 스무딩+재가중)이 입력과 출력 사이에서 어떻게 작동하는지 구체적으로 보여줍니다.

<br/>
# 요약

메서드: FINESSL은 파운데이션 모델을 PEFT(VPT)로 미세튜닝하며, 클래스별 동적 마진을 부여하는 Balanced Margin Softmax로 집적 편향을 지우고, 보조 분류기에 Decoupled Label Smoothing과 전 샘플 가중화를 결합해 인지 편차를 줄이며 버려지던 비신뢰 가짜라벨도 활용합니다. 
결과: CIFAR-10/100, FOOD-101, Semi-Aves, ImageNet 전반에서 SOTA를 기록했고, 특히 CIFAR-10 N1에서 차상위 대비 약 18%p 향상(96.15%)과 함께 WideResNet 스크래치 대비 약 6배 학습 비용 절감(55.6k s→9.6k s)을 달성했으며, 가짜라벨의 정확도·엔트로피 개선과 분포 균형화도 확인했습니다. 
예시: Figure 2·3에서 균형 잡힌 가짜라벨 분포와 향상된 신뢰도 양상을 보였고, Table 1–3에서 CIFAR-100 N4/N25/N100=80.44/84.51/86.66%, Semi-Aves(IND/OOD)=67.25/61.12%, ImageNet(1%/10%)=74.22/79.21% 등 일관된 우위를, Table 5에서 시간 절감을 확인했습니다.

Method: FINESSL fine-tunes foundation models with PEFT (VPT), erases aggregated bias via Balanced Margin Softmax with adaptive class margins, and mitigates cognitive deviation using a decoupled auxiliary classifier with label smoothing plus sample reweighting to exploit all unlabeled data. 
Results: It sets SOTA across CIFAR-10/100, FOOD-101, Semi-Aves, and ImageNet; e.g., on CIFAR-10 N1 it improves by ~18 percentage points (96.15%) over the best baseline and cuts training cost by ~6× versus training WideResNet from scratch, while increasing pseudo-label accuracy/entropy and balancing their distribution. 
Examples: Figures 2–3 show balanced pseudo-labels and better confidence behavior; Tables 1–3 report consistent gains (e.g., CIFAR-100 N4/N25/N100=80.44/84.51/86.66%, Semi-Aves IND/OOD=67.25/61.12%, ImageNet 1%/10%=74.22/79.21%), and Table 5 shows the training-time reduction.

<br/>
# 기타



[피규어(그림)]
- Fig. 1
  - 결과: 사전학습 ViT를 미세조정하면, 스크래치 학습한 WideResNet보다 SSL에서 크게 우수. PEFT인 VPT가 Full Fine-Tuning(FFT)과 Linear Probing(LP) 대비 모든 설정(N2–N100)에서 큰 이득.
  - 인사이트: 대규모 모델을 SSL에 FFT로 전부 업데이트하면 라벨이 적을 때 오히려 불리. VPT처럼 파라미터 효율적 미세조정이 SSL에 특히 적합.

- Fig. 2
  - 좌: FixMatch는 클래스별 의사라벨 분포가 심하게 불균형(aggregated bias), FlexMatch는 부분 완화, FINESSL은 고르게 균형.
  - 우: VPT는 라벨이 극히 적은 설정(N4)에서도 과신(overconfidence) 경향. N25의 평균 신뢰도가 N100보다 높은 모순. DLS(Decoupled Label Smoothing) 도입 시 과신·인지편차 완화, 학습 난이도와 신뢰도의 정렬 개선.
  - 인사이트: 분포 불균형과 과신이 SSL의 의사라벨 선택을 왜곡. BMS(균형 마진 소프트맥스)와 DLS가 각각 불균형과 과신을 직접 교정.

- Fig. 3
  - (a) FINESSL의 의사라벨 정확도는 학습 내내 타 방법 대비 높음. (b) FINESSL의 의사라벨 분포 엔트로피가 높아 더 다양한 클래스를 커버(불균형 완화).
  - (c)(d) λ(라벨 스무딩 강도)와 α0(마진 스케일 베이스)에 대해 성능 변동이 작음.
  - 인사이트: FINESSL는 더 정확하고 더 고르게 분포된 의사라벨을 생산. 핵심 하이퍼파라미터에 견고.

- Fig. 4
  - (a) OpenSSL(CIFAR-100, N5/N10): DLS 제거 시 평균 1.87%p 성능 하락. DLS가 OOD 존재 상황에 특히 도움.
  - (b–d) ID/오의사라벨/OOD의 신뢰도 분포: FixMatch·DLS제거 모델은 분포 간 중첩이 큼. FINESSL은 세 그룹 간 신뢰도 경계가 뚜렷해 샘플 가중치화가 정확.
  - 인사이트: DLS는 “교정(calibration)”보다 “인지편차 완화”에 초점. OOD 혼재시 신뢰도 기반 가중치가 더 신뢰할 수 있게 됨.

- Fig. 5
  - (a) 샘플 가중 스케일 γ 변화에 성능 민감도 낮음.
  - (b) VPT 프롬프트 길이가 길수록 소폭 향상. 길이 50이 효율·성능 균형점.
  - (c) 30 에폭 이후 성능 수렴. 30×500 step만으로 충분.
  - 인사이트: 하이퍼·훈련 에폭에 견고, 비용 효율적.

- Fig. 6
  - CIFAR-10, FOOD-101에서도 FINESSL이 의사라벨 분포를 균형화.
  - 인사이트: 분포 균형화 효과가 다양한 데이터셋으로 일반화.

- Fig. 7
  - VPT는 라벨 수가 늘어도 평균 신뢰도 추이가 학습 난이도를 제대로 반영하지 않는 인지편차. DLS 적용 시 난이도와 신뢰도 정렬.
  - 인사이트: DLS가 “신뢰도-난이도” 정합을 회복.

- Fig. 8
  - (a) ECE 기준으로 강한 DLS(λ=0.5)는 오히려 교정 지표가 나빠질 수 있음.
  - (b–d) 그러나 강한 DLS가 올바른/오라벨 구분 경계는 더 선명하게 만들어 가중치 학습에 유리.
  - 인사이트: DLS의 목적은 교정보다 “의사라벨 가중치의 판별력” 확보와 인지편차 완화.

[테이블]
- Table 1 (CIFAR-10/100, FOOD-101)
  - 결과: 대부분 설정에서 SOTA. 특히 CIFAR-10 N1에서 타법 대비 ~18%p 향상. CIFAR-10 N4는 여러 기법이 상향포화로 근접.
  - 인사이트: 저라벨 극한 상황에서 강점 극대화. 단순 임계값 기반·확률 재조정 기반법보다 안정적.

- Table 2 (Semi-Aves)
  - 결과: Du_in 67.25, Du_in∪Du_out 61.12로 각각 최고. FixMatch 대비 평균 +1.4%p, DebiasPL 대비 +0.35%p.
  - 인사이트: 클래스 불균형·OOD 혼재의 어려운 상황에서도 견고.

- Table 3 (ImageNet 1%/10% 라벨)
  - 결과: 74.22(1%), 79.21(10%)로 최고. 차이는 약 0.5–0.6%p.
  - 인사이트: 대규모 데이터셋에서도 일관된 개선.

- Table 4 (다양한 PEFT: VPT, LoRA, Adapter, AdaptFormer)
  - 결과: 평균적으로 AdaptFormer가 미세 우세(최대 +0.62%p). VPT도 근소 열세 정도로 경쟁력 높음.
  - 인사이트: 프레임워크가 다양한 PEFT와 호환. 저라벨에서는 VPT가 여전히 강력하고 보편적 선택.

- Table 5 (훈련시간)
  - 결과: FINESSL 9,630s vs WideResNet 스크래치 55,574.5s → 약 1/6. VPT 기반 FixMatch 대비 비용 증가 미미.
  - 인사이트: 성능뿐 아니라 실제 비용/시간에서 압도적 이점.

- Table 6 (어블레이션)
  - BMS 제거 시 성능 큰 폭 하락(특히 CIFAR-100 N4에서 −3.74%p). Lm_s에서 마진 제외 시 Semi-Aves 평균 −1.28%p.
  - α 고정(동적 아님) 시 전반적 하락 → 동적 스케일 필수.
  - DLS 제거 시 전반적 하락, 특히 OOD 포함 Semi-Aves에서 −2.54%p.
  - 보조분류기에 LS 미적용 시 악화(특히 OOD). 보조분류기 detach 해제 시 저라벨에서 악화.
  - 인사이트: 두 축(BMS·DLS)과 보조분류기의 “분리” 모두 핵심.

- Table 7 (ResNet-CLIP + BN Tuning/BitFit/SSF)
  - 결과: 모든 전략에서 FINESSL 최상. DebiasPL 대비 평균 +2.19%p.
  - 인사이트: ViT뿐 아니라 ResNet 백본에서도 일반화.

- Table 8 (기본 설정)
  - 결과: 30 epochs × 500 steps, ViT-CLIP + VPT 길이 50, τ=0.7 등.
  - 인사이트: 소량 에폭·스텝으로도 높은 성능 재현 가능.

- Table 9 (장꼬리 LTSSL)
  - 결과: 평균적으로 SoftMatch 대비 +2.95%p. DebiasPL도 강하지만 대부분 설정에서 FINESSL 우위.
  - 인사이트: 데이터 불균형성에도 견고.

- Table 10 (LP/FFT 변형)
  - 결과: FFT는 매우 저조하지만 LP-FT, FFT†(의미기반 초기화)로 대폭 개선. FINESSL은 모든 전략 조합에서도 FixMatch 대비 평균 +4.59%p.
  - 인사이트: 분류기 초기화가 FFT 안정화에 중요. 그럼에도 FINESSL의 핵심 개선은 일관 유지.

- Table 11 (ImageNet-21k 사전학습)
  - 결과: 절대 성능이 CLIP보다 더 높게 나오며 FINESSL이 DebiasPL 대비 평균 +0.36%p.
  - 인사이트: 사전학습 데이터와의 유사성이 높을수록 성능 상향. 방법의 이점은 여전히 유지.

- Table 12 (FPL/IFPL/GRIP 비교)
  - 결과: FINESSL이 N4/N25 등 저라벨에서 유의미 우위. FPL/IFPL은 클래스당 고정 수의 언라벨 샘플이라 양적 확대 제한·쉬운 샘플에 치우침. GRIP은 aggregated bias에 취약.
  - 인사이트: FINESSL의 분포 균형화·가중 전략이 더 실전적.

[부록(핵심 인사이트)]
- A. 구현: 한 장의 3090으로 30×500 step, 임계값 0.7 사용 시 재현 가능. VPT 길이 50.
- B.2: LP-FT/의미기반 초기화로 FFT 개선 가능하나, FINESSL의 상대 우위 유지.
- B.4: ViT 스크래치 학습(FixMatch, 500에폭) 성능이 VPT 대비 −54.77%p. Semiformer 대비 ImageNet 10% 라벨에서 +3.71%p.
- C.3: DLS는 캘리브레이션(ECE) 개선이 목적이 아니라, 올바름/오라벨/OOD 간 신뢰도 판별 경계를 명확히 하여 “인지편차”를 줄이는 데 초점.
- C.4: VPT가 FFT보다 나은 이유: (1) SSL은 라벨이 적어 대규모 FFT에 부적합, (2) 사전학습/다운스트림 목적의 불일치와 데이터 분포 차이 맥락에서 PEFT가 유리.
- D. 한계: CLIP(이미지 인코더) 중심, 텍스트 인코더/다른 자기지도 사전학습(SimCLR, MoCo 등) 미활용. 향후 확장 여지.





[Figures]
- Fig. 1
  - Results: Fine-tuning a pre-trained ViT surpasses training WideResNet from scratch. VPT consistently beats FFT and LP across SSL label regimes.
  - Insight: Parameter-efficient tuning is better suited to SSL than full-model updates under scarce labels.

- Fig. 2
  - Left: FixMatch suffers from highly imbalanced pseudo-labels (aggregated bias), FlexMatch partly mitigates, FINESSL yields balanced distributions.
  - Right: VPT is overconfident; average confidence does not align with task difficulty. DLS corrects this mismatch.
  - Insight: Address imbalance and overconfidence at the model level (BMS) and the confidence level (DLS).

- Fig. 3
  - (a) FINESSL yields higher pseudo-label accuracy throughout training. (b) Higher entropy indicates more diverse pseudo-labels.
  - (c)(d) Low sensitivity to λ and α0.
  - Insight: More accurate and balanced pseudo-labels with robust hyperparameter behavior.

- Fig. 4
  - (a) In OpenSSL, removing DLS degrades by 1.87%p on average. (b–d) DLS sharpens confidence separation among correct/wrong/OOD.
  - Insight: DLS focuses on mitigating cognitive deviation rather than improving calibration.

- Fig. 5
  - (a) Robust to γ. (b) Longer VPT prompts slightly help; length 50 is a good trade-off. (c) Converges by 30 epochs.
  - Insight: Stable and efficient training.

- Fig. 6
  - Balanced pseudo-label distributions generalize to CIFAR-10 and FOOD-101.
  - Insight: Bias erasure is consistent across datasets.

- Fig. 7
  - DLS aligns confidence with task difficulty; vanilla VPT exhibits cognitive deviation.
  - Insight: Confidence reflects learning difficulty after DLS.

- Fig. 8
  - (a) Strong DLS may worsen ECE. (b–d) Yet, it improves separability between correct vs wrong pseudo-labels, aiding sample reweighting.
  - Insight: DLS targets discriminability and deviation, not calibration per se.

[Tables]
- Table 1 (CIFAR-10/100, FOOD-101)
  - Results: New SOTA in most settings; +~18%p on CIFAR-10 N1. CIFAR-10 N4 is near-saturated across methods.
  - Insight: Especially strong under extreme label scarcity.

- Table 2 (Semi-Aves)
  - Results: Best in both ID-only and ID+OOD unlabeled settings.
  - Insight: Robust to class imbalance and OOD contamination.

- Table 3 (ImageNet 1%/10%)
  - Results: Best by ~0.5–0.6%p.
  - Insight: Scales to large datasets.

- Table 4 (PEFT variants)
  - Results: AdaptFormer slightly edges out VPT; all PEFTs integrate well.
  - Insight: Framework-agnostic to PEFT; VPT remains a strong default.

- Table 5 (Training time)
  - Results: ~6× faster than training WideResNet from scratch; overhead vs VPT-FixMatch is negligible.
  - Insight: Practical efficiency for deployment.

- Table 6 (Ablations)
  - Results: Removing BMS, dynamic α, or supervised margins degrades notably. Removing DLS hurts, especially with OOD. Not detaching the aux head harms under low labels.
  - Insight: Both BMS and DLS, plus the decoupled auxiliary head, are essential.

- Table 7 (ResNet-CLIP with BN Tuning/BitFit/SSF)
  - Results: FINESSL best in all cases; +2.19%p over DebiasPL on average.
  - Insight: Benefits generalize beyond ViT.

- Table 8 (Default configs)
  - Insight: Strong results with 30×500 steps, τ=0.7, VPT length 50.

- Table 9 (Long-tailed SSL)
  - Results: Average +2.95%p over SoftMatch; stronger than DebiasPL in most settings.
  - Insight: Robust under label/data imbalance.

- Table 10 (LP/FFT variants)
  - Results: LP-FT and semantic init (FFT†) boost FFT. FINESSL beats FixMatch across all strategies by +4.59%p on average.
  - Insight: Classifier initialization matters, but FINESSL’s advantages persist.

- Table 11 (ImageNet-21k pretraining)
  - Results: Higher absolute performance; FINESSL +0.36%p over DebiasPL on average.
  - Insight: Gains remain with more task-aligned pretraining.

- Table 12 (FPL/IFPL/GRIP)
  - Results: FINESSL superior, especially in low-label regimes; pseudo-label selection limits of FPL/IFPL and bias amplification of GRIP noted.
  - Insight: Balanced pseudo-labeling and reweighting are more effective in practice.

[Appendix highlights]
- A: Reproducible with 1×3090, 30×500 steps, τ=0.7, VPT length 50.
- B.2: LP-FT/semantic init improve FFT; FINESSL still ahead.
- B.4: ViT-from-scratch underperforms VPT by −54.77%p; surpasses Semiformer by +3.71%p (ImageNet 10% labels).
- C.3: DLS addresses cognitive deviation; better discriminability even if ECE worsens.
- C.4: Why VPT > FFT in SSL: limited labels and task/domain mismatch argue for PEFT.
- D: Limitations: focus on CLIP image encoder; not leveraging text encoder or other self-supervised pretraining yet.

<br/>
# refer format:



BibTeX   
@inproceedings{gan2024finessl,
  title     = {Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning},
  author    = {Gan, Kai and Wei, Tong},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
  series    = {Proceedings of Machine Learning Research},
  volume    = {235},
  year      = {2024},
  address   = {Vienna, Austria},
  publisher = {PMLR},
  url       = {https://arxiv.org/abs/2405.11756},
  eprint    = {2405.11756},
  eprinttype= {arXiv},
  eprintclass= {cs.LG}
}




- Notes-Bibliography(참고문헌/각주용):  
Gan, Kai, and Tong Wei. 2024. “Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning.” In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), vol. 235. Vienna, Austria: PMLR. https://arxiv.org/abs/2405.11756.



- Author-Date(본문 내 저자-연도용 참고문헌):   
Gan, Kai, and Tong Wei. 2024. “Erasing the Bias: Fine-Tuning Foundation Models for Semi-Supervised Learning.” In Proceedings of the 41st International Conference on Machine Learning (ICML 2024), vol. 235. Vienna, Austria: PMLR. https://arxiv.org/abs/2405.11756.
