---
layout: post
title:  "[2021]FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"
date:   2025-08-26 17:37:51 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

메서드: 클래스별 학습 상태를 “고신뢰 예측 수”로 추정해 가변 임계값 Tt(c)=M(βt(c))·τ를 동적으로 조정하는 Curriculum Pseudo Labeling(CPL)을 제안하고 FixMatch에 적용한 FlexMatch를 제시(임계값 워밍업·비선형 매핑 포함, 추가 연산·파라미터 거의 없음).

즉, 커리큘럼 러닝을 하는데.. 점진적으로 학습할 새롭게 레이블링된 애들을 잘 찾음..  

추가적으로 잘 학습된 클래스는 임계값을 높여 더 “확실한” 샘플만 사용, 덜 학습된 클래스는 임계값을 낮춰 더 많은 pseudo-labeled 샘플을 통과시킴, 이렇게 해서 결국 클래스 불균형 문제를 완화하고, pseudo-label 활용 효율을 높임   


짧은 요약(Abstract) :

- 문제의식: FixMatch 등 최신 반지도학습(SSL)은 모든 클래스에 동일한 고정 임계값으로 고신뢰 예측만 학습에 사용한다. 이는 학습 초기에 많은 비라벨 데이터를 버리고, 클래스별 난이도 차이를 반영하지 못한다.
- 제안: Curriculum Pseudo Labeling(CPL). 모델의 현재 학습 상태에 따라 클래스별 임계값을 매 스텝 유연하게 조정해, 정보성이 높은 비라벨 샘플과 그 가pseudo 라벨을 선택적으로 통과시킨다. 추가 파라미터나 추가 연산(추가 forward/backward)이 없다.
- 적용: CPL을 FixMatch에 결합한 FlexMatch 제안.
- 성능: 다양한 SSL 벤치마크에서 SOTA 달성. 라벨이 극히 적거나 과제가 어려울수록 이득이 크다. 예: 클래스당 4개 라벨만 있을 때 CIFAR-100과 STL-10에서 FixMatch 대비 오류율을 각각 13.96%, 18.96% 상대 감소.
- 효율: 수렴 속도가 크게 향상되어 FixMatch가 필요로 하는 학습 시간의 1/5만으로도 더 나은 성능에 도달 가능.
- 범용성: CPL은 다른 SSL 알고리즘에도 쉽게 적용되어 성능을 유의미하게 끌어올린다.
- 코드: https://github.com/TorchSSL/TorchSSL 공개.


- Motivation: Modern SSL methods like FixMatch use a single fixed confidence threshold for all classes to select unlabeled samples, ignoring per-class learning difficulty and discarding many samples early in training.
- Proposal: Curriculum Pseudo Labeling (CPL), which adapts class-wise thresholds over time based on the model’s learning status for each class, letting more informative unlabeled samples and their pseudo labels pass. It introduces no extra parameters and no additional forward/backward computation.
- Integration: Applying CPL to FixMatch yields FlexMatch.
- Results: FlexMatch achieves state-of-the-art performance on multiple SSL benchmarks, with especially strong gains when labels are extremely scarce or tasks are challenging. For example, with 4 labels per class, it reduces error over FixMatch by 13.96% on CIFAR-100 and 18.96% on STL-10.
- Efficiency: It significantly accelerates convergence, reaching even better accuracy using only about 1/5 of FixMatch’s training time.
- Generality: CPL can be plugged into other SSL algorithms and markedly improves their performance.
- Code: https://github.com/TorchSSL/TorchSSL


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



개요
- 문제 설정: 반지도학습(SSL)에서 소량의 라벨과 대량의 비라벨 데이터를 함께 학습.
- 기존 강자: FixMatch는 약한 증강으로 만든 예측을 강한 증강 샘플의 학습 목표로 삼고, 일정한 신뢰도 임계값 τ를 넘는 비라벨 샘플만 사용.
- 한계: 모든 클래스에 동일한 고정 임계값을 적용. 클래스별 난이도와 학습 진행도 차이를 반영하지 못해 초기에는 사용 가능한 비라벨이 거의 없고, 전체 데이터 활용과 수렴 속도가 저하됨.
- 핵심 제안: Curriculum Pseudo Labeling(CPL). 시점 t마다 클래스별 학습 상태를 추정하여 클래스별 유연 임계값 T_t(c)을 동적으로 조절. 이를 FixMatch에 접목한 것이 FlexMatch.

모델/아키텍처
- 분류기: 표준 CNN 백본을 그대로 사용. ImageNet에는 ResNet-50, CIFAR/SVHN/STL에는 Wide ResNet 변형 사용.
- 별도 네트워크나 추가 모듈 없음. CPL은 손실 계산 시 임계값만 동적으로 바꾸는 방식으로 동작.

데이터와 증강
- 라벨 데이터: 표준 라벨 배치 사용.
- 비라벨 데이터: 한 샘플당 두 가지 뷰 생성
  - 약한 증강 ω(u): pseudo label 생성용.
  - 강한 증강 Ω(u): 학습 표적(타깃)과의 일치(Consistency) 강제용.
- 강한 증강은 RandAugment 사용. 기타 기본 전처리는 FixMatch와 동일.

핵심 기법: Curriculum Pseudo Labeling(CPL)
1) 클래스별 학습 상태 추정
- 직관: 높은 임계값 τ에서 그 임계값을 통과하는 비라벨 샘플 수가 많을수록 해당 클래스가 더 잘 학습되고 있다고 볼 수 있음.
- 정의: 시점 t에서 각 클래스 c의 “추정 학습효과”를
  σ_t(c) = Σ_n 1(max p_m,t(y|u_n) > τ) · 1(argmax p_m,t(y|u_n) = c)
  로 집계. 즉, 약한 증강에서 최대 확률이 τ를 넘고, 예측 클래스가 c인 비라벨 샘플 수.
- 균형 잡힌 비라벨 분포일 때 σ_t(c)가 클수록 해당 클래스가 상대적으로 잘 학습된 상태로 간주.

2) 정규화와 유연 임계값
- 정규화: β_t(c) = σ_t(c) / max_c σ_t(c)
- 클래스별 유연 임계값: T_t(c) = M(β_t(c)) · τ
  - M은 [0,1] → [0,1]의 단조 증가 매핑. 선형 M(x)=x도 가능하며, 실험적으로 볼록형(예: M(x)=x/(2−x))이 더 우수.
  - 의미: 가장 잘 학습된 클래스는 임계값이 τ에 가깝고, 어려운 클래스는 임계값이 낮아 더 많은 샘플을 통과시켜 학습을 촉진.

3) Warm-up(임계값 워밍업)
- 초기에는 모델이 특정 클래스에 치우친 예측을 할 수 있어 σ_t(c) 기반 추정이 불안정.
- 완화책: β_t(c) 계산 시 분모를 max(max_c σ_t(c), N − Σ_c σ_t(c))로 변경.
  - N − Σ_c σ_t(c)는 아직 사용되지 않은 비라벨 샘플 수.
  - 직관적으로 “미사용”을 하나의 추가 클래스처럼 취급하여, 초기에 모든 클래스의 β가 0에서 서서히 상승하도록 하여 왜곡 완화.

4) 손실 함수와 포함 규칙
- 라벨 손실 L_s: 약한 증강 라벨 배치에 대한 표준 cross-entropy.
- 비라벨 손실 L_u,t:
  - 약한 증강 확률 q_b = p_m(y|ω(u_b))를 계산.
  - 예측 클래스 c* = argmax q_b에 대해, max(q_b) > T_t(c*)이면 해당 샘플을 사용.
  - 사용 시 pseudo label(일반적으로 hard one-hot)을 목표로 강한 증강 확률 p_m(y|Ω(u_b))에 cross-entropy 부과.
- 전체 손실: L_t = L_s + λ · L_u,t

5) 알고리즘 흐름(요약)
- 각 비라벨 샘플의 “최근에 τ를 넘겨 사용된 예측 클래스”를 캐시(초기값 -1).
- 각 반복에서:
  - 캐시로 σ(c)를 누적 집계 → β(c) 계산(warm-up 규칙 포함) → T(c) 산출.
  - 현재 배치의 약한 증강 예측이 τ를 넘으면 캐시를 해당 클래스 c로 갱신.
  - 유연 임계값 T(c*) 기준을 통과하는 샘플만 비라벨 손실에 반영.
- 추가 순전파/역전파나 추가 파라미터 없이 구현 가능. 시간/메모리 오버헤드는 무시해도 될 수준.

학습 설정(주요 하이퍼파라미터; FixMatch와 동일하게 설정)
- 최적화: SGD(m=0.9), 코사인 러닝레이트 디케이, 초기 lr=0.03.
- EMA: 모멘텀 0.999.
- 배치: 라벨 배치 64, μ=7(라벨:비라벨 비율).
- 임계값 상한 τ: 기본 0.95(FixMatch와 동일). UDA는 0.8.
- 강한 증강: RandAugment.
- 백본: CIFAR/STL/SVHN에 WRN, ImageNet에 ResNet-50.

계산 복잡도 및 구현 비용
- CPL은 임계값 계산을 위한 카운팅/정규화만 추가. 별도의 추론/학습 스텝 없음.
- 실험적으로 반복당 실행 시간 증가가 거의 없으며, 오히려 수렴이 빨라 총 학습 시간을 단축.

성능 및 효과
- 저라벨 상황에서 큰 개선: CIFAR-100, 400 라벨(클래스당 4개)에서 FixMatch 46.42% → FlexMatch 39.94% 오류율.
- 어려운 과제에서 큰 개선: STL-10, 40 라벨에서 35.97% → 29.15%.
- 수렴 가속: 동일 장비/셋업에서 FixMatch 최종 성능을 FlexMatch가 약 1/5 학습 시간에 달성하거나 능가.
- 범용성: Pseudo-Labeling, UDA에도 CPL을 삽입하면 성능 향상(Flex-UDA 등).

설계 선택의 고찰
- τ(임계값 상한): 너무 낮거나 높으면 성능 저하. 대체로 0.95 부근이 최적.
- 매핑 M: 볼록형(예: x/(2−x))이 가장 우수, 선형은 중간, 오목형은 열세.
- 워밍업: 초기 예측 쏠림과 확인편향 위험을 줄여 안정적 학습에 기여.

제약 및 관찰
- 클래스 불균형 데이터에서는 특정 클래스의 β가 장기간 낮게 유지될 수 있어, 지나치게 낮은 임계값이 노이즈 라벨을 통과시키는 부작용 가능(SVHN에서 관찰). 과제가 매우 쉬운 경우엔 FixMatch의 높은 고정 임계값이 오히려 장점이 될 수 있음.

요약
- FlexMatch는 “클래스별·시점별 동적 임계값”이라는 단순하고 비용 없는 커리큘럼을 통해, 비라벨 데이터 활용률을 높이고 어려운 클래스를 먼저 보완하여 수렴을 가속하고 최종 성능을 개선한다. 별도 아키텍처 변경 없이 기존 SSL 프레임워크에 쉽게 삽입 가능하다.




Overview
- Task: Semi-supervised learning (SSL) with few labeled and many unlabeled samples.
- Strong baseline: FixMatch uses weak-to-strong consistency with a fixed confidence threshold τ, only training on unlabeled samples above τ.
- Limitation: A single fixed τ ignores class-wise difficulty and learning progress. Early in training, very few samples pass τ, wasting data and slowing convergence.
- Key idea: Curriculum Pseudo Labeling (CPL). Dynamically adjust a class-specific threshold T_t(c) at each step based on the estimated learning status of that class. FlexMatch = FixMatch + CPL.

Model/Architecture
- Standard CNN backbones; no extra heads or modules. ResNet-50 for ImageNet, Wide ResNet variants for CIFAR/STL/SVHN.
- CPL operates purely on the loss side by changing thresholds; no architectural change.

Data and Augmentation
- Labeled data: standard supervised mini-batches.
- Unlabeled data: two views per sample
  - Weak augmentation ω(u): to generate pseudo labels.
  - Strong augmentation Ω(u): target for consistency training.
- Strong augmentations use RandAugment; other preprocessing follows FixMatch.

Core Technique: Curriculum Pseudo Labeling (CPL)
1) Estimating class-wise learning status
- Intuition: Under a high τ, the number of unlabeled samples that pass τ and are predicted as class c reflects how well class c is learned.
- Define, at step t:
  σ_t(c) = sum over unlabeled n of 1(max p_m,t(y|u_n) > τ) · 1(argmax p_m,t(y|u_n) = c)
- For balanced unlabeled data, larger σ_t(c) indicates better learning of class c.

2) Normalization and flexible thresholds
- Normalize: β_t(c) = σ_t(c) / max_c σ_t(c)
- Class-wise flexible threshold: T_t(c) = M(β_t(c)) · τ
  - M is a monotone map from [0,1] to [0,1]. Linear M(x)=x is viable; convex functions (e.g., M(x)=x/(2−x)) work best empirically.
  - Effect: Well-learned classes keep thresholds near τ; hard classes get lower thresholds to admit more samples and accelerate learning.

3) Threshold warm-up
- Early predictions can collapse to one class, making σ-based estimates unreliable.
- Remedy: Compute β_t(c) with denominator max(max_c σ_t(c), N − sum_c σ_t(c)),
  where N − sum σ counts unused unlabeled samples. This behaves like adding an “unused” class so β ramps up smoothly from zero, stabilizing early estimates.

4) Loss and sample inclusion
- Labeled loss L_s: standard cross-entropy on weakly augmented labeled batch.
- Unlabeled loss L_u,t:
  - Compute q_b = p_m(y|ω(u_b)).
  - With c* = argmax q_b, include sample if max(q_b) > T_t(c*).
  - When included, use a pseudo label (typically hard one-hot) to supervise p_m(y|Ω(u_b)) via cross-entropy.
- Total loss: L_t = L_s + λ · L_u,t

5) Algorithm sketch
- Cache, per unlabeled sample, its most recent “used” predicted class (init -1).
- Each iteration:
  - Aggregate σ(c) from the cache → compute β(c) with warm-up → derive T(c).
  - For current batch, if weak-view confidence > τ, update cache to that class.
  - Apply class-wise threshold T(c*) to decide inclusion in L_u,t.
- No extra forward/backward or trainable parameters; overhead is negligible.

Training Setup (key hyperparameters; same as FixMatch unless noted)
- Optimizer: SGD (momentum 0.9), cosine LR decay, initial lr=0.03.
- EMA with momentum 0.999.
- Batch: labeled 64; unlabeled ratio μ=7.
- Threshold upper bound τ: typically 0.95 (0.8 for UDA).
- Strong aug: RandAugment.
- Backbones: WRN for CIFAR/STL/SVHN, ResNet-50 for ImageNet.

Complexity and Implementation Cost
- CPL adds simple counting/normalization for thresholds. No extra inference/training steps.
- Per-iteration time virtually unchanged; faster convergence often reduces total training time.

Effectiveness
- Low-label regimes: Large gains, e.g., CIFAR-100 with 400 labels improves from 46.42% to 39.94% error.
- Hard tasks: Larger gains on STL-10 with 40 labels, 35.97% to 29.15%.
- Convergence: Reaches FixMatch’s final accuracy in roughly 1/5 of the training time in some settings.
- Generality: Plugging CPL into Pseudo-Labeling and UDA also improves them (e.g., Flex-UDA).

Design Choices
- τ: Too low or too high degrades performance; around 0.95 works well in practice.
- Mapping M: Convex mapping (e.g., x/(2−x)) > linear > concave in experiments.
- Warm-up: Helps stabilize early training and reduces confirmation bias.

Limitations and Observations
- Class imbalance can keep β low for minority classes, leading to overly low thresholds and noisier pseudo labels (observed on SVHN). On very easy tasks, FixMatch’s high fixed τ can be competitive.

Summary
- FlexMatch uses a simple, cost-free curriculum—class- and time-dependent thresholds—to increase unlabeled data utilization, focus on hard classes sooner, accelerate convergence, and improve final accuracy. It is easy to integrate into existing SSL methods without architectural changes.


<br/>
# Results






1) 비교 대상(경쟁 모델)
- 기본선: Pseudo-Labeling(PL), UDA, FixMatch, Fully-Supervised(모든 데이터에 라벨 부여) 
- 제안 기법: FlexMatch = FixMatch + CPL
- 추가 비교: Flex-PL, Flex-UDA (각각 PL, UDA에 CPL을 적용한 변형)

2) 데이터셋과 라벨 분할(테스트 데이터)
- CIFAR-10: 40, 250, 4000 라벨
- CIFAR-100: 400, 2500, 10000 라벨
- STL-10: 40, 250, 1000 라벨(비라벨 데이터 분포가 더 넓어 난이도가 높음; unlabeled에 라벨이 없어서 fully-supervised 결과는 없음)
- SVHN(+extra set 포함): 40, 1000 라벨
- ImageNet-1K: 클래스당 100개(총 100K) 라벨(전체의 <8%), 220k iteration 시점 비교

3) 평가 지표/프로토콜
- 주요 지표: 분류 에러율(%) 
- ImageNet: Top-1/Top-5 에러율(%)
- 체크포인트 선택: 전체 학습 과정 중 “최고 성능(best)” 에러율을 주로 보고(수렴 속도가 기법별로 달라 median-of-last-20 방식은 불리할 수 있다고 지적). 각 실험은 서로 다른 시드로 3회 반복하고 평균±표준편차로 보고.
- 구현/세팅 공정성: FixMatch와 동일한 하이퍼파라미터 사용(optimizer: SGD+momentum, cosine LR decay, EMA, μ, τ 등), 강한 증강은 RandAugment, 표준 WRN/ResNet 백본 사용.

4) 핵심 결과 요약(표 1, 표 2 및 본문 서술 기반)

- CIFAR-10
  - 40 라벨: FixMatch 7.47±0.28 → FlexMatch 4.97±0.06로 큰 개선. UDA도 CPL 적용 시 10.62→5.44로 개선, PL은 미소 개선.
  - 250 라벨: FixMatch 4.86±0.05 ↔ FlexMatch 4.98±0.09(근소 열세). UDA/PL은 CPL 적용 시 소폭 개선.
  - 4000 라벨: FixMatch 4.21±0.08 ↔ FlexMatch 4.19±0.01(근소 우세). Fully-supervised는 4.62±0.05로, FlexMatch가 이를 소폭 하회(=더 낮은 에러)하여 SSL의 강점을 보임.

- CIFAR-100
  - 400 라벨(클래스당 4개): FixMatch 46.42±0.82 → FlexMatch 39.94±1.62로 대폭 개선.
  - 2500 라벨: 28.03±0.16 → 26.49±0.20로 개선.
  - 10000 라벨: 22.20±0.12 → 21.90±0.15로 개선.
  - UDA/PL도 CPL 적용 시 전반적으로 소폭 개선. Fully-supervised는 19.30±0.09.

- STL-10(난이도 높고 분포 차이 존재)
  - 40 라벨: FixMatch 35.97±4.14 → FlexMatch 29.15±4.16로 크게 개선. UDA도 Flex-UDA 37.42→29.53로 큰 개선.
  - 250 라벨: 9.81±1.04 → 8.23±0.39 개선.
  - 1000 라벨: 6.25±0.33 → 5.77±0.18 개선.
  - unlabeled 라벨 정보가 없어 fully-supervised 비교값은 없음.

- SVHN(+extra)
  - 40 라벨: Flex-UDA가 3.42±1.51로 최저(원 UDA 5.12±4.27 대비 개선). FixMatch(3.81±1.18) 대비 FlexMatch는 8.19±3.20으로 열세.
  - 1000 라벨: UDA 1.89±0.01가 최저, Flex-UDA 2.02±0.05 근접. FixMatch 1.96±0.03, FlexMatch 6.72±0.30로 FlexMatch는 열세.
  - 저자 해석: SVHN은 상대적으로 쉽고 클래스 불균형이 있어, CPL의 클래스별 가변 임계치가 소수 클래스에서 지나치게 낮아져 잡음 유입(확신 낮은 가짜 라벨 채택)이 발생할 수 있음. 이 경우 FixMatch의 고정 높은 임계치(0.95)가 오히려 유리.

- ImageNet-1K(표 2, 220k iters 시점)
  - FixMatch: Top-1 43.66, Top-5 21.80
  - FlexMatch: Top-1 42.02, Top-5 19.49
  - 동일 설정에서 FlexMatch가 더 낮은 에러율. 다만 220k에서 완전 수렴 전이며, 추가 튜닝 없이 보고.

5) 수렴 속도 및 연산 비용
- 수렴/안정성
  - CIFAR-100(400 라벨)에서 FlexMatch는 손실이 더 빠르고 매끄럽게 하강, 약 50k iteration 만에 FixMatch 최종 성능을 상회. 이후 과적합 가능성으로 손실/정확도 소폭 악화 경향은 양쪽 모두 관찰.
  - CIFAR-10 초기 구간(200k iters) 클래스별 정확도: FixMatch는 56.4% 수준으로 반 수준 미달 클래스 다수 존재, FlexMatch는 94.3%로 매우 빠르게 전반적 학습 달성. CPL이 어려운 클래스를 더 일찍/적절히 학습하도록 유도함을 시사.
- 시간/자원
  - 단일 iteration 평균 시간은 CPL 도입 전후 큰 차이 없음(Figure 2). 즉, 성능과 수렴속도 향상에도 연산 오버헤드는 사실상 없음.

6) 추가 분석(요약; 세부 수치는 그림 참조)
- 임계치 상한 τ: 약 0.95가 최적, 과소/과대 시 성능 저하.
- 매핑함수 M(x): 볼록(convex, 예: x/(2−x))이 최적, 선형은 중간, 오목(concave)은 열세.
- 임계치 워밍업: 초기 편향(한 클래스로 쏠림) 완화에 효과, 특히 CIFAR-100에서 안정적 성능 개선.

7) 총평
- FlexMatch는 라벨이 극소하거나(예: CIFAR-100 400라벨) 과제가 어려운 경우(STL-10) FixMatch 대비 큰 개선을 보이며, UDA/PL에도 CPL을 적용하면 전반적으로 성능 향상이 관측됨.
- SVHN과 같이 쉽고 클래스 불균형이 큰 데이터에서는 고정 높은 임계치가 더 유리할 수 있어 FlexMatch가 항상 우월한 것은 아님.
- 연산 오버헤드 없이 수렴을 크게 가속하고, 일부 설정에서는 fully-supervised를 능가하는 성능을 달성.





1) Competitors
- Baselines: Pseudo-Labeling (PL), UDA, FixMatch, Fully-Supervised (all data labeled)
- Proposed: FlexMatch = FixMatch + CPL
- Also reported: Flex-PL and Flex-UDA (CPL added to PL/UDA)

2) Datasets and label splits
- CIFAR-10: 40, 250, 4000 labels
- CIFAR-100: 400, 2500, 10000 labels
- STL-10: 40, 250, 1000 labels (unlabeled set is broader; harder/more realistic; no fully-supervised number)
- SVHN (+extra set): 40, 1000 labels
- ImageNet-1K: 100 labels/class (100K total, <8% of full labels), evaluated at 220k iterations

3) Metrics and evaluation protocol
- Main metric: classification error rate (%)
- ImageNet: Top-1 and Top-5 error (%)
- Checkpoint selection: report the best error across training (median of last-20 also measured but can be unfair when convergence speeds differ). Each result averaged over 3 runs with different seeds (mean ± std).
- Fairness: same hyperparameters as FixMatch (optimizer, schedule, EMA, μ, τ), RandAugment for strong aug, standard WRN/ResNet backbones.

4) Key results (from Table 1, Table 2, and text)

- CIFAR-10
  - 40 labels: FixMatch 7.47±0.28 → FlexMatch 4.97±0.06 (large gain). UDA improves with CPL (10.62→5.44); PL slightly improved.
  - 250 labels: FixMatch 4.86±0.05 vs FlexMatch 4.98±0.09 (slightly worse). UDA/PL with CPL: small gains.
  - 4000 labels: 4.21±0.08 vs 4.19±0.01 (slightly better). Fully-supervised 4.62±0.05; FlexMatch surpasses it slightly, showing SSL advantage.

- CIFAR-100
  - 400 labels: 46.42±0.82 → 39.94±1.62 (strong improvement).
  - 2500 labels: 28.03±0.16 → 26.49±0.20 (improved).
  - 10000 labels: 22.20±0.12 → 21.90±0.15 (improved).
  - UDA/PL also benefit modestly from CPL. Fully-supervised is 19.30±0.09.

- STL-10 (harder, distribution shift in unlabeled set)
  - 40 labels: 35.97±4.14 → 29.15±4.16 (large gain). UDA also greatly improved with CPL (37.42→29.53).
  - 250 labels: 9.81±1.04 → 8.23±0.39 (improved).
  - 1000 labels: 6.25±0.33 → 5.77±0.18 (improved).
  - Fully-supervised N/A.

- SVHN (+extra)
  - 40 labels: Flex-UDA is best at 3.42±1.51 (vs UDA 5.12±4.27). FlexMatch underperforms FixMatch (8.19±3.20 vs 3.81±1.18).
  - 1000 labels: UDA 1.89±0.01 is best; Flex-UDA 2.02±0.05 close; FixMatch 1.96±0.03; FlexMatch 6.72±0.30 underperforms.
  - Authors’ explanation: SVHN is relatively easy but class-imbalanced; CPL’s class-wise flexible thresholds can remain low for minority classes, admitting noisy pseudo-labels. A fixed high threshold (0.95) in FixMatch can be advantageous here.

- ImageNet-1K (Table 2; 220k iterations)
  - FixMatch: Top-1 43.66, Top-5 21.80
  - FlexMatch: Top-1 42.02, Top-5 19.49
  - FlexMatch is better under identical settings; note neither method is fully converged at 220k and no extra tuning was done.

5) Convergence speed and compute
- Convergence
  - On CIFAR-100 (400 labels), FlexMatch’s loss drops faster and smoother, surpassing FixMatch’s final result around 50k iterations. Later both show mild overfitting tendencies.
  - Early-stage class-wise accuracy on CIFAR-10 (200k iters): FixMatch ~56.4% overall with many under-learned classes; FlexMatch ~94.3% overall. CPL encourages learning of harder classes earlier.
- Compute overhead
  - Average per-iteration runtime is essentially unchanged with CPL (Figure 2). Performance and convergence gains come with negligible overhead.

6) Ablations (high-level takeaways; see figures for details)
- Threshold upper bound τ: ~0.95 works best; moving away degrades performance.
- Mapping function M(x): convex (e.g., x/(2−x)) > linear > concave.
- Threshold warm-up: stabilizes early training (reduces biased early predictions), particularly helpful on CIFAR-100.

7) Overall
- FlexMatch delivers strong gains over FixMatch when labels are extremely scarce (e.g., CIFAR-100 400 labels) and on challenging tasks (STL-10). Adding CPL also improves UDA/PL in most cases.
- On easy yet imbalanced datasets like SVHN, a fixed high threshold can be preferable; FlexMatch is not universally superior.
- FlexMatch accelerates convergence dramatically with negligible runtime overhead, and can even surpass fully-supervised baselines in some regimes.


<br/>
# 예제



1) 태스크 정의
- 문제 유형: 이미지 분류, 반지도학습(SSL)
- 입력(학습 시):
  - 소량의 라벨된 데이터 X = {(x_m, y_m)}: 이미지 x_m, 정답 레이블 y_m
  - 대량의 라벨 없는 데이터 U = {u_n}: 이미지 u_n
- 출력(학습 후/추론 시):
  - 주어진 입력 이미지에 대한 클래스 확률 p(y|x), 최종 예측 클래스 ŷ = argmax p
- 목적:
  - 지도 손실 Ls(라벨된 소량 데이터에 대한 cross-entropy) + 비지도 손실 Lu(라벨 없는 데이터에 대한 pseudo-label consistency, 강증강에 대해 CE)을 함께 최소화
  - CPL/FlexMatch는 클래스별 학습 상태에 따라 클래스별 임계값을 동적으로 조정해, “어떤 클래스의 어떤 샘플을 언제 학습에 쓸지”를 커리큘럼 방식으로 결정

2) 데이터셋별 구체 예시
- CIFAR-10
  - 클래스: 10개(airplane, automobile, bird, …, truck)
  - 이미지: 32×32 RGB
  - 라벨 분포(예시 세팅): 40 라벨(클래스당 4개)만 지도, 나머지(약 49,960장)는 unlabeled로 사용
  - 테스트셋: 10,000장(모두 GT 라벨 보유)
- CIFAR-100
  - 클래스: 100개
  - 이미지: 32×32 RGB
  - 라벨 분포(예시 세팅): 400 라벨(클래스당 4개), 나머지 전부 unlabeled
  - 테스트셋: 10,000장
- STL-10
  - 클래스: 10개
  - 이미지: 96×96 RGB
  - 특이점: 매우 큰 unlabeled set이 레이블 분포보다 더 넓고 약간 다른 분포 포함(보다 현실적/도전적)
  - 라벨 분포(예시 세팅): 40, 2,500, 10,000 등 실험별로 상이
- SVHN
  - 클래스: 10개(숫자 0–9)
  - 이미지: 32×32 RGB
  - 특이점: digit 인식이라 상대적으로 쉽지만 클래스 불균형 존재 가능, extra set(531,131장) 포함
  - 라벨 분포(예시 세팅): 40 또는 1,000 라벨
- ImageNet-1K
  - 클래스: 1,000개
  - 이미지: 일반적으로 224×224 RGB로 학습
  - 라벨 분포(예시 세팅): 100K 라벨(클래스당 100개), 나머지는 unlabeled로 간주
  - 테스트/검증: 공식 validation set에서 Top-1/Top-5 평가

3) 입력 포맷과 전처리/증강
- 약한 증강 ω(weak aug): 랜덤 크롭+패딩, 수평 플립 등
- 강한 증강 Ω(strong aug): RandAugment(논문 설정), Cutout 등을 조합
- 정규화: 데이터셋 별 평균/표준편차로 정규화
- 배치 구성:
  - 라벨 배치 크기 B = 64
  - 비라벨/라벨 비율 μ = 7 → unlabeled 배치 크기 μB = 448
- 모델/학습:
  - 옵티마이저: SGD(momentum 0.9)
  - 초기 LR 0.03, cosine decay
  - EMA 0.999
  - τ(기준 임계값): FixMatch/FlexMatch 0.95 (UDA 0.8)
  - 손실 가중치 λ = 1 (기본)

4) CPL/FlexMatch의 핵심(클래스별 가변 임계값)
- 표기:
  - pm,t(y|x): 시점 t에서의 모델 확률
  - q_b = pm(y|ω(u_b)): unlabeled 샘플의 약한 증강 확률
  - c* = argmax q_b: 약한 증강에서의 예측 클래스
- CPL의 학습 상태 추정:
  - 고정 임계값 τ를 넘은 unlabeled 예측만 카운트하여, 클래스별 “학습 효과” σ_t(c)를 누적/갱신
    - σ_t(c) = Σ_n 1(max pm,t(y|u_n) > τ) · 1(argmax pm,t(y|u_n) = c)
  - 정규화 β_t(c) = σ_t(c) / max_c σ_t(c)  (워밍업 시 분모를 max{max_c σ_t(c), N − Σ_c σ_t(c)}로 바꿈)
  - 클래스별 가변 임계값 T_t(c) = M(β_t(c)) · τ  (기본은 선형 M(x)=x 또는 논문 실험의 볼록 매핑 사용)
- 마스킹된 비지도 손실:
  - Lu,t = (1/(μB)) Σ_b 1(max q_b > T_t(c*)) · H(one-hot(c*), pm(y|Ω(u_b)))

5) 한 번의 학습 스텝 예시(CIFAR-10, 40라벨)
- 가정:
  - τ = 0.95, B = 64, μ = 7 → unlabeled 448장/스텝
  - 지난 스텝까지 “τ를 넘긴” 예측으로부터 σ_t(c) 카운트가 다음과 같음:
    - σ_t = [30, 80, 10, 15, 60, 45, 20, 100, 25, 15] (총 10클래스, 최대값 100)
  - β_t(c) = σ_t(c)/100 → [0.30, 0.80, 0.10, 0.15, 0.60, 0.45, 0.20, 1.00, 0.25, 0.15]
  - 선형 매핑 M(x)=x라고 가정 → T_t(c) = β_t(c)·0.95
    - 예: class 8(인덱스 7)의 T = 0.95, class 3의 T = 0.095 등
- unlabeled 배치 내 개별 샘플 판단 3건 예:
  1) 샘플 u1의 약증강 확률 q1의 최대값이 class 8에서 0.93
     - 비교: 0.93 > T_t(8)=0.95 → False → 이 샘플은 이번 스텝 비지도 손실에서 제외(마스크)
  2) 샘플 u2의 최대값이 class 5에서 0.92
     - class 5의 T_t(5)=0.60·0.95=0.57
     - 0.92 > 0.57 → True → pseudo-label ŷ=class 5, 강증강 Ω(u2)에 대해 CE 계산
  3) 샘플 u3의 최대값이 class 3에서 0.60
     - class 3의 T_t(3)=0.10·0.95=0.095
     - 0.60 > 0.095 → True → ŷ=class 3로 학습에 사용(난이도 높은 클래스의 더 많은 샘플을 조기에 수용)
- 라벨 배치에 대해서는 표준 CE로 Ls 계산
- 최종 손실 Lt = Ls + λ·Lu,t로 역전파/update
- σ 카운트 갱신:
  - 별도로, “약증강 예측의 최대값이 고정 τ=0.95를 초과한” unlabeled 샘플만 ˆu_n을 업데이트하여 σ_t+1(c)에 반영
  - 이 카운트는 다음 스텝의 β와 T를 갱신하는 데 쓰임
- 워밍업 예시:
  - 학습 초반에는 τ를 넘긴 샘플이 적어 Σ_c σ_t(c) << N
  - 이때 정규화 분모를 max{max_c σ_t(c), N − Σ_c σ_t(c)}로 두어 β_t(c)가 급격히 치우치지 않도록 완만하게 증가

6) 테스트(추론) 단계의 인풋/아웃풋
- 입력: 테스트 이미지(예: CIFAR-10 10,000장), 일반적으로 약한 증강 또는 센터 크롭만 적용
- 출력:
  - 각 이미지의 클래스 확률벡터와 argmax가 최종 예측
  - 평가 지표: CIFAR/SVHN/STL-10은 Top-1 error/accuracy, ImageNet은 Top-1/Top-5 error/accuracy

7) 또 다른 구체 예시(클래스가 많은 경우: CIFAR-100, 400 라벨)
- 상황: 클래스가 100개라 일부 클래스는 학습이 더디다
- 가정: σ_t의 최대값이 250이고, 어떤 어려운 클래스 c의 σ_t(c)=5라면, β_t(c)=5/250=0.02
- 선형 매핑 시 T_t(c)=0.02·0.95≈0.019 → 해당 클래스는 낮은 임계값을 통해 더 많은 unlabeled 샘플을 조기에 수용
- 결과적으로 어려운 클래스의 리콜을 조기에 올리고, 전반적 수렴을 가속

8) 하이퍼파라미터/모델 예
- 모델: CIFAR류는 Wide ResNet(논문은 WRN 및 변형 사용), ImageNet은 ResNet-50
- μ=7, τ=0.95(FlexMatch/FixMatch), λ=1, LR 0.03, SGD+momentum 0.9, cosine decay, EMA 0.999
- 강증강: RandAugment
- 평가: 베스트 체크포인트 성능(또는 마지막 20 체크포인트의 중앙값도 보조 지표로 사용)

9) 산출물
- 학습 완료 후: 학습된 모델 파라미터(분류기)
- 테스트셋에 대해: 각 이미지의 예측 클래스를 출력하며, 에러율/정확도/Top-5 등 보고

10) 유의사항/엣지 케이스
- 불균형 unlabeled(예: SVHN)에서는 일부 클래스의 β_t(c)가 계속 낮게 유지될 수 있어, 해당 클래스 임계값이 과도하게 낮아지는 부작용 가능 → 노이즈 유입 증가 → 손실 변동성 증가
- STL-10 같이 unlabeled 분포가 넓은 경우 CPL의 장점이 크게 나타남(난이도 높은 클래스에 더 많은 기회 제공)
- 매핑 함수 M는 선형 외에도 볼록형을 사용할 수 있음(논문에서는 다양한 M에 대한 비교 수행)




1) Task definition
- Problem: Image classification under semi-supervised learning
- Inputs (during training):
  - Small labeled set X = {(x_m, y_m)}: image x_m and its ground-truth label y_m
  - Large unlabeled set U = {u_n}: image u_n without labels
- Outputs (after training/during inference):
  - For any input image, the model outputs class probabilities p(y|x) and a predicted class ŷ = argmax p
- Objective:
  - Minimize Ls (cross-entropy on labeled data) + Lu (masked cross-entropy on strongly augmented unlabeled data, using pseudo labels)
  - CPL/FlexMatch dynamically adjusts class-wise thresholds based on each class’s learning status to decide which unlabeled samples to learn when

2) Dataset-specific examples
- CIFAR-10
  - Classes: 10 (airplane, automobile, bird, …, truck)
  - Images: 32×32 RGB
  - Labels (example split): 40 labeled images total (4 per class); the remaining ~49,960 images are treated as unlabeled
  - Test set: 10,000 images with labels
- CIFAR-100
  - Classes: 100
  - Images: 32×32 RGB
  - Labels (example split): 400 labeled images (4 per class); the rest are unlabeled
  - Test set: 10,000 images
- STL-10
  - Classes: 10
  - Images: 96×96 RGB
  - Notes: large unlabeled set from a broader/partly different distribution; more realistic and challenging
  - Labels: e.g., 40, 2,500, or 10,000 depending on the split
- SVHN
  - Classes: 10 (digits 0–9)
  - Images: 32×32 RGB
  - Notes: relatively easy but potentially class-imbalanced; includes a large “extra” set (~531k)
  - Labels: e.g., 40 or 1,000 labeled
- ImageNet-1K
  - Classes: 1,000
  - Images: typically 224×224 RGB crops for training
  - Labels (example split): 100K labeled (100 per class), remaining treated as unlabeled
  - Evaluation: official validation with Top-1/Top-5

3) Input format and preprocessing/augmentation
- Weak augmentation ω: random crop with padding, horizontal flip, etc.
- Strong augmentation Ω: RandAugment (as in the paper), often with Cutout
- Normalization: dataset-specific mean/std
- Batch construction:
  - Labeled batch size B = 64
  - Unlabeled-to-labeled ratio μ = 7 → unlabeled batch size μB = 448
- Optimization:
  - SGD with momentum 0.9, initial LR 0.03, cosine decay, EMA 0.999
  - Base confidence threshold τ = 0.95 for FixMatch/FlexMatch (0.8 for UDA)
  - Loss weight λ = 1

4) Core of CPL/FlexMatch (class-wise adaptive thresholds)
- Notation:
  - pm,t(y|x): model probabilities at time t
  - q_b = pm(y|ω(u_b)): weak-aug probabilities for an unlabeled sample
  - c* = argmax q_b: predicted class on the weak view
- Estimating learning status with high-confidence counts:
  - Count only unlabeled predictions above the fixed τ to form a per-class tally σ_t(c):
    - σ_t(c) = Σ_n 1(max pm,t(y|u_n) > τ) · 1(argmax pm,t(y|u_n) = c)
  - Normalize β_t(c) = σ_t(c) / max_c σ_t(c)  (with warm-up: denominator = max{max_c σ_t(c), N − Σ_c σ_t(c)})
  - Class-wise threshold T_t(c) = M(β_t(c)) · τ  (M can be identity/linear or a convex mapping as in the paper)
- Masked unlabeled loss:
  - Lu,t = (1/(μB)) Σ_b 1(max q_b > T_t(c*)) · H(one-hot(c*), pm(y|Ω(u_b)))

5) One training step example (CIFAR-10 with 40 labels)
- Assumptions:
  - τ = 0.95, B = 64, μ = 7 → unlabeled batch size = 448
  - From previous steps, the high-confidence counts (σ_t) are:
    - σ_t = [30, 80, 10, 15, 60, 45, 20, 100, 25, 15] (max = 100)
  - β_t(c) = σ_t(c)/100 → [0.30, 0.80, 0.10, 0.15, 0.60, 0.45, 0.20, 1.00, 0.25, 0.15]
  - With a linear mapping M(x)=x, T_t(c) = β_t(c)·0.95
- Three unlabeled samples in the batch:
  1) u1: max(q1)=0.93 at class 8
     - Compare to T_t(8)=0.95 → 0.93 > 0.95 is False → masked out (not used this step)
  2) u2: max(q2)=0.92 at class 5
     - T_t(5)=0.60·0.95=0.57 → 0.92 > 0.57 is True → use ŷ=class 5, compute CE on strong aug Ω(u2)
  3) u3: max(q3)=0.60 at class 3
     - T_t(3)=0.10·0.95=0.095 → 0.60 > 0.095 is True → use ŷ=class 3 (encourages learning of a harder class)
- Labeled batch uses standard CE to compute Ls
- Final Lt = Ls + λ·Lu,t, then backprop/update
- Updating σ:
  - Independently, only predictions with max(q_b) > fixed τ=0.95 update ˆu_n and thus σ_t+1(c), which will refresh β and T in the next step
- Warm-up example:
  - Early in training, Σ_c σ_t(c) is small vs N, so denominator becomes max{max_c σ_t(c), N − Σ_c σ_t(c)}, keeping β_t(c) from spiking too early

6) Test-time inputs/outputs
- Input: test images (e.g., CIFAR-10: 10,000)
- Output: per-image class probabilities and predicted class (argmax)
- Metrics: Top-1 error/accuracy for CIFAR/SVHN/STL-10; Top-1/Top-5 for ImageNet

7) Additional concrete example (many classes: CIFAR-100 with 400 labels)
- Situation: 100 classes, some harder than others
- Suppose max σ_t is 250, and a hard class c has σ_t(c)=5 → β_t(c)=5/250=0.02
- With linear mapping, T_t(c)=0.02·0.95≈0.019 → many unlabeled samples of class c will pass the threshold earlier, improving recall and accelerating convergence for that class

8) Hyperparameters/models
- Models: Wide ResNet for CIFAR/STL/SVHN (as in the paper), ResNet-50 for ImageNet
- μ=7, τ=0.95 (FlexMatch/FixMatch), λ=1, LR 0.03, SGD+momentum 0.9, cosine decay, EMA 0.999
- Strong augmentation: RandAugment
- Evaluation: best checkpoint performance (and optionally median of the last 20 checkpoints)

9) Deliverables
- Trained classifier parameters
- Predictions on the test set with error/accuracy (and Top-5 for ImageNet)

10) Notes/edge cases
- Class-imbalanced unlabeled sets (e.g., SVHN) may keep some β_t(c) low for underrepresented classes, potentially admitting more noise for those classes
- For broader unlabeled distributions (e.g., STL-10), CPL tends to shine by giving more learning opportunities to hard classes
- Mapping M can be linear or convex; the paper compares several options and shows convex can help

이상으로, 각 데이터셋에서의 입력/출력, 학습·추론 파이프라인, 한 스텝의 수치 예시, 임계값 갱신 방식, 그리고 평가지표/산출물을 구체적으로 설명했습니다.

<br/>
# 요약


- 메서드: 클래스별 학습 상태를 “고신뢰 예측 수”로 추정해 가변 임계값 Tt(c)=M(βt(c))·τ를 동적으로 조정하는 Curriculum Pseudo Labeling(CPL)을 제안하고 FixMatch에 적용한 FlexMatch를 제시(임계값 워밍업·비선형 매핑 포함, 추가 연산·파라미터 거의 없음).
- 결과: 여러 SSL 벤치마크에서 SOTA 달성, 라벨이 매우 적거나 과제가 어려울수록 이점이 크고 수렴 속도도 크게 향상(예: FixMatch 대비 약 1/5 학습 시간 내에 더 나은 성능), Pseudo-Label·UDA에도 적용 시 성능 향상하며 반복당 시간 증가 거의 없음.
- 예시: CIFAR-100(400 라벨) 오류율 46.42%→39.94%, STL-10(40 라벨) 35.97%→29.15%(상대 18.96% 감소), ImageNet(클래스당 100 라벨) Top-1 43.66%→42.02%; UDA도 STL-10(40 라벨) 37.42%→29.53%로 개선.




- Method: Proposes Curriculum Pseudo Labeling (CPL) that estimates per-class learning status via counts of high-confidence predictions and sets flexible thresholds Tt(c)=M(βt(c))·τ; applied to FixMatch as FlexMatch with threshold warm-up and a non-linear mapping, adding virtually no extra computation or parameters.
- Results: Achieves state-of-the-art across SSL benchmarks, with larger gains when labels are scarce or tasks are harder, and markedly faster convergence (e.g., surpasses FixMatch in about 1/5 of the training time); also boosts Pseudo-Labeling and UDA without increasing per-iteration runtime.
- Examples: CIFAR-100 (400 labels) error 46.42%→39.94%, STL-10 (40 labels) 35.97%→29.15% (18.96% relative reduction), ImageNet (100 labels/class) Top-1 43.66%→42.02%; UDA on STL-10 (40 labels) improves 37.42%→29.53%.

<br/>
# 기타



- Figure 1: CPL 개념도
  - 결과: 각 클래스의 “학습 효과”를, 높은 고정 임계값(τ)을 넘는 예측 수로 근사해 추정하고, 이를 정규화(βt(c))해 클래스별 유연 임계값 Tt(c)=βt(c)·τ로 조절.
  - 인사이트: 잘 배우는 클래스는 높은 임계값을 유지해 고품질 샘플만 통과시키고, 어려운 클래스는 임계값을 낮춰 더 많은 샘플을 반영해 데이터 활용률과 균형 학습을 동시에 달성. 임계값은 학습 진행에 따라 올라가거나 내려갈 수 있음.

- Algorithm 1: FlexMatch 알고리즘(의사코드)
  - 결과: 클래스별 추정 학습효과 σ(c)→정규화 β(c)→임계값 T(c) 계산, 약한 증강으로 예측이 τ를 넘는 unlabeled에 대해 예측 클래스 갱신, 강한 증강에 대한 비지도 손실로 학습.
  - 인사이트: 추가 추론이나 파라미터 없이 거의 비용 없이 동작. 초기 편향을 완화하기 위해 “미사용(unused)” 클래스를 포함하는 워밍업 정규화(식 11)로 임계값을 점진적으로 올림.

- Table 1: CIFAR-10/100, STL-10, SVHN 성능
  - 결과: FlexMatch가 대부분 벤치마크에서 SOTA. 특히 레이블이 극히 적거나 과제가 어려울수록 이점 큼.
    - CIFAR-100, 400 labels: FlexMatch 39.94% vs FixMatch 46.42% (−6.48pt).
    - STL-10, 40 labels: 29.15% vs 35.97% (상대 18.96% 개선).
    - CIFAR-10, 40 labels: 4.97% vs 7.47% (−2.50pt).
  - 인사이트:
    - CPL은 다른 SSL에도 이식 시 이득(Flex-PL/Flex-UDA 모두 개선). 예: STL-10 40 labels에서 UDA 37.42% → Flex-UDA 29.53%.
    - 예외적으로 SVHN에서는 CPL이 덜 유리. 클래스 불균형 탓에 일부 클래스 임계값이 낮게 유지되어 노이즈가 유입될 수 있어, 높은 고정 임계값(예: FixMatch, UDA)이 더 안정적일 수 있음. 표에서도 SVHN 40 labels는 Flex-UDA가 최저(3.42%), 1000 labels는 UDA가 최저(1.89%).

- Figure 2: 평균 반복당 러닝타임
  - 결과: CPL 추가 전후 반복당 시간 거의 동일.
  - 인사이트: CPL의 실용 비용이 사실상 무시 가능해, 성능 향상을 “공짜로” 얻음.

- Figure 3: 수렴 분석
  - 결과:
    - CIFAR-100, 400 labels에서 FlexMatch는 손실이 더 빠르고 매끄럽게 감소, 5만 iter 부근에 이미 FixMatch 최종 성능을 추월.
    - CIFAR-10 초기 20만 iter에서 FixMatch 전체 정확도 ≈56.4% vs FlexMatch ≈94.3%. FixMatch는 절반의 클래스가 미성숙, FlexMatch는 대부분의 클래스를 빠르게 견인.
  - 인사이트: 클래스별 임계값으로 “어려운 클래스”에 더 많은 샘플을 조기 공급, 전반 학습 속도 가속. 수렴 속도 차이로 “마지막 체크포인트들의 중앙값” 평가는 공정성 이슈가 있을 수 있음.

- Table 2: ImageNet-1K(약 8% 레이블, 220k iter)
  - 결과: Top-1 에러 43.66→42.02, Top-5 21.80→19.49로 FlexMatch 우위.
  - 인사이트: 대규모·부분 불균형 환경에서도 CPL 이득 지속. 아직 완전 수렴 전이며 별도 튜닝 없이도 개선 확인.

- Figure 4: 어블레이션
  - (a) 임계값 상한 τ: 최적은 약 0.95. 너무 낮거나 높으면 성능 저하. FlexMatch에선 τ가 상한뿐 아니라 “학습효과 카운트” 기준 자체에도 영향.
  - (b) 매핑 함수 M(x): 볼록(convex)이 최적, 선형은 중간, 오목(concave)은 최하. 즉 β가 커질수록 임계값이 더 민감히 커지도록 하는 형태가 유리.
  - (c) 워밍업: CIFAR-10/100 모두에서 워밍업이 성능과 안정성 개선(특히 CIFAR-100). 초기 예측 편향·확증편향을 줄여 클래스별 임계값이 성급히 왜곡되는 것을 방지.

- Appendix(논문 본문 언급 범위)
  - 포함 내용: 추가 지표(Precision/Recall/F1/AUC), 마지막 20 체크포인트 중앙값 결과(여기서도 FlexMatch 우수), 더 많은 베이스라인 비교, 하이퍼파라미터 상세, TorchSSL 코드베이스 설명.
  - 인사이트: 다양한 평가 관점과 구현 세부까지 고려해도 CPL/ FlexMatch의 개선 추세가 일관적임을 뒷받침.




- Figure 1: CPL schematic
  - Result: Estimate per-class “learning effect” by counting predictions above a high fixed threshold, normalize to βt(c), and set class-wise flexible thresholds Tt(c)=βt(c)·τ.
  - Insight: Easy classes keep higher thresholds to admit only high-quality pseudo-labels; hard classes get lower thresholds to leverage more data early, improving utilization and class balance. Thresholds can go up or down as learning evolves.

- Algorithm 1: FlexMatch pseudocode
  - Result: Compute σ(c)→β(c)→T(c), update unlabeled predictions that pass τ under weak aug, train strong aug with unsupervised loss.
  - Insight: Near-zero overhead—no extra forward passes or parameters. Warm-up via an “unused” class stabilizes early-phase thresholds against initialization bias.

- Table 1: CIFAR-10/100, STL-10, SVHN
  - Result: FlexMatch is SOTA on most splits, shining with scarce labels and harder tasks.
    - CIFAR-100, 400 labels: 39.94% vs 46.42% (−6.48pt).
    - STL-10, 40 labels: 29.15% vs 35.97% (−6.82pt; 18.96% relative).
    - CIFAR-10, 40 labels: 4.97% vs 7.47% (−2.50pt).
  - Insight:
    - CPL generalizes to other SSLs (Flex-PL/Flex-UDA outperform their bases). Example: STL-10 40 labels, UDA 37.42% → Flex-UDA 29.53%.
    - Exception on SVHN: class imbalance keeps thresholds low for minority classes, inviting noise; high fixed thresholds (FixMatch/UDA) can be preferable. Table shows Flex-UDA best at 40 labels (3.42%), UDA best at 1000 labels (1.89%), while FlexMatch is worse on SVHN.

- Figure 2: Per-iteration runtime
  - Result: Adding CPL does not increase iteration time noticeably.
  - Insight: CPL delivers accuracy gains at virtually no computational cost.

- Figure 3: Convergence analysis
  - Result: On CIFAR-100 (400 labels), FlexMatch’s loss decreases faster and smoother, surpassing FixMatch’s final performance around 50k iters. On CIFAR-10 at 200k iters, overall acc ≈56.4% (FixMatch) vs ≈94.3% (FlexMatch).
  - Insight: Class-wise thresholds feed more data to hard classes early, accelerating global convergence. Median-of-last-checkpoints can be unfair when convergence speeds differ substantially.

- Table 2: ImageNet-1K (≈8% labels, 220k iters)
  - Result: Top-1 error 43.66→42.02, Top-5 21.80→19.49 with FlexMatch.
  - Insight: CPL helps even on large, partially imbalanced datasets without extra tuning, though training was not fully converged.

- Figure 4: Ablations
  - (a) τ upper bound: ≈0.95 is best; too low/high degrades. In FlexMatch τ also affects the learning-effect counts.
  - (b) Mapping M(x): Convex best, linear middle, concave worst. Making thresholds more sensitive at higher β is beneficial.
  - (c) Warm-up: Improves stability and accuracy on CIFAR-10/100 (especially CIFAR-100) by mitigating early confirmation bias.

- Appendix (as referenced)
  - Contents: Extra metrics (Precision/Recall/F1/AUC), median-of-last-20 results (still favoring FlexMatch), more baselines, hyperparameter details, TorchSSL codebase notes.
  - Insight: Broad evaluations and implementation details consistently support the effectiveness of CPL/FlexMatch.

<br/>
# refer format:



BibTeX
@inproceedings{zhang2021flexmatch,
  title     = {FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling},
  author    = {Zhang, Bowen and Wang, Yidong and Hou, Wenxin and Wu, Hao and Wang, Jindong and Okumura, Manabu and Shinozaki, Takahiro},
  booktitle = {Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021)},
  year      = {2021}
}


Zhang, Bowen, Yidong Wang, Wenxin Hou, Hao Wu, Jindong Wang, Manabu Okumura, and Takahiro Shinozaki. 2021. “FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling.” In Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS 2021).
