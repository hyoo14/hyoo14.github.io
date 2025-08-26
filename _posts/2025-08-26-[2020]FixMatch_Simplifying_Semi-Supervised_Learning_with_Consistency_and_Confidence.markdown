---
layout: post
title:  "[2020]FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
date:   2025-08-26 17:49:54 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

FixMatch는 약한 증강으로 얻은 고신뢰 예측을 원-핫 가짜 라벨로 채택하고, 같은 이미지의 강한 증강에 대해 그 라벨을 맞추도록 교차엔트로피로 학습하는 간단한 결합 방식(일관성 정규화 + pseudo-labeling)이며, 강한 증강으로 RandAugment/CTAugment와 Cutout을 사용한다.

즉, FixMatch 같은 SSL 기법은 **모든 클래스에 동일한 고정 임계값(예: 0.95)**을 두고, 그 이상 confidence가 나온 unlabeled sample만 학습에 사용  


짧은 요약(Abstract) :

- 이 논문은 라벨이 적을 때도 성능을 높이기 위한 반지도학습(SSL)을 더 단순하게 만드는 방법 FixMatch를 제안합니다.
- 핵심 아이디어는: 약한 증강(간단한 플립/시프트)을 적용한 미라벨 이미지에 대한 모델 예측으로 가짜 라벨(추정 라벨)을 만들고, 그 확신도가 임계값 이상일 때만 채택합니다. 그런 다음 같은 이미지에 강한 증강(강한 왜곡)을 적용해 입력했을 때 이 가짜 라벨을 맞추도록 모델을 학습시킵니다.
- 방법은 매우 단순하지만, 여러 표준 SSL 벤치마크에서 최신 성능을 달성했습니다. 예: CIFAR-10에서 250개 라벨로 94.93% 정확도, 40개 라벨(즉, 클래스당 4개)로 88.61% 정확도.
- 또한 광범위한 어블레이션을 통해 FixMatch 성능에 중요한 실험 요소들을 분석했습니다.
- 코드: https://github.com/google-research/fixmatch


Semi-supervised learning (SSL) provides an effective means of leveraging unlabeled data to improve a model’s performance. This domain has seen fast progress recently, at the cost of requiring more complex methods. In this paper we propose FixMatch, an algorithm that is a significant simplification of existing SSL methods. FixMatch first generates pseudo-labels using the model’s predictions on weakly-augmented unlabeled images. For a given image, the pseudo-label is only retained if the model produces a high-confidence prediction. The model is then trained to predict the pseudo-label when fed a strongly-augmented version of the same image. Despite its simplicity, we show that FixMatch achieves state-of-the-art performance across a variety of standard semi-supervised learning benchmarks, including 94.93% accuracy on CIFAR-10 with 250 labels and 88.61% accuracy with 40 – just 4 labels per class. We carry out an extensive ablation study to tease apart the experimental factors that are most important to FixMatch’s success. The code is available at https://github.com/google-research/fixmatch .


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



1) 목표와 핵심 아이디어
- 목표: 극소량의 라벨과 대량의 비라벨 데이터를 활용해 이미지 분류 성능을 크게 향상시키는 간단한 반지도학습(SSL) 알고리즘.
- 핵심 아이디어: 두 가지 표준 기법을 간결하게 결합
  - Consistency regularization: 같은 이미지의 서로 다른 변형(증강)에 대해 일관된 예측을 하도록 학습.
  - Pseudo-labeling: 비라벨 샘플에 대해 모델의 예측을 인공 라벨로 사용하되, 높은 신뢰도(확률 임계값)를 만족하는 경우만 사용.
- 중요 포인트: 인공 라벨은 약한 증강(weak augmentation)에서 얻고, 그 라벨을 강한 증강(strong augmentation)을 적용한 같은 이미지에 대해 맞추도록 학습시킨다. 즉, 약증강→라벨 생성, 강증강→예측 정합의 형태로 일관성 규제를 건다.

2) 알고리즘 개요와 손실 함수
- 표기:
  - L-클래스 분류, 라벨 배치 X={(xb, pb)}(b=1..B), 비라벨 배치 U={ub}(b=1..μB). μ는 비라벨/라벨 배치 크기 비율.
  - 모델 분포 pm(y|x). 교차엔트로피 H(p, q).
  - 약한 증강 φ(·), 강한 증강 A(·).
- 지도 손실(라벨 데이터):
  - 약증강된 라벨 샘플에 대한 표준 교차엔트로피: Ls = (1/B) Σ H(pb, pm(y|φ(xb))).
- 비지도 손실(비라벨 데이터):
  - 약증강 이미지 ub에 대한 예측분포 q_b = pm(y|φ(ub))를 계산.
  - 최대 확률이 임계값 τ(논문에서는 0.95 사용) 이상일 때만, ŷ_b = argmax(q_b)를 원-핫 가짜라벨로 채택.
  - 강증강 이미지 A(ub)에 대해 ŷ_b와의 교차엔트로피를 계산: Lu = (1/(μB)) Σ 1(max(q_b) ≥ τ) H(ŷ_b, pm(y|A(ub))).
- 전체 손실:
  - L = Ls + λu·Lu, 여기서 λu는 비지도 손실 가중치(논문 기본값 1).
- 학습 동작상의 특징:
  - 초반에는 모델 신뢰도가 낮아 비지도 항이 자연히 억제되고, 학습이 진행되며 점차 더 많은 비라벨 샘플이 기여한다(자연스러운 커리큘럼 효과).
  - 실무 팁: 라벨 데이터를 U에도(라벨 제거 후) 포함해 비라벨 배치를 구성.

3) 데이터 증강 설계
- 약한 증강(weak):
  - 수평 뒤집기(일부 데이터셋 제외, 예: SVHN)와 최대 12.5% 범위의 평행 이동.
- 강한 증강(strong):
  - AutoAugment 계열의 두 방법을 사용하고, 이후 Cutout을 추가 적용.
    - RandAugment(RA): 변환 종류를 무작위로 선택하고, 변환 강도는 사전 정의 범위에서 무작위 샘플링.
    - CTAugment(CTA): 각 변환의 강도를 학습 중 동적으로 조절.
  - 강증강 뒤 Cutout을 결합하는 것이 성능에 중요(어블레이션에서 제거 시 에러 증가).

4) 최적화와 학습 설정(간단하지만 중요한 요소)
- 옵티마이저: SGD(momentum) 사용(Adam은 성능 저하 관찰). Nesterov 사용 여부는 큰 차이 없음.
- 정규화: weight decay가 특히 중요.
- 학습률 스케줄: cosine decay 사용.
- 평가: 파라미터의 지수이동평균(EMA)을 사용해 최종 성능 보고.
- 하이퍼파라미터(논문 기본 설정 예):
  - λu=1, 초기 학습률=0.03, momentum=0.9, 신뢰도 임계값 τ=0.95, μ=7, 라벨 배치 크기 B=64.
  - 모든 데이터셋(CIFAR-10/100, SVHN, STL-10)에서 동일 설정으로 강건한 성능.

5) 모델과 아키텍처
- 방법 자체는 아키텍처 비종속적.
- 실험에서는 표준 비전 백본 사용:
  - CIFAR-10, SVHN: Wide ResNet-28-2
  - CIFAR-100: Wide ResNet-28-8
  - STL-10: Wide ResNet-37-2
  - ImageNet: ResNet-50
- FixMatch는 특별한 네트워크 설계 없이도 효과를 보임.

6) 성능 요약(대표 결과)
- CIFAR-10(라벨 250개): 94.93% 정확도(SOTA).
- CIFAR-10(라벨 40개): 88.61% 정확도(클래스당 4개 라벨).
- ImageNet(라벨 10%): Top-1 에러 28.54%로 UDA 대비 2.68%p 개선.
- SVHN, STL-10 등에서도 강력한 성능. CIFAR-100에서는 ReMixMatch가 약간 우세했지만, FixMatch에 Distribution Alignment를 추가하면 더 향상.

7) 어블레이션(무엇이 중요한가)
- 신뢰도 임계값(τ): 높을수록(예: 0.95) 가짜라벨 품질이 좋아져 성능 향상. 낮으면 잡음 증가로 성능 하락.
- Sharpening(소프트 라벨 온도 조정): 임계값 기반의 하드 가짜라벨을 대체해도 성능 이득이 거의 없고, 오히려 하이퍼파라미터가 늘어남.
- 강한 증강의 필요성: 강증강 없이(혹은 잘못된 위치에 사용) 학습이 불안정하거나 붕괴. “약증강→라벨, 강증강→예측”의 역할 분리가 핵심.
- Cutout: 강증강 뒤 Cutout 포함이 큰 도움.
- 최적화/정규화: SGD+momentum, weight decay, cosine decay, EMA가 중요한 기반 요소.

8) 극저라벨(1-shot per class)에서도 유효
- CIFAR-10에서 클래스당 1개 라벨로도 중간값 기준 약 64% 정확도(최대 85%도 관찰).
- 선택된 라벨의 “대표성(prototypicality)”이 높을수록 성능이 크게 오른다는 관찰.

9) 확장 가능성
- 간단 구조 덕분에 다른 SSL 기법을 쉽게 결합 가능:
  - Distribution Alignment(라벨 분포 정합), Augmentation Anchoring(강증강 M개로 일관성), MixUp, 적대적 교란 등.
- 실제로 CIFAR-100에서 DA 결합 시 큰 이득.

10) 장점과 한계
- 장점: 매우 간단(표준 CE 손실만), 하이퍼파라미터가 적고 재현/이식 쉽다. 강력한 성능과 레이블 효율성.
- 한계/주의: 극저라벨 구간에서 fold/seed에 대한 분산이 커질 수 있음. 클래스 분포 편향이 존재하면 DA 같은 보완책이 유효.

11) 구현 절차(요약)
- 각 스텝에서:
  1) 라벨 배치 X에 약증강 φ 적용 → CE로 지도 손실 계산.
  2) 비라벨 배치 U에 약증강 φ 적용 → q_b 계산.
  3) max(q_b) ≥ τ인 샘플만 ŷ_b=argmax(q_b) 채택.
  4) 같은 샘플에 강증강 A 적용 → pm(y|A(ub))와 ŷ_b 사이 CE로 비지도 손실 계산.
  5) 전체 손실 Ls + λu·Lu로 업데이트(SGD+momentum, cosine LR).
  6) EMA로 파라미터 추적 및 평가.





1) Goal and Core Idea
- Goal: A simple yet powerful SSL algorithm that leverages large unlabeled data with very few labels for image classification.
- Core idea: A clean combination of two common techniques
  - Consistency regularization: enforce consistent predictions under input perturbations.
  - Pseudo-labeling: use model predictions as artificial labels, but only when the model is confident.
- Key twist: Generate artificial labels from weakly augmented images and enforce them against predictions on strongly augmented versions of the same images. That is, “weak→label, strong→prediction,” creating a strong form of consistency.

2) Algorithm and Losses
- Notation:
  - L-class classification, labeled batch X={(xb, pb)}(b=1..B), unlabeled batch U={ub}(b=1..μB) where μ is unlabeled-to-labeled ratio.
  - Model distribution pm(y|x). Cross-entropy H(p, q).
  - Weak augmentation φ(·), strong augmentation A(·).
- Supervised loss:
  - Standard cross-entropy on weakly augmented labeled data: Ls = (1/B) Σ H(pb, pm(y|φ(xb))).
- Unsupervised loss:
  - Compute q_b = pm(y|φ(ub)) on weakly augmented ub.
  - If max(q_b) ≥ τ (confidence threshold, 0.95 in the paper), take ŷ_b = argmax(q_b) as a one-hot pseudo-label.
  - Compute Lu = (1/(μB)) Σ 1(max(q_b) ≥ τ) H(ŷ_b, pm(y|A(ub))) on the strongly augmented A(ub).
- Total loss:
  - L = Ls + λu·Lu with λu typically set to 1.
- Training dynamics:
  - Early in training, few unlabeled examples pass the threshold, yielding a natural curriculum; later, more unlabeled samples contribute as confidence grows.
  - In practice, labeled examples (without labels) are also included in U.

3) Augmentation Design
- Weak augmentation:
  - Random horizontal flip (not used for SVHN) and random translations up to 12.5%.
- Strong augmentation:
  - AutoAugment-style policies followed by Cutout:
    - RandAugment (RA): randomly sample transform types; sample a global magnitude from a predefined range.
    - CTAugment (CTA): learn magnitudes on the fly during training.
  - Adding Cutout after strong augmentation is important; removing it degrades performance in ablations.

4) Optimization and Training Setup
- Optimizer: SGD with momentum (Adam underperformed in this setting). Nesterov did not make a notable difference.
- Regularization: weight decay is particularly important.
- Learning rate schedule: cosine decay.
- Evaluation: use an exponential moving average (EMA) of model parameters.
- Default hyperparameters (used across datasets with strong results):
  - λu=1, initial LR=0.03, momentum=0.9, threshold τ=0.95, μ=7, labeled batch size B=64.

5) Model and Architecture
- Architecture-agnostic method.
- Backbones used in experiments:
  - CIFAR-10, SVHN: Wide ResNet-28-2
  - CIFAR-100: Wide ResNet-28-8
  - STL-10: Wide ResNet-37-2
  - ImageNet: ResNet-50
- No special architectural tricks are required.

6) Performance Highlights
- CIFAR-10 with 250 labels: 94.93% accuracy (state of the art).
- CIFAR-10 with 40 labels (4 per class): 88.61% accuracy.
- ImageNet with 10% labels: 28.54% top-1 error, 2.68% better than UDA.
- Strong results on SVHN and STL-10. On CIFAR-100, ReMixMatch slightly outperforms FixMatch, but adding Distribution Alignment (DA) to FixMatch yields larger gains.

7) Ablation Insights
- Confidence threshold τ: higher is better (e.g., 0.95). Low thresholds admit noisy pseudo-labels and hurt performance.
- Sharpening vs. hard pseudo-labels: when thresholding is used, replacing argmax with temperature-sharpened soft targets yields no benefit and adds a hyperparameter.
- Necessity of strong augmentation: using strong augmentation for the prediction path is crucial; violating the “weak→label, strong→prediction” pattern causes instability or collapse.
- Cutout matters: combining strong augmentation with Cutout yields notable gains.
- Optimization choices: SGD+momentum, weight decay, cosine LR, and EMA are important training ingredients.

8) Barely Labeled Regime (1 shot per class)
- On CIFAR-10, FixMatch achieves around 64% median accuracy with 1 label per class (up to 85% observed).
- The “prototypicality” of the labeled exemplars strongly influences performance.

9) Extensibility
- The simplicity of FixMatch allows easy integration of other SSL ideas:
  - Distribution Alignment, Augmentation Anchoring, MixUp, adversarial perturbations, etc.
- For example, adding DA helps substantially on CIFAR-100.

10) Pros and Caveats
- Pros: very simple (just cross-entropy losses), few hyperparameters, strong label efficiency and performance, easy to reproduce and port.
- Caveats: in extremely low-label settings, variance across folds/seeds can be high; if class distribution bias exists, DA-like corrections help.

11) Implementation Steps (concise)
- For each training step:
  1) Apply weak augmentation φ to labeled batch X → compute Ls with cross-entropy.
  2) Apply φ to unlabeled batch U → get q_b.
  3) For samples with max(q_b) ≥ τ, set ŷ_b = argmax(q_b).
  4) Apply strong augmentation A to the same unlabeled samples → compute Lu between ŷ_b and pm(y|A(ub)).
  5) Optimize L = Ls + λu·Lu with SGD+momentum and cosine LR.
  6) Track EMA of parameters and use it for evaluation.

위 요약은 주어진 논문 본문(알고리즘 정의, 증강 전략, 하이퍼파라미터, 학습 설정, 어블레이션, 실험 결과)에 기반해 작성되었습니다.


<br/>
# Results



개요
- FixMatch는 약한 증강(weak aug.)으로 얻은 예측을 고신뢰(threshold)인 경우에만 하드(one-hot) 의사라벨로 채택하고, 같은 이미지의 강한 증강(strong aug.)에 대해 그 의사라벨을 맞추도록 학습하는 간단한 SSL 기법입니다.
- 강한 증강은 RandAugment(RA) 또는 CTAugment(CTA)에 Cutout을 결합해 사용했고, 학습은 표준 교차엔트로피만으로 진행합니다.
- 주요 메트릭: 분류 오류율(낮을수록 좋음), 일부 실험에서 정확도(높을수록 좋음). ImageNet에서는 top-1, top-5 오류율.
- 공정 비교: 동일 코드베이스·아키텍처·최적화 설정 하에 Π-Model, Pseudo-Labeling, Mean Teacher, MixMatch, UDA, ReMixMatch와 직접 비교. 결과는 5개 라벨 폴드에 대한 평균±표준편차로 보고.

테스트 데이터셋과 설정
- CIFAR-10/100, SVHN: 표준 SSL 프로토콜. 아키텍처는 WRN-28-2(CIFAR-10, SVHN), WRN-28-8(CIFAR-100).
- STL-10: 라벨 1,000개/폴드(총 5폴드), WRN-37-2.
- ImageNet: 학습 데이터의 10%만 라벨 사용, 나머지는 비라벨. ResNet-50.
- 공통 하이퍼파라미터(이미지넷 제외): unlabeled: labeled 배치 비율 μ=7, 배치 B=64, EMA 사용, unlabeled 손실 가중치 1, 신뢰 임계값 τ=0.95 등 동일하게 고정.

비교 대상(경쟁 모델)
- Π-Model, Pseudo-Labeling, Mean Teacher, MixMatch, UDA, ReMixMatch
- FixMatch은 RA/CTA 두 강증강 변형으로 보고: “FixMatch (RA)”, “FixMatch (CTA)”

정량 결과 요약

1) CIFAR-10
- 40 라벨(클래스당 4장): FixMatch (CTA) 11.39±3.35% 오류, FixMatch (RA) 13.81±3.37%
  - ReMixMatch 19.10±9.64%, UDA 29.05±5.93%, MixMatch 47.54±11.50% 대비 크게 우수
  - 논문 메인 주장: 40 라벨에서 88.61% 정확도
- 250 라벨: FixMatch (CTA/RA) 5.07±0.33/0.65% 오류
  - ReMixMatch 5.44±0.05%, UDA 8.82±1.08%, MixMatch 11.05±0.86% 대비 우수
  - 논문 메인 주장: 250 라벨에서 94.93% 정확도(= 약 5.07% 오류와 일치)
- 4,000 라벨: FixMatch (CTA) 4.31±0.15%, (RA) 4.26±0.05%
  - ReMixMatch 4.72±0.13%, UDA 4.88±0.18%, MixMatch 6.42±0.10% 대비 우수

2) CIFAR-100
- 400 라벨: FixMatch (CTA) 49.95±3.01%
  - ReMixMatch 44.28±2.06%가 더 낮은 오류(더 우수)
  - 단, FixMatch에 Distribution Alignment(DA)를 추가하면 40.14%로 ReMixMatch(44.28%)를 능가
- 2,500 라벨: FixMatch (CTA) 28.64±0.24% vs ReMixMatch 27.43±0.31%
- 10,000 라벨: FixMatch (CTA) 23.18±0.11% vs ReMixMatch 23.03±0.56%
- 요약: CIFAR-100에서는 ReMixMatch가 근소하게 앞섰으나, DA를 결합한 FixMatch 변형은 400라벨 설정에서 SOTA 달성

3) SVHN
- 40 라벨: FixMatch (RA) 3.96±2.17%, (CTA) 7.65±7.65%로 분산 큼
  - ReMixMatch 3.34±0.20%가 근소 우위
- 250 라벨: FixMatch (RA/CTA) 2.48±0.38/0.64% vs ReMixMatch 2.92±0.48% → FixMatch 우수
- 1,000 라벨: FixMatch (RA/CTA) 2.28±0.11/0.19% vs ReMixMatch 2.65±0.08% → FixMatch 우수
- 요약: 극저라벨(40)에서는 ReMixMatch가 조금 낫지만, 250/1000 라벨에서는 FixMatch가 더 좋음

4) STL-10 (라벨 1,000)
- FixMatch (CTA) 5.17±0.63%, ReMixMatch 5.23±0.45% → 사실상 동급 혹은 근소 우위
- FixMatch (RA) 7.98±1.50%는 CTA 대비 열세
- 요약: CTA를 사용할 경우 SOTA에 필적하는 성능을 단순한 방법으로 달성

5) ImageNet (라벨 10%)
- FixMatch: top-1 오류 28.54±0.52%, top-5 오류 10.87±0.28%
- UDA 대비 top-1 오류 2.68%p 개선
- S4L은 추가 훈련 단계(의사라벨 재학습 + 지도 미세튜닝)까지 포함할 때 26.79%로 더 낮은 오류이나, 1단계 결과만 비교하면 FixMatch가 S4L의 1단계 결과(30.27%)보다 우수

강증강 선택의 영향
- RA와 CTA는 대부분 벤치마크에서 유사 성능. 단, STL-10에서는 CTA가 명확히 우수.
- Cutout 제거 또는 Cutout만 사용(증강 없이)은 모두 오류 증가(예: CIFAR-10 250라벨 단일 폴드에서 4.84% → 6.15%)

추가 관찰(극저라벨: 1-shot per class, CIFAR-10)
- 클래스당 1장 라벨만 사용할 때, 4개의 무작위 데이터셋에서 정확도 48.58%~85.32%(중앙값 64.28%)
- “전형적(prototypical)”인 예시들만 선택하면 중앙값 78%(최대 84%)까지 향상, 반대로 이상치만 선택하면 학습 실패(10% 정확도)

메트릭·프로토콜 요약
- 메트릭: 오류율(=100-정확도), ImageNet은 top-1/top-5 오류율
- 보고 방식: 5개 라벨 폴드 평균±표준편차
- 공정성: 동일한 네트워크, 최적화기(SGD+모멘텀), 코사인 러닝레이트, 전처리·증강, 배치/스텝 수로 통제

핵심 결론
- FixMatch는 간결한 파이프라인(추가 손실·스케줄 최소화)에도 불구하고 대부분의 표준 SSL 벤치마크에서 SOTA 혹은 SOTA급 성능을 달성.
- CIFAR-10, SVHN(250/1000 라벨), STL-10에서 경쟁 기법들을 능가하거나 동급. CIFAR-100에서는 ReMixMatch가 근소 우위지만, DA를 결합한 변형 FixMatch는 400라벨에서 최고 성능.
- 강한 증강과 신뢰 임계값 기반 하드 의사라벨이 성능 핵심. 임계값을 높여 의사라벨 품질을 담보하는 것이 양보다 중요하며, sharpening은 임계값 기반 하드 라벨 대비 이득이 거의 없음.




Overview
- FixMatch forms hard pseudo-labels from weakly augmented images if the prediction confidence exceeds a threshold, and trains the model to match these labels on strongly augmented versions. Strong augmentations are RandAugment (RA) or CTAugment (CTA) plus Cutout.
- Training uses only standard cross-entropy losses. Metrics are error rates (lower is better), with ImageNet reported as top-1/top-5 error. All baselines are re-implemented in the same codebase and trained under identical protocols for fair comparison; results are mean±std over 5 labeled folds.

Datasets and setup
- CIFAR-10/100, SVHN with WRN-28-2 (CIFAR-10, SVHN) and WRN-28-8 (CIFAR-100).
- STL-10 with 1,000 labeled samples per predefined fold (5 folds), WRN-37-2.
- ImageNet with 10% labeled data, ResNet-50.
- Common hyperparameters (except ImageNet) fixed across datasets: unlabeled:labeled ratio μ=7, batch B=64, confidence threshold τ=0.95, unlabeled loss weight 1, EMA, etc.

Baselines
- Π-Model, Pseudo-Labeling, Mean Teacher, MixMatch, UDA, ReMixMatch
- Two FixMatch variants: “FixMatch (RA)” and “FixMatch (CTA)”

Key quantitative results

1) CIFAR-10
- 40 labels: FixMatch (CTA) 11.39±3.35% error; (RA) 13.81±3.37%
  - Beats ReMixMatch 19.10±9.64%, UDA 29.05±5.93%, MixMatch 47.54±11.50%
  - Headline number: 88.61% accuracy with 40 labels
- 250 labels: FixMatch (CTA/RA) 5.07±0.33/0.65%
  - Better than ReMixMatch 5.44±0.05%, UDA 8.82±1.08%, MixMatch 11.05±0.86%
  - Headline number: 94.93% accuracy with 250 labels
- 4,000 labels: FixMatch (CTA/RA) 4.31±0.15/4.26±0.05%
  - Better than ReMixMatch 4.72±0.13%, UDA 4.88±0.18%, MixMatch 6.42±0.10%

2) CIFAR-100
- 400 labels: FixMatch (CTA) 49.95±3.01% vs ReMixMatch 44.28±2.06% (ReMixMatch better)
  - However, FixMatch + Distribution Alignment (variant) reaches 40.14%, surpassing ReMixMatch
- 2,500 labels: FixMatch (CTA) 28.64±0.24% vs ReMixMatch 27.43±0.31%
- 10,000 labels: FixMatch (CTA) 23.18±0.11% vs ReMixMatch 23.03±0.56%
- Summary: ReMixMatch slightly ahead on CIFAR-100 unless adding DA to FixMatch

3) SVHN
- 40 labels: FixMatch (RA) 3.96±2.17% (CTA shows high variance 7.65±7.65%); ReMixMatch 3.34±0.20% slightly better
- 250 labels: FixMatch (RA/CTA) 2.48±0.38/0.64% vs ReMixMatch 2.92±0.48% → FixMatch better
- 1,000 labels: FixMatch (RA/CTA) 2.28±0.11/0.19% vs ReMixMatch 2.65±0.08% → FixMatch better

4) STL-10 (1,000 labeled)
- FixMatch (CTA) 5.17±0.63% vs ReMixMatch 5.23±0.45% → on par or slightly better
- FixMatch (RA) 7.98±1.50% worse than CTA
- Summary: With CTA, FixMatch matches SOTA with a simpler pipeline

5) ImageNet (10% labeled)
- FixMatch: top-1 error 28.54±0.52%, top-5 error 10.87±0.28%
- Improves over UDA by 2.68% absolute top-1 error
- S4L achieves 26.79% only after two extra training phases; after the first phase, FixMatch outperforms S4L’s first-phase result (30.27%)

Effect of strong augmentation
- RA and CTA perform similarly on most datasets; CTA is notably better on STL-10.
- Removing Cutout or using Cutout alone (without strong augmentation) increases error (e.g., on a CIFAR-10 250-label split, 4.84% → 6.15%).

Extreme low-label setting (1-shot per class, CIFAR-10)
- Across four random labeled sets: 48.58%–85.32% accuracy (median 64.28%)
- With “prototypical” examples only: median 78% (max 84%); with outlier examples only: fails to learn (10% accuracy)

Metrics and protocol
- Main metric: error rate (100 - accuracy); ImageNet reports top-1/top-5 error
- Reporting: mean±std over 5 labeled folds
- Fairness: same backbone, optimizer (SGD+momentum), cosine LR schedule, preprocessing/augmentation, batch/steps across methods

Key takeaways
- Despite its simplicity and few extra hyperparameters, FixMatch delivers SOTA or near-SOTA on most standard SSL benchmarks.
- It clearly outperforms prior methods on CIFAR-10 and on SVHN (for 250/1000 labels), and matches SOTA on STL-10 with CTA; on CIFAR-100 ReMixMatch is slightly better unless FixMatch is extended with Distribution Alignment.
- High-confidence hard pseudo-labels plus strong augmentation are the main drivers; prioritizing pseudo-label quality (via a high threshold) matters more than quantity, and sharpening brings little benefit when thresholding is used.


<br/>
# 예제


1) 문제 정의와 출력 형태
- 태스크: 이미지 분류(semi-supervised learning). 소량의 라벨된 이미지와 다량의 라벨 없는 이미지를 함께 사용해 분류기를 학습한다.
- 모델 출력: 입력 이미지 x에 대해 C개 클래스(예: CIFAR-10은 10클래스, CIFAR-100은 100클래스)의 확률 분포 p_m(y|x)와 최종 예측 클래스 argmax_y p_m(y|x).

2) 데이터셋별 입력/출력(이미지 크기, 클래스, 라벨 개수)
- CIFAR-10
  - 입력: 32x32 컬러 이미지, 10개 클래스.
  - 라벨 수(실험 설정): 예) 40 라벨(클래스당 4장), 250 라벨(클래스당 25장), 4000 라벨 등. 라벨 없는 데이터는 전체 50,000장(라벨 포함본도 비식별 상태로 함께 사용).
  - 출력: 10개 클래스 중 하나(예: airplane, automobile, bird, ...).
- CIFAR-100
  - 입력: 32x32 컬러 이미지, 100개 클래스.
  - 라벨 수(예): 400, 2500, 10,000 등.
  - 출력: 100개 클래스 중 하나.
- SVHN
  - 입력: 32x32 컬러 이미지(거리 숫자 이미지), 10개 클래스(0~9).
  - 라벨 수(예): 40, 250, 1000 등.
  - 출력: 0~9 중 하나.
- STL-10
  - 입력: 96x96 컬러 이미지, 10개 클래스.
  - 라벨 수: 각 fold마다 1,000장(총 5 folds). 라벨 없는 데이터 100,000장(여기엔 분포 밖(Out-of-Distribution) 이미지 일부 포함).
  - 출력: 10개 클래스 중 하나.
- ImageNet(반지도 설정)
  - 입력: 다양한 크기(표준 전처리 후 모델 입력 크기로 리사이즈/크롭), 1000 클래스.
  - 라벨 수: 학습 데이터의 10%만 라벨, 나머지 90%는 비라벨로 사용.
  - 출력: top-1 또는 top-5 예측(1000 클래스).

3) 학습 시의 구체적 입출력 흐름(배치 단위, 증강, 손실)
- 배치 구성
  - 라벨된 배치 X = {(x_b, p_b)}: 크기 B. p_b는 원-핫 라벨.
  - 비라벨 배치 U = {u_b}: 크기 μB (μ는 비라벨:라벨 비율, 논문 기본값 μ=7).
  - 논문 기본값 예: B=64, μ=7 → 라벨 64장, 비라벨 448장/스텝.
- 증강
  - 약한 증강 φ(·): 수평 뒤집기(대부분 데이터셋에서 확률 50%, SVHN은 제외), 상하/좌우 최대 12.5% 평행이동.
  - 강한 증강 A(·): RandAugment 또는 CTAugment로 강한 변형을 적용한 뒤 Cutout을 추가.
- 모델 입출력과 손실
  1) 지도(supervised) 손실 L_s:
     - 입력: 약하게 증강된 라벨 데이터 φ(x_b)
     - 출력: p_m(y|φ(x_b))
     - 손실: CE(p_b, p_m(y|φ(x_b)))의 평균
  2) 비지도(unsupervised) 손실 L_u:
     - 입력 A: 약하게 증강된 비라벨 φ(u_b)을 모델에 넣어 분포 q_b = p_m(y|φ(u_b)) 예측
     - 임계값(신뢰도) 검사: max(q_b) ≥ τ 이면(논문 기본값 τ=0.95) pseudo-label \hat{q}_b = one-hot(argmax(q_b)) 생성, 아니면 해당 샘플은 비지도 손실에 제외
     - 입력 B: 같은 원본 u_b의 강하게 증강된 A(u_b)을 모델에 넣어 p_m(y|A(u_b)) 예측
     - 손실: 임계 통과 샘플에 대해 CE(\hat{q}_b, p_m(y|A(u_b)))의 평균
  - 총 손실: L = L_s + λ_u L_u (논문 기본값 λ_u=1)
  - 옵티마이저/스케줄 등: SGD(m=0.9) + cosine learning rate decay, 가중치 감쇠(Weight Decay), EMA로 파라미터 이동평균 평가 등 사용.

4) 학습 단계 실제 예시(수치 예 포함)
- 예: CIFAR-10, 250 라벨(클래스당 25장), μ=7, B=64
  - 한 스텝에서
    - 라벨 배치: 64장 입력 → 약한 증강 φ(x) 적용 → p_m(y|φ(x)) 출력 → 지도 손실 계산
    - 비라벨 배치: 448장 입력
      - 각 u에 대해 약한 증강 φ(u) → q = p_m(y|φ(u)) 출력
      - 만약 q의 최대값이 0.97로 class=“cat”이라면 τ=0.95를 넘으므로 pseudo-label \hat{q} = one-hot(“cat”)
      - 같은 u에 대해 강한 증강 A(u) → p_m(y|A(u)) 출력
      - CE(\hat{q}, p_m(y|A(u)))를 비지도 손실에 더함
      - 반대로 max(q)=0.80처럼 낮으면 해당 샘플은 비지도 손실에서 제외
  - 직관: “덜 왜곡된(약한 증강) 입력에서 모델이 아주 자신 있는 예측만, 더 세게 왜곡된(강한 증강) 입력에 대해 동일하게 맞추도록 학습” → 일종의 일관성 규제 + 고신뢰 가짜 라벨링.

5) 테스트(평가) 단계의 구체적 입출력
- 입력: 테스트 세트의 원본 이미지(표준화 외에 특별한 강한 증강 없음; 일반적으로 한 번의 전방패스).
- 출력: 클래스 확률과 argmax 클래스.
- 지표:
  - CIFAR-10/100, SVHN, STL-10: Top-1 정확도(또는 에러율).
  - ImageNet(반지도 설정): Top-1/Top-5 에러율 보고.
- 주의: 학습 말기에 EMA 파라미터로 평가하여 더 안정적인 성능을 보고.

6) 태스크별 구체 시나리오 예
- 태스크 A: CIFAR-10, 250 라벨
  - 학습 입력: 50,000장 중 250장은 (이미지, 클래스 원-핫)로 지도, 나머지 49,750장은 비라벨로 사용.
  - 비지도 파이프라인: φ(u)에서 고신뢰 예측만 \hat{q}로 채택 → A(u) 예측을 \hat{q}에 맞추도록 CE로 학습.
  - 테스트 입력/출력: 테스트 10,000장 각각에 대해 클래스 확률 → argmax로 정확도 측정.
- 태스크 B: CIFAR-100, 400 라벨
  - 학습 입력: 50,000장 중 400장만 라벨, 나머지 49,600장 비라벨.
  - 출력: 100개 클래스 중 하나. 강한 증강/임계값/손실은 동일.
  - DA(Distribution Alignment) 같은 확장 기법을 추가하면 CIFAR-100에서 특히 도움.
- 태스크 C: SVHN, 1000 라벨
  - 입력 특이점: 숫자(0~9), 수평 플립은 사용하지 않음(약한 증강의 예외).
  - 나머지 파이프라인은 동일.
- 태스크 D: STL-10, fold당 1,000 라벨
  - 비라벨 100,000장 중 일부는 OOD. 강한 증강(A)로도 견고하게 일관성 학습.
  - WRN-37-2 사용(논문 실험 설정).
- 태스크 E: ImageNet, 10% 라벨
  - 학습 입력: 전체 학습셋의 10%만 라벨, 90% 비라벨.
  - 출력: 1000 클래스 확률. 보고 지표: Top-1/Top-5 에러율.
  - 결과: UDA 대비 유의한 개선.

7) 거의 무라벨(Barely-Supervised) 사례(1클래스당 1장)
- CIFAR-10에서 클래스당 1장(총 10장)만 라벨로 학습.
- 나머지 전부를 비라벨로 사용해 동일 파이프라인 적용.
- 결과는 선택된 10장의 “대표성”에 크게 좌우(최대 ~84%까지 도달 사례, 대표성 낮으면 실패). 이는 고신뢰 임계값 기반의 pseudo-label 품질이 초기 라벨 표본의 질에 민감함을 시사.

8) 핵심 하이퍼파라미터 요약(논문 기본)
- 임계값 τ=0.95
- 비라벨 가중치 λ_u=1
- 비라벨:라벨 비율 μ=7
- 배치 크기 B=64(라벨 배치)
- 옵티마이저: SGD(momentum=0.9), cosine LR decay, weight decay, EMA 사용
- 강한 증강: RandAugment 또는 CTAugment + Cutout
- 약한 증강: 수평 플립(대부분 데이터셋에서 50%), ±12.5% 평행이동




1) Task and outputs
- Task: Image classification with semi-supervised learning. Train a classifier using a small labeled set and a large unlabeled set.
- Model output: For input image x, a probability distribution over C classes (e.g., 10 for CIFAR-10, 100 for CIFAR-100), p_m(y|x), and the predicted class argmax_y p_m(y|x).

2) Dataset-wise inputs/outputs (image size, classes, label counts)
- CIFAR-10
  - Input: 32x32 RGB images, 10 classes.
  - Labels (examples): 40 labels (4 per class), 250 labels (25 per class), 4000 labels, etc. The unlabeled pool is the full 50,000 training images (including the labeled ones treated as unlabeled too).
  - Output: One of the 10 classes.
- CIFAR-100
  - Input: 32x32 RGB images, 100 classes.
  - Labels (examples): 400, 2500, 10,000.
  - Output: One of 100 classes.
- SVHN
  - Input: 32x32 RGB (street-view digits), 10 classes (0–9).
  - Labels (examples): 40, 250, 1000.
  - Output: One of 0–9.
- STL-10
  - Input: 96x96 RGB images, 10 classes.
  - Labels: 1,000 per fold (5 predefined folds). 100,000 unlabeled images including some OOD images.
  - Output: One of 10 classes.
- ImageNet (semi-supervised setting)
  - Input: Images resized/cropped to the model’s input size during preprocessing; 1000 classes.
  - Labels: 10% labeled; 90% unlabeled.
  - Output: Top-1 or Top-5 prediction among 1000 classes.

3) Training I/O pipeline (per batch, augmentations, losses)
- Batch composition
  - Labeled batch X = {(x_b, p_b)} of size B; p_b is one-hot label.
  - Unlabeled batch U = {u_b} of size μB (μ is unlabeled:labeled ratio; default μ=7).
  - Example default: B=64, μ=7 → 64 labeled + 448 unlabeled per step.
- Augmentations
  - Weak φ(·): random horizontal flip (50% for most datasets; not used for SVHN) and random translation up to 12.5% horizontally/vertically.
  - Strong A(·): RandAugment or CTAugment followed by Cutout.
- Model I/O and losses
  1) Supervised loss L_s:
     - Input: φ(x_b)
     - Output: p_m(y|φ(x_b))
     - Loss: mean CE(p_b, p_m(y|φ(x_b)))
  2) Unsupervised loss L_u:
     - Input A: φ(u_b) → q_b = p_m(y|φ(u_b))
     - Confidence threshold: if max(q_b) ≥ τ (default τ=0.95), form pseudo-label \hat{q}_b as one-hot(argmax(q_b)); otherwise skip the sample
     - Input B: A(u_b) → p_m(y|A(u_b))
     - Loss: CE(\hat{q}_b, p_m(y|A(u_b))) averaged over thresholded samples
  - Total loss: L = L_s + λ_u L_u (default λ_u=1)
  - Optimizer/schedule: SGD with momentum=0.9, cosine LR decay, weight decay; evaluate with EMA-averaged parameters.

4) Concrete training-step example (with numbers)
- Example: CIFAR-10, 250 labels, μ=7, B=64
  - Supervised:
    - Feed 64 labeled images with weak augmentation φ(x) → get p_m(y|φ(x)) → compute CE with ground truth one-hot labels.
  - Unsupervised:
    - For each of 448 unlabeled images u:
      - Weak pass: φ(u) → q = p_m(y|φ(u))
      - If max(q)=0.97 and the class is “cat” (τ=0.95), set \hat{q} = one-hot(“cat”)
      - Strong pass: A(u) → p_m(y|A(u))
      - Add CE(\hat{q}, p_m(y|A(u))) to L_u
      - If max(q)=0.80 (below τ), skip this sample in L_u
  - Intuition: Train the model so that its prediction on a strongly augmented view matches a high-confidence prediction from a weakly augmented view (consistency + pseudo-labeling).

5) Test-time I/O and evaluation
- Input: Each test image once (standard normalization; no strong augmentation).
- Output: Class probabilities and argmax class.
- Metrics:
  - CIFAR-10/100, SVHN, STL-10: Top-1 accuracy (or error rate).
  - ImageNet (semi-supervised setting): Top-1/Top-5 error rates.
- Note: Final evaluation often uses EMA parameters for stability.

6) Task-specific scenarios
- Task A: CIFAR-10 with 250 labels
  - Training input: 250 labeled (image, one-hot) pairs; rest of the 50,000 used as unlabeled.
  - Unlabeled pipeline: accept only high-confidence pseudo-labels from φ(u), train consistency on A(u).
  - Testing: 10,000 test images → predict top-1 for accuracy.
- Task B: CIFAR-100 with 400 labels
  - Training input: 400 labeled; 49,600 unlabeled.
  - Output space: 100 classes. Same loss/threshold/augmentations; DA can further help.
- Task C: SVHN with 1000 labels
  - Peculiarity: No horizontal flip in weak augmentation for digits.
  - Otherwise same pipeline.
- Task D: STL-10 with 1,000 labels per fold
  - 100,000 unlabeled with some OOD images; strong augmentation and consistency help.
  - WRN-37-2 used in experiments.
- Task E: ImageNet with 10% labels
  - Training input: 10% labeled, 90% unlabeled.
  - Output: 1000-way classification; report Top-1/Top-5 error.
  - FixMatch improves over UDA in this setting.

7) Barely-supervised example (1 label per class)
- CIFAR-10 with only 10 labeled images (one per class), rest unlabeled.
- Same pipeline; results vary widely depending on how “prototypical” those 10 images are (up to ~84% accuracy when picks are representative; can fail with outliers). Highlights sensitivity of pseudo-label quality to initial labeled seed.

8) Key hyperparameters (defaults in paper)
- Confidence threshold τ=0.95
- Unlabeled loss weight λ_u=1
- Unlabeled:labeled ratio μ=7
- Labeled batch size B=64
- Optimizer: SGD (momentum=0.9), cosine LR decay, weight decay, EMA for evaluation
- Strong augmentation: RandAugment or CTAugment + Cutout
- Weak augmentation: horizontal flip (50% for most datasets), ±12.5% translations

<br/>
# 요약
FixMatch는 약한 증강으로 얻은 고신뢰 예측을 원-핫 가짜 라벨로 채택하고, 같은 이미지의 강한 증강에 대해 그 라벨을 맞추도록 교차엔트로피로 학습하는 간단한 결합 방식(일관성 정규화 + pseudo-labeling)이며, 강한 증강으로 RandAugment/CTAugment와 Cutout을 사용한다. 이 방법은 표준 벤치마크에서 SOTA를 기록했으며(CIFAR-10 250라벨 94.93% 정확도, 40라벨 88.61%, ImageNet top-1 error 28.54%로 UDA 대비 2.68%p 향상), CIFAR-100은 DA를 결합하면 추가 개선된다. 예시로, 임계값≈0.95가 가장 효과적이고 강한 증강이 필수이며, 라벨이 극히 적은 경우(클래스당 1개)에도 데이터 선택에 따라 최대 약 84–85%(중앙값 ~64%, 대표성이 높은 샘플만 쓰면 ~78%)까지 도달했다.

FixMatch is a simple combination of consistency regularization and pseudo-labeling: it takes high-confidence predictions from weakly augmented inputs as one-hot pseudo-labels, then trains the model to match them on strongly augmented versions using cross-entropy, with RandAugment/CTAugment plus Cutout for strong augmentation. It achieves state-of-the-art results across benchmarks (e.g., 94.93% on CIFAR-10 with 250 labels, 88.61% with 40 labels, and 28.54% ImageNet top-1 error, improving UDA by 2.68%); on CIFAR-100, adding Distribution Alignment yields further gains. For examples, a confidence threshold around 0.95 works best, strong augmentation is crucial, and in a barely supervised setting with one label per class it reaches up to ~84–85% accuracy (median ~64%, or ~78% when using more prototypical labeled examples).

<br/>
# 기타



- Figure 1 (FixMatch 개요 다이어그램)
  - 결과: 약한 증강으로 얻은 예측에서 최대 확률이 임계값(예: 0.95) 이상일 때 one-hot 가짜 레이블을 만들고, 동일 이미지의 강한 증강 예측이 이 가짜 레이블과 일치하도록 교차엔트로피 손실로 학습.
  - 인사이트: 약한→강한 증강의 “불일치 최소화”가 핵심. 임계값 덕분에 초기에는 미사용(저신뢰) 비라벨 데이터가 많다가, 학습이 진행되며 자연스러운 커리큘럼처럼 점차 더 많은 비라벨 샘플이 학습에 기여.

- Table 1 (SSL 방법 비교표)
  - 결과: 각 방법의 인공 라벨 생성·예측 시 증강 수준과 후처리를 비교. FixMatch는 “약한 증강으로 가짜 라벨 생성 + 강한 증강으로 일치 학습 + 하드한 pseudo-labeling(임계값)” 조합.
  - 인사이트: UDA/ ReMixMatch 등이 쓰는 “샤프닝, 시그널 애닐링, 분포 정렬, 보조 self-supervised 손실” 등을 덜어낸 단순 설계로도 SOTA에 도달.

- Table 2 (주요 벤치마크 정량 결과)
  - 결과: CIFAR-10/100, SVHN, STL-10에서 FixMatch가 대부분 설정에서 SOTA. 예:
    - CIFAR-10, 라벨 40개: 11.39% (CTA), 13.81% (RA)
    - CIFAR-10, 라벨 250개: 5.07% (CTA/RA)
    - CIFAR-10, 라벨 4000개: 4.26–4.31%
    - CIFAR-100, 라벨 400개: 48.85–49.95%
    - CIFAR-100, 라벨 2500/10000개: 28%대/23%대
    - SVHN, 라벨 40/250/1000개: 3.96–7.65%, 2.48–2.64%, 2.28–2.36%
    - STL-10, 라벨 1000개: 5.17–7.98%
  - 인사이트:
    - 단순한 구조로도 MixMatch, UDA, ReMixMatch와 경쟁하거나 능가.
    - CIFAR-100에서는 ReMixMatch가 약간 우수했으나, FixMatch에 Distribution Alignment(DA)를 추가하면 400라벨에서 40.14%까지 향상(> ReMixMatch 44.28%).
    - 강한 희소 라벨(클래스당 4개)에서는 분산이 커지며 RA/CTA 간 차이와 시드 민감도가 증가(보충자료 Table 8 언급).

- Figure 2 (거의 무라벨 학습: 1라벨/클래스)
  - 결과: 클래스당 단 1개 라벨에서도 데이터 선택 품질에 따라 48.6%–85.3% 정확도. “전형적(prototypical)” 예시를 고르면 중앙값 78%까지 도달, 반대로 아웃라이어만 고르면 수렴 실패(10%).
  - 인사이트: 극한 저라벨에서는 “라벨 수”보다 “라벨 품질/대표성”이 성패를 좌우.

- Figure 3 (임계값·샤프닝 어블레이션)
  - 결과:
    - 임계값(τ) ~0.95에서 오류율 최소. 낮은 임계값은 정확도 현저히 하락.
    - 샤프닝(온도 T) 도입해도 임계값 기반 필터링이 있으면 이득이 거의 없음.
  - 인사이트: 비라벨 학습에서 “가짜 라벨의 양보다 질”이 중요. 임계값이 필연적으로 고품질 샘플을 선별하고 자연스러운 커리큘럼을 형성.

- Table 3 (강한 증강 구성: Cutout의 역할)
  - 결과: CTAugment+Cutout 조합이 최적(4.84%). Cutout만 쓰거나(6.15%), Cutout 제거(6.15%) 모두 성능 저하.
  - 인사이트: 강한 정책(CTAugment/RandAugment)과 Cutout의 결합이 중요. 둘 중 하나만으로는 부족.

- 강한/약한 증강 배치에 대한 추가 관찰
  - 결과:
    - 가짜 라벨 추정에 강한 증강을 쓰면 학습이 초기에 발산.
    - 약한 증강 대신 무증강이면 가짜 라벨에 과적합.
    - 학습 경로(하단)에서 강한 증강 대신 약한 증강을 쓰면 일시적으로 45%까지 올랐다가 12%로 붕괴.
  - 인사이트: FixMatch의 “약한(라벨 추정)↔강한(일치 학습)” 구분이 안정성과 성능의 핵심.

- ImageNet (Appendix C 관련)
  - 결과: 10% 라벨, ResNet-50, RA로 Top-1 오류 28.54±0.52%, Top-5 오류 10.87±0.28%. UDA 대비 Top-1 2.68%p 개선. S4L의 전체 파이프라인(추가 단계 포함)보다는 높지만, S4L 1단계 결과(30.27%)는 능가.
  - 인사이트: 대규모·복잡 데이터에서도 단순한 FixMatch가 강력. 추가 단계(재학습/미세조정)를 붙일 여지도 있음.

- 부록(Appendices)에서의 주요 시사점
  - Appendix B (하이퍼파라미터·어블레이션)
    - B.1: 모든 소형 데이터셋에서 동일한 설정 사용(λu=1, lr=0.03, momentum=0.9, τ=0.95, μ=7, B=64, K=220k steps 등). 구현 복잡도·튜닝 부담 감소.
    - B.2: 임계값을 높일수록 가짜 라벨 정확도↑, 사용 샘플 수↓. 성능에는 정확도(질)가 더 중요.
    - B.3: Adam은 성능 저하, SGD(m) 권장.
    - B.4: 코사인 러닝레이트 감쇠가 효과적.
    - B.5: 라벨/비라벨 배치 비율 μ도 중요(본 논문 기본 μ=7).
    - B.6: Weight decay가 특히 저라벨 환경 성능에 결정적.
    - B.7: BSL(1샷/클래스) 분석 보강—대표성 있는 라벨 선택이 성패 좌우.
  - Appendix D (확장)
    - Augmentation Anchoring(강한 증강 다중 샘플), Distribution Alignment(클래스 분포 정렬), MixUp, 적대적 섭동 등과의 결합이 용이. 특히 CIFAR-100에서 DA가 큰 개선.
  - Appendix E (증강 세부)
    - RA/CTA 구체 설정과 Cutout 결합 방식 설명. RA는 전역 크기 범위에서, CTA는 변환별 크기를 온라인으로 학습. 두 정책 모두 강한 변형으로 SSL에 효과적.
  - Supplementary Algorithm 1
    - FixMatch 전체 알고리즘 절차 요약(본문 손실식 구현 가이드).

핵심 총평
- 임계값 기반 하드 pseudo-label + 약·강 증강 분리 = 간단하지만 강력한 조합.
- “가짜 라벨의 질”과 “강한 증강의 적절한 사용”이 성능의 핵심 지렛대.
- 기본 학습 선택(최적화/스케줄/정규화)이 SSL 성능에 매우 민감함을 재확인.





- Figure 1 (FixMatch schematic)
  - Result: Use weak augmentation to obtain a prediction; if its max prob exceeds a threshold (e.g., 0.95), convert to a one-hot pseudo-label, then train the model so its prediction on a strongly augmented view matches the pseudo-label via cross-entropy.
  - Insight: The weak→strong augmentation consistency is crucial. Thresholding creates a natural curriculum—few unlabeled samples are used early, then more as confidence increases.

- Table 1 (Comparison across SSL methods)
  - Result: Compares augmentation for label generation vs. prediction and label post-processing. FixMatch uses weak-for-label, strong-for-prediction, and hard pseudo-labeling with threshold.
  - Insight: Despite omitting sharpening, training-signal annealing, distribution alignment, auxiliary losses, etc., FixMatch reaches SOTA with a simpler recipe.

- Table 2 (Benchmark numbers)
  - Result: FixMatch is SOTA on CIFAR-10, SVHN, STL-10 and competitive on CIFAR-100. Examples:
    - CIFAR-10, 40 labels: 11.39% (CTA), 13.81% (RA)
    - CIFAR-10, 250 labels: 5.07% (CTA/RA)
    - CIFAR-10, 4000 labels: 4.26–4.31%
    - CIFAR-100, 400 labels: 48.85–49.95%
    - CIFAR-100, 2500/10000 labels: ~28%/~23%
    - SVHN, 40/250/1000 labels: 3.96–7.65%, 2.48–2.64%, 2.28–2.36%
    - STL-10, 1000 labels: 5.17–7.98%
  - Insight:
    - Strong or SOTA performance with a simpler pipeline vs. MixMatch/UDA/ReMixMatch.
    - On CIFAR-100, ReMixMatch slightly better; adding Distribution Alignment to FixMatch yields 40.14% at 400 labels (> ReMixMatch’s 44.28%).
    - With extremely few labels (e.g., 4/class), variance grows and sensitivity to seeds increases (see supplementary Table 8).

- Figure 2 (Barely supervised: 1 label/class)
  - Result: With only 1 label per class, accuracy ranges 48.6%–85.3% depending on which examples are chosen; selecting “prototypical” samples yields a 78% median, while outliers fail to train (10%).
  - Insight: In extreme low-label regimes, label quality/representativeness dominates over label quantity.

- Figure 3 (Threshold and sharpening ablation)
  - Result:
    - Best error near threshold ≈ 0.95; low thresholds degrade accuracy significantly.
    - Adding temperature-based sharpening provides little to no benefit when confidence thresholding is used.
  - Insight: Quality over quantity of pseudo-labels is key; thresholding ensures high-quality targets and acts as a curriculum.

- Table 3 (Role of Cutout in strong augmentation)
  - Result: CTAugment+Cutout is best (4.84%); using Cutout alone or removing Cutout both hurt (6.15%).
  - Insight: The combination of a strong policy (CTAugment/RandAugment) with Cutout is important; either alone is insufficient.

- Additional observations on weak/strong placement
  - Result:
    - Using strong augmentation for pseudo-label generation causes divergence.
    - Removing weak augmentation (i.e., no aug) overfits pseudo-labels.
    - Replacing strong with weak augmentation on the training path temporarily rises to ~45% accuracy but collapses to ~12%.
  - Insight: The weak-for-label vs. strong-for-training split is critical for stability and performance.

- ImageNet (Appendix C context)
  - Result: With 10% labels, ResNet-50, RA, FixMatch gets Top-1 28.54±0.52%, Top-5 10.87±0.28%, improving UDA by 2.68% Top-1. It surpasses S4L’s phase-1 (30.27%) but not S4L’s full pipeline (26.79%).
  - Insight: FixMatch scales to large, complex data effectively; additional phases (e.g., re-training/fine-tuning) might yield further gains.

- Appendices: key takeaways
  - Appendix B (Hyperparams and ablations)
    - B.1: Same hyperparameters across small datasets (λu=1, lr=0.03, momentum=0.9, τ=0.95, μ=7, B=64, K=220k), reducing tuning burden.
    - B.2: Higher threshold → higher pseudo-label accuracy but fewer samples; overall accuracy prefers quality over quantity.
    - B.3: Adam underperforms; prefer SGD with momentum.
    - B.4: Cosine LR decay works well.
    - B.5: The unlabeled ratio μ also matters (default μ=7).
    - B.6: Weight decay is especially important in low-label regimes.
    - B.7: Further analysis of 1-shot/class—representative labels are crucial.
  - Appendix D (Extensions)
    - Easy to add Augmentation Anchoring, Distribution Alignment, MixUp, adversarial perturbations; DA notably improves CIFAR-100.
  - Appendix E (Augmentation details)
    - Implementation specifics for RA/CTA and Cutout; RA samples a global magnitude, CTA learns per-transform magnitudes online. Both provide strong, useful distortions for SSL.
  - Supplementary Algorithm 1
    - Full procedural pseudocode for FixMatch aligned with the main loss definitions.

Bottom line
- Hard pseudo-labeling with confidence thresholding plus weak/strong augmentation split is a simple yet powerful recipe.
- Emphasize pseudo-label quality and proper strong augmentation.
- Base training choices (optimizer, schedule, regularization) are highly impactful for SSL outcomes.

<br/>
# refer format:



BibTeX
@inproceedings{Sohn2020FixMatch,
  title        = {FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence},
  author       = {Sohn, Kihyuk and Berthelot, David and Li, Chun-Liang and Zhang, Zizhao and Carlini, Nicholas and Cubuk, Ekin D. and Kurakin, Alex and Zhang, Han and Raffel, Colin},
  booktitle    = {Advances in Neural Information Processing Systems 33 (NeurIPS 2020)},
  year         = {2020},
  address      = {Vancouver, Canada},
  url          = {https://arxiv.org/abs/2001.07685}
}


Sohn, Kihyuk, David Berthelot, Chun-Liang Li, Zizhao Zhang, Nicholas Carlini, Ekin D. Cubuk, Alex Kurakin, Han Zhang, and Colin Raffel. 2020. “FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence.” In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), Vancouver, Canada. https://arxiv.org/abs/2001.07685.
