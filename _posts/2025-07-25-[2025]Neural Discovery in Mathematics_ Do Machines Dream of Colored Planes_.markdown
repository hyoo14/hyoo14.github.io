---
layout: post
title:  "[2025]Neural Discovery in Mathematics: Do Machines Dream of Colored Planes?"  
date:   2025-07-25 14:09:40 +0900
categories: study
---

{% highlight ruby %}


한줄 요약: 

평면색칠 문제를 ml적으로 풀어냄  



짧은 요약(Abstract) :    


이 논문은 신경망이 수학적 발견을 어떻게 도울 수 있는지를 보여주기 위해, 이산기하학과 극값 조합론의 오래된 미해결 문제인 Hadwiger–Nelson 문제를 사례로 제시합니다. 이 문제는 평면을 같은 색의 점 쌍이 단위 거리만큼 떨어지지 않도록 색칠하는 문제입니다. 저자들은 신경망을 함수 근사기로 사용하여 이 문제를 확률적이고 미분 가능한 손실 함수를 가진 최적화 문제로 재구성하였고, 이를 통해 경사 하강법 기반의 탐색을 수행했습니다. 그 결과, 30년 만에 처음으로 기존 문제의 확장판에 대해 두 가지 새로운 6색 칠하기 방식을 발견하는 데 성공했습니다. 이 논문은 그 방법론과 함께 추가적인 수치적 통찰도 제공합니다.



We demonstrate how neural networks can drive mathematical discovery through a case study of the Hadwiger-Nelson problem, a long-standing open problem at the intersection of discrete geometry and extremal combinatorics that is concerned with coloring the plane while avoiding monochromatic unit-distance pairs. Using neural networks as approximators, we reformulate this mixed discrete-continuous geometric coloring problem with hard constraints as an optimization task with a probabilistic, differentiable loss function. This enables gradient-based exploration of admissible configurations that most significantly led to the discovery of two novel six-colorings, providing the first improvement in thirty years to the off-diagonal variant of the original problem (Mundinger et al., 2024a). Here, we establish the underlying machine learning approach used to obtain these results and demonstrate its broader applicability through additional numerical insights.





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



이 논문에서는 Hadwiger-Nelson 문제를 풀기 위해 기존의 이산적인 색칠 문제를 **확률적 색칠 함수(probabilistic coloring function)**로 변환하고, 이를 신경망을 이용한 연속 최적화 문제로 모델링했습니다.

문제의 연속화:
각 평면상의 점에 고정된 색을 부여하는 대신, 각 점에 대해 c개의 색 중 하나일 확률 분포를 할당하는 방식으로 문제를 완화시켰습니다. 이 확률 분포 간의 내적이 같은 색일 확률을 나타내고, 이를 이용해 거리 제약(단위 거리상 같은 색 금지)을 만족하지 못할 확률을 손실 함수로 계산합니다.

신경망 아키텍처:
이 확률 분포를 예측하는 함수는 **Multilayer Perceptron (MLP)**으로 구현되며, 입력은 평면 좌표이고 출력은 색의 확률 분포입니다.

신경망은 24개의 은닉층을 갖고 있으며, 각 층에는 32256개의 뉴런이 존재합니다.

활성화 함수로는 sine 함수를 사용하여 주기적인 패턴 표현력을 높였습니다.

마지막 출력층에는 softmax 함수를 사용해 확률 분포를 생성합니다.

손실 함수 및 학습:
손실 함수는 단위 거리상에서 같은 색일 확률을 통합해 측정하는 방식입니다. 이를 미분 가능하게 설계하여 **경사 하강법(gradient descent)**으로 최적화합니다.
손실의 계산은 몬테카를로 샘플링을 통해 이루어지며, 하나의 학습 단계에서 수천 개의 중심 점과 그 주변 점들을 샘플링하여 손실을 근사합니다.

확장성 및 변형들:
이 방식은 다양한 변형 문제(예: 특정 색이 특정 거리만 피하도록 하는 비대각적 색칠 문제, 삼각형 회피 문제, 고차원 확장 등)에도 적용 가능하도록 설계되어 있으며, 각 경우에 손실 함수의 정의와 샘플링 방식만 조정됩니다.




To tackle the Hadwiger-Nelson problem, the authors reformulate the classical discrete coloring task into a continuous optimization problem using probabilistic colorings, where each point on the plane is assigned a probability distribution over c colors instead of a fixed label.

Problem Relaxation:
The discrete constraint of assigning a single color to each point is relaxed by mapping each point to a probability distribution over the colors. The inner product of the distributions at two unit-distance points represents the probability of a color conflict, which is integrated over the domain to define a differentiable loss function.

Neural Network Architecture:
The mapping from spatial coordinates to color distributions is implemented via a Multilayer Perceptron (MLP):

The network consists of 2 to 4 hidden layers, each with 32 to 256 neurons.

Sine activation functions are used to capture periodic structures.

A softmax layer ensures the output forms a valid probability distribution.

Training with Gradient Descent:
The objective function, representing expected color conflicts, is optimized via gradient descent. The gradient is approximated using Monte Carlo sampling: for each training step, center points and unit-distance neighbors are sampled to compute an estimate of the loss and its gradient.

Flexible Extension to Variants:
The framework is designed to handle variants of the HN problem (e.g., off-diagonal colorings with varying distance constraints, triangle avoidance, and higher-dimensional spaces) by modifying the loss definition and sampling method accordingly.




   
 
<br/>
# Results  



이 논문은 기존의 전통적인 기법들과 직접적으로 "경쟁 모델"을 비교한 것은 아니지만, 수학계에서 오랫동안 유지되어 온 **기존 최적 해(known bounds)**들과 비교하여 신경망 기반 접근이 어떤 발전을 이뤘는지를 정량적으로 보여줍니다. 주요 결과는 다음과 같습니다:

30년 만의 개선: 오프-대각선 6색 칠하기 문제

기존에 Hoffman & Soifer (1996) 등이 제시한 거리 범위는 **[0.415, 0.447]**였음.

본 논문에서는 신경망을 활용해 **[0.354, 0.657]**로 확장함.

이는 "한 색은 거리 d를 피하고 나머지 색은 단위 거리 피함"이라는 설정에서 30년 만의 첫 개선.

거의 5색 칠하기 (Almost Coloring)

평면의 100%를 5색으로 칠하는 것은 불가능하지만, 일정 비율만 제외하면 가능.

기존 최고 기록: Parts (2020)에서 **약 4.01%**의 영역을 제거해야 했음.

본 연구에서는 이 수치를 3.74%로 줄임, 이는 엄밀한 형식화 과정을 거친 결과임.

3차원 공간 칠하기 (Rn으로의 확장)

기존의 알려진 상한은 χ(R³) ≤ 15 (Coulson, 2002).

본 연구는 15색으로 충돌 없이 거의 전체를 칠하는 해를 재현했으며,

나아가 3.46%만 제거하면 되는 14색 칠하기 구성도 제안.

삼각형 회피 색칠 문제 개선

기존 연구(Aichholzer & Perz, 2019)는 특정 조건의 삼각형을 회피하는 데 최소 색 수를 분석.

본 연구는 이 문제에서 3~6색으로 회피 가능한 삼각형 범위를 확장함.

특히 기존 도표가 잘못 축소한 4색 가능 영역도 수정 및 확장.

정량적 평가지표:

주요 메트릭은 **conflict rate (충돌률)**로, 색칠된 평면에서 단위 거리 내 동일 색이 나올 확률을 측정.

1,024~4,096개의 중심 점과 다수의 주변 점을 몬테카를로로 샘플링하여 정밀하게 추정.

최적화는 PyTorch 기반 Adam optimizer 사용.




While the paper does not benchmark against traditional ML models, it quantitatively compares its neural approach with long-standing mathematical bounds, producing several first-of-their-kind improvements in decades:

First Improvement in 30 Years for Off-diagonal Six-Coloring

Previous best distance interval: [0.415, 0.447] (Hoffman & Soifer, 1996).

This work extends it to [0.354, 0.657], allowing more flexible configurations where one color avoids a different distance.

Almost 5-Coloring of the Plane

Prior best: Around 4.01% of the plane had to be removed to achieve a valid 5-coloring (Parts, 2020).

This method reduces the uncovered fraction to 3.74%, verified via a formal discretization and periodic tiling pipeline.

Extension to 3D Space (χ(R³))

Known bound: χ(R³) ≤ 15.

The model reproduces this with near conflict-free results and also finds a 14-coloring covering 96.54% of 3D space.

Avoiding Monochromatic Triangles

Based on Aichholzer & Perz (2019), previous bounds for triangle avoidance used stripe-based colorings.

This work expands the parameter space where 3 to 6 colors suffice, correcting and improving previously underestimated 4-color regions.

Quantitative Evaluation Metrics:

Key metric is conflict rate, measuring the probability that two points at a forbidden distance share the same color.

Monte Carlo sampling of 1,024–4,096 centers and neighbor points per step ensures precise estimation.

Training uses the Adam optimizer in PyTorch, with learning rate schedules and tens of thousands of steps per experiment.









<br/>
# 예제  




이 논문에서의 학습 및 테스트 데이터는 전통적인 CSV나 이미지가 아니라, 다음과 같은 수학적 공간 내의 좌표 기반 샘플입니다.



1. 입력 (Training Input):
입력은 2차원 또는 3차원 좌표값입니다. 예를 들어, (x, y) 또는 (x, y, z) 형태의 실수 좌표입니다.

때때로 입력에는 **거리 정보(d)**가 함께 포함됩니다. 예: 특정 색이 피해야 할 거리값을 조건으로 넣기 위해 (x, y, d)처럼 구성.



2. 출력 (Training Output):
출력은 각 좌표에 대해 c개의 색 중 하나를 선택할 확률 분포입니다.

즉, 출력은 c차원 소프트맥스 분포로, 예를 들어 [0.1, 0.2, 0.7]처럼 표현됩니다. 이건 “이 점이 3번 색일 가능성이 70%”라는 뜻이죠.



3. 테스크 (Task):
주어진 공간(평면 또는 공간) 상에서, 단위 거리 또는 특정 거리만큼 떨어진 두 점이 같은 색이 되지 않도록 색을 할당하는 문제입니다.

문제는 네 가지 변형이 있음:

기본형: 단위 거리의 같은 색 피하기

거의 색칠 (Almost Coloring): 아주 작은 부분을 제외하고 유효한 색칠 만들기

오프-대각선형: 색마다 피해야 할 거리가 다르게 설정됨

삼각형 회피: 특정 변 길이(예: 1, a, b)를 가진 삼각형이 단일 색이 되지 않도록



4. 테스트 과정:
학습된 신경망으로부터 출력된 확률 분포를 argmax하여 실제 색을 할당함 (즉, 가장 확률 높은 색 선택).

그 후 샘플링된 쌍이나 삼각형들에 대해, 금지 거리 조건을 만족하는지 확인.

평가 메트릭은 "충돌률(conflict rate)"로, 잘못 색칠된 점 쌍의 비율입니다.



This paper doesn't use traditional labeled datasets, but instead operates on coordinate-based sampling within mathematical spaces.



1. Input (Training Input):
Inputs are real-valued coordinates from 2D or 3D space, e.g., (x, y) or (x, y, z).

In some task variants, the target distance (d) is also part of the input: e.g., (x, y, d) or even (x, y, d₁, d₂) for multiple constraints.



2. Output (Training Output):
The output is a probability distribution over c colors, represented as a softmax vector.

For example, [0.1, 0.2, 0.7] means the point has a 70% chance of being assigned the third color.



3. Task (Problem Definition):
The model learns to assign colors to points such that no two points at a forbidden distance (typically unit distance) share the same color.

Four task variants are explored:

Original problem: avoid monochromatic unit-distance pairs.

Almost coloring: color all but a small fraction of the plane.

Off-diagonal variant: each color has its own forbidden distance.

Triangle avoidance: prevent triangles with sides (1, a, b) from being monochromatic.



4. Testing Phase:
The trained NN outputs are converted to hard color assignments via argmax.

Pairs or triangles of points are then sampled to verify if the coloring satisfies the geometric constraints.

The key evaluation metric is the conflict rate, i.e., the proportion of constraint-violating pairs or triangles.





<br/>  
# 요약   


이 논문은 평면 색칠 문제를 확률적 색칠 함수와 미분 가능한 손실 함수로 연속 최적화 문제로 변환하고, 이를 좌표 입력 기반 신경망으로 학습하였다. 그 결과, 30년 만에 처음으로 오프-대각선 6색칠 문제에서 거리 범위를 확장하고, 거의 5색칠 문제에서도 기존 대비 더 넓은 영역을 유효하게 색칠하는 성과를 냈다. 입력은 평면상의 좌표와 조건 거리이며, 출력은 각 점에 대한 색 확률 분포이고, 평가 기준은 단위 거리 충돌률(conflict rate)이다.



This paper reformulates the plane coloring problem as a continuous optimization task using probabilistic colorings and differentiable loss, trained via coordinate-based neural networks. As a result, it achieves the first extension in 30 years of the feasible distance range for off-diagonal six-colorings, and improves almost-coloring coverage compared to previous bests. Inputs are spatial coordinates (and sometimes distances), outputs are color probability distributions, and evaluation is based on the conflict rate over unit-distance point pairs.



<br/>  
# 기타  


Figure 1 / 3 (6색 칠하기 결과 시각화)

신경망 출력으로 생성된 색칠 패턴과 이를 수학적으로 형식화한 결과를 시각적으로 비교함.

특정 색(예: 빨간색)이 단위 거리와 다른 거리(예: 0.45)를 회피하는 것을 보여줌.
인사이트: 신경망이 주기적이지 않은 독창적인 패턴도 탐색 가능하며, 이는 새로운 해를 유도함.

Figure 2 (거의 5색 칠하기 구성)

평면에서 96.26%를 5색으로 안전하게 칠하고 나머지 빨간색 부분은 제거해야 함을 시각적으로 보여줌.
인사이트: 이전보다 적은 면적을 희생하고도 유효한 색칠이 가능하다는 증거.

Figure 4 / 8 (삼각형 회피 가능 영역 시각화)

각 삼각형 모양(길이 1, a, b)에서 몇 가지 색으로 회피 가능한지를 색으로 표현한 2D 맵.

Figure 4는 기존(Aichholzer & Perz, 2019)과 본 논문의 개선 영역을 나란히 비교.
인사이트: 이 접근법이 기존의 보수적인 추정보다 훨씬 넓은 해 영역을 발견할 수 있음을 보여줌.

Table 1 (각 색 개수에 따른 Almost Coloring 성능 비교)

기존 문헌과 본 논문이 얻은 면적 제거 비율(%)을 나란히 비교함.
인사이트: 특히 5색과 6색 설정에서 눈에 띄는 개선이 이루어졌음을 수치로 명확히 증명함.

Figure 6 / 7 (거리 변수에 따른 성능 곡선 및 히트맵)

다양한 거리값에 대해 충돌률(conflict rate)이 어떻게 변화하는지를 곡선이나 히트맵으로 시각화함.
인사이트: 신경망은 기존 이론적 경계 근처뿐만 아니라 그 외 구간에서도 의미 있는 가능성을 탐색할 수 있음.

Appendix A–D (시각적 결과 및 수학적 검증 상세)

다양한 색 개수에 따른 실제 색칠 결과, 주기적 타일링 방식, 고차원 확장 결과 등을 포함.
인사이트: 본 논문의 결과가 단지 수치적 근사에 그치지 않고, 형식화 가능한 구조를 띠고 있음을 보여줌.





Figure 1 / 3 (Six-coloring visualizations)

Shows side-by-side comparison of neural network outputs and formalized geometric constructions.

Highlights how one color avoids non-unit distances (e.g., 0.45) while others avoid unit distances.
Insight: Neural networks can suggest novel, asymmetric patterns that lead to previously undiscovered solutions.

Figure 2 (Almost 5-coloring structure)

Visualizes a periodic coloring covering 96.26% of the plane with five colors, removing only the red region.
Insight: Demonstrates improved area efficiency over previous best-known constructions.

Figure 4 / 8 (Triangle-avoidance heatmaps)

Show parameter regions (a, b) where triangle avoidance is possible with 3 to 6 colors.

Figure 4 contrasts prior results with the expanded zones found in this work.
Insight: The method uncovers broader feasible regions than previously thought, correcting earlier underestimations.

Table 1 (Almost-coloring coverage by color count)

Compares plane coverage percentages for k = 1–6 colors against previous works.
Insight: The framework achieves notable improvements in 5- and 6-color settings, verified with formal reconstructions.

Figure 6 / 7 (Conflict rate vs. distance visualizations)

Plots how conflict rate varies with distance parameters (e.g., for off-diagonal coloring types).

Includes pointwise minima and heatmaps over (d₁, d₂) combinations.
Insight: Shows that neural models can explore beyond known safe zones and suggest new valid intervals.

Appendix A–D (Detailed visual outputs and verification procedures)

Include tilings, periodic discretizations, and results in 3D space.
Insight: Confirms that many results are not just numerical but can be formalized and generalized rigorously.

필요하다면 이 피규어들을 정리한 슬라이드용 요약 도표나 시각적 마인드맵도 만들어줄 수 있어!






<br/>
# refer format:     




@inproceedings{mundinger2025neural,
  title     = {Neural Discovery in Mathematics: Do Machines Dream of Colored Planes?},
  author    = {Mundinger, Konrad and Zimmer, Max and Kiem, Aldo and Spiegel, Christoph and Pokutta, Sebastian},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year      = {2025},
  series    = {Proceedings of Machine Learning Research},
  volume    = {267},
  publisher = {PMLR},
  address   = {Vancouver, Canada},
  url       = {https://github.com/ZIB-IOL/neural-discovery-icml25}
}




Mundinger, Konrad, Max Zimmer, Aldo Kiem, Christoph Spiegel, and Sebastian Pokutta. “Neural Discovery in Mathematics: Do Machines Dream of Colored Planes?” In Proceedings of the 42nd International Conference on Machine Learning (ICML), PMLR 267, Vancouver, Canada, 2025. https://github.com/ZIB-IOL/neural-discovery-icml25.




