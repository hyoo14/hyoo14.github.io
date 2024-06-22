---
layout: post
title:  "[2024]KAN: Kolmogorov–Arnold Networks"  
date:   2024-06-21 07:11:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    


Kolmogorov-Arnold 표현 정리에 영감을 받아, 우리는 다층 퍼셉트론(MLP)의 유망한 대안으로 Kolmogorov-Arnold 네트워크(KAN)를 제안합니다. MLP가 노드(“뉴런”)에 고정된 활성화 함수를 가지는 반면, KAN은 엣지(“가중치”)에 학습 가능한 활성화 함수를 가집니다. KAN은 선형 가중치를 전혀 사용하지 않고, 모든 가중치 매개변수를 스플라인으로 매개변수화된 단변수 함수로 대체합니다. 우리는 이러한 간단한 변화가 KAN을 정확도와 해석 가능성 면에서 MLP보다 우수하게 만든다는 것을 보여줍니다. 정확도 면에서, 훨씬 작은 KAN이 데이터 적합과 편미분 방정식(PDE) 해결에서 훨씬 큰 MLP보다 더 높은 정확도를 달성할 수 있습니다. 이론적 및 실험적으로, KAN은 MLP보다 더 빠른 신경 확장 법칙을 가지고 있습니다. 해석 가능성 측면에서, KAN은 직관적으로 시각화할 수 있으며, 인간 사용자가 쉽게 상호작용할 수 있습니다. 수학과 물리학의 두 가지 예를 통해, KAN은 과학자들이 수학적 및 물리적 법칙을 (재)발견하는 데 유용한 "협력자"로 작용할 수 있음을 보여줍니다. 요약하자면, KAN은 MLP의 유망한 대안으로, MLP에 크게 의존하는 오늘날의 딥러닝 모델을 더욱 개선할 수 있는 기회를 제공합니다.


Inspired by the Kolmogorov-Arnold representation theorem, we propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs). While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation functions on edges (“weights”). KANs have no linear weights at all – every weight parameter is replaced by a univariate function parametrized as a spline. We show that this seemingly simple change makes KANs outperform MLPs in terms of accuracy and interpretability. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users. Through two examples in mathematics and physics, KANs are shown to be useful “collaborators” helping scientists (re)discover mathematical and physical laws. In summary, KANs are promising alternatives for MLPs, opening opportunities for further improving today’s deep learning models which rely heavily on MLPs.



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


우리는 Kolmogorov-Arnold 표현 정리에 영감을 받아 Kolmogorov-Arnold 네트워크(KAN)를 다층 퍼셉트론(MLP)의 유망한 대안으로 제안합니다. KAN은 노드가 아닌 엣지에 학습 가능한 활성화 함수를 사용하여 모든 가중치 매개변수를 단변수 함수로 대체합니다. 이를 통해 KAN은 정확도와 해석 가능성 측면에서 MLP를 능가합니다. KAN의 구조는 다양한 간단화 기법을 적용하여 사용자와 상호작용할 수 있도록 설계되었습니다. 이러한 기법에는 가중치 스플라인화, 정규화, 시각화 및 가지치기 등이 포함됩니다. 각 기법은 KAN의 해석 가능성을 높이기 위한 도구로 제공됩니다.

KAN의 주요 구조는 다음과 같습니다:
1. **학습 가능한 활성화 함수**: 모든 가중치 매개변수를 스플라인으로 대체하여 활성화 함수를 학습할 수 있습니다.
2. **스플라인화 및 정규화**: 선형 가중치를 사용하지 않고, 각 가중치 매개변수를 단변수 함수로 매개변수화하여 정규화합니다.
3. **시각화**: 활성화 함수의 크기에 따라 투명도를 조절하여 중요도를 직관적으로 파악할 수 있습니다.
4. **가지치기**: 중요도가 낮은 뉴런을 가지치기하여 네트워크를 간소화합니다.
5. **기호화**: 특정 활성화 함수를 기호적 형태로 설정하여 직관적인 해석을 돕습니다.

KAN은 이론적 및 실험적으로 더 빠른 신경 확장 법칙을 가지고 있으며, 다양한 수학 및 물리학 예제를 통해 KAN이 과학자들이 수학적 및 물리적 법칙을 (재)발견하는 데 유용한 도구임을 보여줍니다.


We propose Kolmogorov-Arnold Networks (KANs) as promising alternatives to Multi-Layer Perceptrons (MLPs), inspired by the Kolmogorov-Arnold representation theorem. Unlike MLPs that use fixed activation functions on nodes, KANs utilize learnable activation functions on edges, replacing all weight parameters with univariate functions. This design allows KANs to outperform MLPs in terms of accuracy and interpretability. The KAN architecture is equipped with various simplification techniques to enhance user interaction, including weight spline parameterization, regularization, visualization, and pruning. Each technique serves as a tool to improve the interpretability of KANs.

The main components of KANs are as follows:
1. **Learnable Activation Functions**: All weight parameters are replaced with splines to enable learnable activation functions.
2. **Spline Parameterization and Regularization**: Linear weights are replaced with univariate functions, regularized to enhance model performance.
3. **Visualization**: Activation functions' transparency is adjusted based on their magnitude, allowing users to intuitively understand the importance of different inputs.
4. **Pruning**: Less important neurons are pruned to simplify the network.
5. **Symbolification**: Specific activation functions can be set to symbolic forms to facilitate intuitive interpretation.

KANs possess faster neural scaling laws than MLPs, both theoretically and empirically, and have shown their utility in helping scientists (re)discover mathematical and physical laws through various examples in mathematics and physics   .




<br/>
# Results  


우리의 연구 결과는 KAN이 MLP를 여러 측면에서 능가한다는 것을 보여줍니다. KAN은 데이터 적합과 편미분 방정식(PDE) 해결에서 더 높은 정확도를 달성하며, 특히 더 작은 네트워크 크기로도 유사한 또는 더 나은 성능을 보입니다. 우리는 다양한 데이터셋을 사용하여 KAN의 성능을 평가했습니다.

1. **Feynman 데이터셋**: Feynman 데이터셋에서 KAN은 MLP보다 적은 매개변수로 더 나은 정확도를 보여주었습니다. 예를 들어, Deepmind의 4-layer, width-300 MLP는 78.0%의 테스트 정확도를 달성했으나, 2-layer, [17, 1, 14] KAN은 81.6%의 정확도를 달성했습니다.

2. **특수 함수 데이터셋**: KAN은 다양한 특수 함수에 대해 MLP보다 더 나은 성능을 보였습니다. 각 데이터셋에 대해, KAN은 더 낮은 RMSE와 더 적은 매개변수로 더 좋은 성능을 보였습니다. 예를 들어, Bessel 함수의 첫 번째 종류의 경우, KAN은 1.64×10^-3의 테스트 RMSE를 달성했으며, 이는 MLP의 5.52×10^-3보다 낮습니다.

3. **매듭 이론**: KAN은 매듭 이론에서 중요한 변수를 식별하고, 인간 과학자들이 유의미한 가설을 도출하는 데 도움을 주었습니다. 예를 들어, KAN은 서명(σ)이 주로 실수 부분 μ_r와 약간의 허수 부분 μ_i 및 종단 거리 λ에 의존함을 발견했습니다. 이는 인간 과학자들이 식별한 결과와 유사합니다.

4. **회피 학습(Continual Learning)**: KAN은 지역적 가소성을 활용하여 파국적인 망각을 방지할 수 있습니다. MLP와 비교하여, KAN은 새로운 데이터가 도입될 때 이전 데이터를 잊지 않고 유지할 수 있습니다.


Our study results show that KANs outperform MLPs in various aspects. KANs achieve higher accuracy in data fitting and PDE solving, especially with smaller network sizes achieving similar or better performance. We evaluated KANs' performance using various datasets.

1. **Feynman Datasets**: On the Feynman datasets, KANs demonstrated better accuracy with fewer parameters compared to MLPs. For instance, Deepmind's 4-layer, width-300 MLP achieved a 78.0% test accuracy, whereas a 2-layer, [17, 1, 14] KAN achieved 81.6% accuracy.

2. **Special Functions Dataset**: KANs performed better than MLPs across various special functions. For each dataset, KANs showed lower RMSE and better performance with fewer parameters. For example, for the Bessel function of the first kind, KAN achieved a test RMSE of 1.64×10^-3, which is lower than MLP's 5.52×10^-3.

3. **Knot Theory**: In knot theory, KANs identified important variables and assisted human scientists in forming meaningful hypotheses. For instance, KANs found that the signature (σ) mostly depends on the real part of the meridinal distance μ_r, slightly on the imaginary part μ_i, and the longitudinal distance λ. This finding aligns with the results identified by human scientists.

4. **Continual Learning**: KANs can avoid catastrophic forgetting by leveraging local plasticity. Compared to MLPs, KANs retain previous data when new data is introduced, demonstrating better performance in continual learning scenarios.



<br/>
# 예시  


KAN이 수학적 문제를 잘 해결한다는 것을 다양한 예제를 통해 입증했습니다. KAN은 복잡한 수학적 함수와 물리학적 법칙을 정확하게 모델링할 수 있으며, 이를 통해 MLP보다 더 높은 정확도와 해석 가능성을 보여줍니다. 다음은 KAN의 성능을 보여주는 몇 가지 예제입니다.

#### 예제 1: 피사체의 위치 예측
**수식:** \( \exp\left(- \frac{\theta^2}{2\sigma^2}\right) / \sqrt{2\pi\sigma^2} \)
**입력 변수:** \(\theta\), \(\sigma\)
**KAN 구조:** [2, 2, 1]
**결과:** 테스트 RMSE 2.86×10^-5

#### 예제 2: 중력 법칙
**수식:** \( G \frac{m_1 m_2}{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2} \)
**입력 변수:** \(a, b, c, d, e, f\)
**KAN 구조:** [6, 4, 1, 1]
**결과:** 테스트 RMSE 8.62×10^-3


KAN has demonstrated its capability in solving mathematical problems effectively through various examples. KAN can accurately model complex mathematical functions and physical laws, showing higher accuracy and interpretability compared to MLPs. Here are some examples that illustrate the performance of KAN.

#### Example 1: Predicting Object Position
**Formula:** \( \exp\left(- \frac{\theta^2}{2\sigma^2}\right) / \sqrt{2\pi\sigma^2} \)
**Input Variables:** \(\theta\), \(\sigma\)
**KAN Structure:** [2, 2, 1]
**Result:** Test RMSE 2.86×10^-5

#### Example 2: Law of Gravity
**Formula:** \( G \frac{m_1 m_2}{(x_2 - x_1)^2 + (y_2 - y_1)^2 + (z_2 - z_1)^2} \)
**Input Variables:** \(a, b, c, d, e, f\)
**KAN Structure:** [6, 4, 1, 1]
**Result:** Test RMSE 8.62×10^-3

<br/>  
# 요약 

Kolmogorov-Arnold 네트워크(KAN)는 다층 퍼셉트론(MLP)에 대한 유망한 대안으로, 학습 가능한 활성화 함수를 사용하여 더 높은 정확도와 해석 가능성을 제공합니다. KAN은 모든 가중치 매개변수를 스플라인 함수로 대체하여 더 작은 네트워크로도 높은 성능을 달성할 수 있습니다. 다양한 데이터셋과 수학적 예제를 통해 KAN이 MLP를 능가하는 성능을 입증했습니다. 특히 Feynman 데이터셋과 특수 함수 데이터셋에서 더 낮은 오류율을 보였습니다. KAN은 복잡한 수학적 문제 해결에 효과적이며, 과학적 발견을 돕는 유용한 도구입니다.


Kolmogorov-Arnold Networks (KAN) are promising alternatives to Multi-Layer Perceptrons (MLP), offering higher accuracy and interpretability by using learnable activation functions. KAN replaces all weight parameters with spline functions, achieving high performance with smaller networks. Extensive evaluations on various datasets and mathematical examples demonstrate KAN's superior performance over MLPs. Specifically, KAN showed lower error rates on the Feynman and special functions datasets. KAN is effective in solving complex mathematical problems and serves as a valuable tool for scientific discovery.

# 기타  



<br/>
# refer format:     
Liu, Ziming, Wang, Yixuan, Vaidya, Sachin, Ruehle, Fabian, Halverson, James, Soljačić, Marin, Hou, Thomas Y., & Tegmark, Max. (2024). KAN: Kolmogorov-Arnold Networks. *Machine Learning (cs.LG); Disordered Systems and Neural Networks (cond-mat.dis-nn); Artificial Intelligence (cs.AI); Machine Learning (stat.ML)*. arXiv:2404.19756. https://doi.org/10.48550/arXiv.2404.19756
  

@article{Liu2024KAN,
  title={KAN: Kolmogorov-Arnold Networks},
  author={Ziming Liu and Yixuan Wang and Sachin Vaidya and Fabian Ruehle and James Halverson and Marin Soljačić and Thomas Y. Hou and Max Tegmark},
  journal={arXiv preprint arXiv:2404.19756},
  year={2024},
  doi={10.48550/arXiv.2404.19756},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
