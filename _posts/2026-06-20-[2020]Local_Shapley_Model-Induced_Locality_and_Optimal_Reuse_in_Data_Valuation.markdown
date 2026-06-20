---
layout: post
title:  "[2020]Local Shapley: Model-Induced Locality and Optimal Reuse in Data Valuation"
date:   2026-06-20 14:56:05 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Local Shapley를 통해 데이터 가치 평가의 효율성을 높이는 방법을 제안하며, 모델의 구조적 특성을 활용하여 훈련 데이터의 영향력을 평가하는 새로운 알고리즘 LSMR과 LSMR-A를 개발하였다.  

Contribution allocation do it right?  
올바르게 allocation하는지를 보는?  데이터 기여도 평가? LLM을 위한건가?(그런셈..)  
근데 다 계산 힘드니 로컬리티만 보겠다 같은데..    


짧은 요약(Abstract) :


이 논문에서는 데이터 가치 평가를 위한 원칙적인 기초를 제공하는 샤플리 값(Shapley value)에 대해 다루고 있습니다. 그러나 정확한 계산은 지수적인 조합 공간 때문에 계산 복잡도가 매우 높습니다. 기존의 가속화 방법들은 전역적이며, 특정 테스트 인스턴스에 대해 예측에 영향을 미치는 훈련 포인트의 작은 부분집합만을 고려하는 현대 예측기의 구조적 특성을 고려합니다. 저자들은 이러한 구조적 특성을 '모델 유도 지역성(model-induced locality)'으로 형식화하고, 모델의 계산 경로에 의해 정의된 지원 집합(support sets)을 통해 샤플리 계산을 이러한 지원 집합으로 투영할 수 있음을 보여줍니다. 이로 인해 샤플리 평가가 전통적인 조합 열거 방식이 아닌, 겹치는 지원 집합에 대한 구조적 데이터 처리 문제로 재구성됩니다. 저자들은 LSMR(Local Shapley via Model Reuse)라는 최적의 부분집합 중심 알고리즘을 제안하며, 각 영향력 있는 부분집합을 지원 매핑과 피벗 스케줄링을 통해 정확히 한 번만 훈련하도록 합니다. 또한, LSMR-A라는 재사용 인식 몬테 카를로 추정기를 개발하여, 샤플리 값의 추정 정확도를 유지하면서도 훈련 시간을 줄이는 방법을 제시합니다. 여러 모델 패밀리에 대한 실험을 통해 이 방법이 상당한 재훈련 감소와 속도 향상을 보여주며, 높은 평가 충실도를 유지함을 입증합니다.



This paper addresses the Shapley value, which provides a principled foundation for data valuation. However, exact computation is computationally hard due to the exponential coalition space. Existing accelerations are global and ignore a structural property of modern predictors: for a given test instance, only a small subset of training points influences the prediction. The authors formalize this structural property as "model-induced locality" through support sets defined by the model's computational pathway. They show that Shapley computation can be projected onto these supports without loss when locality is exact. This reframes Shapley evaluation as a structured data processing problem over overlapping support-induced subset families rather than exhaustive coalition enumeration. The authors propose LSMR (Local Shapley via Model Reuse), an optimal subset-centric algorithm that trains each influential subset exactly once via support mapping and pivot scheduling. Additionally, they develop LSMR-A, a reuse-aware Monte Carlo estimator that remains unbiased while achieving substantial retraining reductions and speedups, demonstrating high valuation fidelity across multiple model families through experiments.


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



이 논문에서는 데이터 가치 평가를 위한 새로운 접근 방식인 "Local Shapley"를 제안합니다. 이 방법은 데이터의 기여도를 평가하는 데 있어 모델의 구조적 특성을 활용하여 계산 효율성을 높이는 데 중점을 둡니다. 특히, 데이터 포인트가 특정 테스트 인스턴스에 미치는 영향을 평가할 때, 모든 훈련 데이터 포인트가 아닌, 오직 해당 테스트 인스턴스에 영향을 미치는 소수의 훈련 데이터 포인트만을 고려합니다. 이를 통해 계산 복잡성을 크게 줄일 수 있습니다.

#### 모델 및 아키텍처
1. **K-최근접 이웃 (KNN)**: KNN 모델은 특정 테스트 포인트에 대해 가장 가까운 K개의 훈련 데이터 포인트를 사용하여 예측을 수행합니다. 이 경우, 지원 집합 N(t)는 해당 테스트 포인트의 K개의 최근접 이웃으로 정의됩니다. 이 구조는 데이터 포인트 간의 거리 기반의 근접성을 활용하여 예측을 수행합니다.

2. **결정 트리 (Decision Tree)**: 결정 트리는 입력 데이터의 특성에 따라 데이터를 분할하여 예측을 수행합니다. 각 테스트 포인트는 특정 리프 노드에 도달하며, 지원 집합 N(t)는 해당 리프 노드에 도달하는 훈련 데이터 포인트로 구성됩니다. 이 구조는 규칙 기반의 분할을 통해 데이터의 구조적 특성을 반영합니다.

3. **RBF 커널 서포트 벡터 머신 (RBF Kernel SVM)**: RBF 커널 SVM은 데이터 포인트 간의 거리를 기반으로 예측을 수행합니다. 지원 집합 N(t)는 테스트 포인트와의 커널 값이 특정 임계값을 초과하는 훈련 데이터 포인트로 정의됩니다. 이 구조는 커널 기반의 영향을 통해 데이터의 근접성을 평가합니다.

4. **그래프 신경망 (Graph Neural Network, GNN)**: GNN은 그래프 구조를 기반으로 한 데이터에서 예측을 수행합니다. 각 테스트 포인트의 지원 집합 N(t)는 해당 포인트의 2-홉 이고 네트워크로 정의됩니다. 이 구조는 그래프의 메시지 전달 메커니즘을 통해 데이터 간의 관계를 반영합니다.

#### 훈련 데이터 및 기법
이 논문에서는 다양한 데이터 세트를 사용하여 각 모델의 성능을 평가합니다. 예를 들어, MNIST 데이터 세트는 이미지 기반의 KNN 모델에 사용되며, Iris 데이터 세트는 결정 트리 모델에 사용됩니다. Breast Cancer 데이터 세트는 RBF 커널 SVM에, Cora 데이터 세트는 GNN에 사용됩니다. 각 데이터 세트는 모델의 구조적 특성과 잘 맞아떨어지도록 선택되었습니다.

이러한 모델과 데이터 세트를 통해, Local Shapley는 데이터 가치 평가의 효율성을 높이고, 훈련 데이터의 기여도를 보다 정확하게 평가할 수 있는 방법을 제시합니다.

---




This paper proposes a new approach to data valuation called "Local Shapley." This method focuses on leveraging the structural properties of models to enhance computational efficiency in evaluating the contributions of data points. Specifically, when assessing the impact of data points on a particular test instance, it considers only a small subset of training data points that influence that test instance, rather than all training data points. This significantly reduces computational complexity.

#### Models and Architectures
1. **K-Nearest Neighbors (KNN)**: The KNN model predicts based on the K closest training data points to a specific test point. In this case, the support set N(t) is defined as the K nearest neighbors of the test point. This structure utilizes distance-based proximity between data points for predictions.

2. **Decision Tree**: Decision trees partition data based on the features of the input data to make predictions. Each test point reaches a specific leaf node, and the support set N(t) consists of training data points that pass through the same parent node as the leaf reached by the test point. This structure reflects the structural properties of the data through rule-based partitioning.

3. **RBF Kernel Support Vector Machine (RBF Kernel SVM)**: RBF Kernel SVM makes predictions based on the distances between data points. The support set N(t) is defined as the training data points whose kernel value with the test point exceeds a certain threshold. This structure evaluates proximity through kernel-induced decay.

4. **Graph Neural Network (GNN)**: GNNs make predictions based on graph structures. The support set N(t) consists of nodes within the two-hop ego network of the test point. This structure reflects the relationships between data points through the message-passing mechanism of the graph.

#### Training Data and Techniques
The paper evaluates the performance of each model using diverse datasets. For instance, the MNIST dataset is used for the image-based KNN model, while the Iris dataset is used for the decision tree model. The Breast Cancer dataset is applied to the RBF Kernel SVM, and the Cora dataset is used for the GNN. Each dataset is chosen to align well with the structural properties of the respective model.

Through these models and datasets, Local Shapley presents a method to enhance the efficiency of data valuation and accurately assess the contributions of training data.


<br/>
# Results



이 논문에서는 Local Shapley라는 새로운 데이터 가치 평가 방법을 제안하고, 이를 통해 모델의 구조적 특성을 활용하여 데이터의 기여도를 효율적으로 계산하는 방법을 제시합니다. 실험을 통해 제안된 방법이 기존의 경쟁 모델들과 비교하여 얼마나 효과적인지를 평가하였습니다.

#### 실험 설정
1. **모델 및 데이터셋**: 
   - **WKNN (Weighted K-Nearest Neighbors)**: MNIST 데이터셋을 사용하여 1,000개의 훈련 이미지와 1,000개의 테스트 이미지를 샘플링했습니다. 
   - **Decision Tree**: Iris 데이터셋을 사용하여 105개의 훈련 샘플과 45개의 테스트 샘플을 사용했습니다.
   - **RBF Kernel SVM**: Breast Cancer 데이터셋을 사용하여 398개의 훈련 샘플과 171개의 테스트 샘플을 사용했습니다.
   - **GNN (Graph Neural Network)**: Cora 데이터셋을 사용하여 1,000개의 테스트 노드와 1,708개의 훈련 노드를 사용했습니다.

2. **기존 모델**: 
   - **Global-MC**: 전체 훈련 세트를 대상으로 하는 전통적인 몬테 카를로 샘플링 방법.
   - **Local-MC**: 각 테스트 포인트의 지원 세트(N(t))에 대해 샘플링을 제한하여 계산 비용을 줄이는 방법.
   - **TMC-S**: 조기 중지를 통해 효율성을 높이는 방법.
   - **Comple-S**: 쌍으로 평가하여 샤플리 값을 추정하는 방법.

#### 메트릭
- **정확도 (Accuracy)**: 각 모델이 훈련 데이터의 상위 x%를 선택했을 때의 테스트 정확도를 측정했습니다.
- **재훈련 비용 (Retraining Cost)**: 각 모델이 수렴하는 데 걸리는 총 시간과 모델 훈련 횟수를 측정했습니다.

#### 결과
- **WKNN**: LSMR-A는 Global-MC에 비해 훈련 횟수를 1,126M에서 0.9M으로 줄이며, 속도는 10배 이상 빨라졌습니다.
- **Decision Tree**: LSMR-A는 2.8M에서 0.2M으로 훈련 횟수를 줄였고, 총 실행 시간도 크게 단축되었습니다.
- **RBF Kernel SVM**: LSMR-A는 81.7M에서 0.4M으로 훈련 횟수를 줄였고, 실행 시간도 크게 감소했습니다.
- **GNN**: LSMR-A는 28.2M에서 1.7M으로 훈련 횟수를 줄였고, 실행 시간도 크게 단축되었습니다.

이러한 결과는 LSMR-A가 기존의 방법들에 비해 데이터 가치 평가에서 더 높은 효율성과 정확성을 제공함을 보여줍니다. 특히, LSMR-A는 모델의 구조적 특성을 활용하여 중복 계산을 줄이고, 데이터 선택의 유용성을 높이는 데 기여했습니다.

---



This paper proposes a new data valuation method called Local Shapley, which leverages the structural properties of models to efficiently compute the contributions of data. Experiments were conducted to evaluate how effective the proposed method is compared to existing competitive models.

#### Experimental Setup
1. **Models and Datasets**: 
   - **WKNN (Weighted K-Nearest Neighbors)**: The MNIST dataset was used, sampling 1,000 training images and 1,000 test images.
   - **Decision Tree**: The Iris dataset was used, with 105 training samples and 45 test samples.
   - **RBF Kernel SVM**: The Breast Cancer dataset was used, with 398 training samples and 171 test samples.
   - **GNN (Graph Neural Network)**: The Cora dataset was used, with 1,000 test nodes and 1,708 training nodes.

2. **Existing Models**: 
   - **Global-MC**: The traditional Monte Carlo sampling method that operates over the entire training set.
   - **Local-MC**: A method that restricts sampling to the support set (N(t)) of each test point, reducing computational costs.
   - **TMC-S**: A method that enhances efficiency through early stopping.
   - **Comple-S**: A method that estimates Shapley values via paired evaluations.

#### Metrics
- **Accuracy**: The test accuracy when training on the top x% of data selected by each model.
- **Retraining Cost**: The total running time and number of model trainings required for each model to converge.

#### Results
- **WKNN**: LSMR-A reduced the training count from 1,126M to 0.9M compared to Global-MC, achieving speedups of over 10 times.
- **Decision Tree**: LSMR-A reduced the training count from 2.8M to 0.2M, significantly decreasing total execution time.
- **RBF Kernel SVM**: LSMR-A reduced the training count from 81.7M to 0.4M, with a substantial reduction in execution time.
- **GNN**: LSMR-A reduced the training count from 28.2M to 1.7M, also significantly decreasing execution time.

These results demonstrate that LSMR-A provides higher efficiency and accuracy in data valuation compared to existing methods. In particular, LSMR-A contributes to improving the utility of data selection by reducing redundant computations and leveraging the structural properties of the model.


<br/>
# 예제



이 논문에서는 데이터 가치 평가를 위한 Local Shapley 접근 방식을 제안하고, 이를 다양한 모델과 데이터셋에 적용하여 실험을 수행합니다. 여기서는 각 모델과 데이터셋의 구체적인 인풋과 아웃풋, 그리고 수행하는 작업(Task)에 대해 설명하겠습니다.

#### 1. 모델 및 데이터셋

- **모델**: Weighted K-Nearest Neighbors (WKNN)
  - **데이터셋**: MNIST
  - **인풋**: 1,000개의 훈련 이미지와 1,000개의 테스트 이미지, 각 이미지는 1,024차원의 특징 벡터로 표현됩니다.
  - **아웃풋**: 각 테스트 이미지에 대해 예측된 클래스 레이블 (0-9)과 해당 클래스에 대한 확률 점수.
  - **작업**: 테스트 이미지에 대해 가장 가까운 K개의 이웃을 찾아 그들의 클래스 레이블을 기반으로 최종 예측을 수행합니다.

- **모델**: Decision Tree
  - **데이터셋**: Iris
  - **인풋**: 105개의 훈련 샘플과 45개의 테스트 샘플, 각 샘플은 4개의 연속적인 속성으로 구성됩니다.
  - **아웃풋**: 각 테스트 샘플에 대해 예측된 클래스 레이블 (3개의 클래스 중 하나).
  - **작업**: 훈련된 결정 트리를 사용하여 테스트 샘플의 클래스 레이블을 예측합니다.

- **모델**: RBF Kernel SVM
  - **데이터셋**: Breast Cancer
  - **인풋**: 398개의 훈련 샘플과 171개의 테스트 샘플, 각 샘플은 30개의 진단 특징으로 구성됩니다.
  - **아웃풋**: 각 테스트 샘플에 대해 예측된 클래스 레이블 (양성 또는 악성).
  - **작업**: RBF 커널을 사용하여 훈련된 SVM 모델을 통해 테스트 샘플의 클래스 레이블을 예측합니다.

- **모델**: Graph Neural Network (GNN)
  - **데이터셋**: Cora
  - **인풋**: 1,708개의 노드와 1,000개의 테스트 노드, 각 노드는 1,433차원의 bag-of-words 특징으로 표현됩니다.
  - **아웃풋**: 각 테스트 노드에 대해 예측된 클래스 레이블 (7개의 클래스 중 하나).
  - **작업**: 그래프 구조를 기반으로 메시지 패싱을 통해 각 노드의 임베딩을 업데이트하고, 이를 통해 테스트 노드의 클래스를 예측합니다.

#### 2. 실험 과정

각 모델에 대해 Local Shapley 값을 계산하고, 이를 통해 데이터의 중요성을 평가합니다. 실험은 다음과 같은 단계로 진행됩니다:

1. **데이터 준비**: 각 모델에 맞는 데이터셋을 준비하고, 훈련 및 테스트 세트를 나눕니다.
2. **모델 훈련**: 각 모델을 훈련 데이터로 학습시킵니다.
3. **Local Shapley 값 계산**: 각 테스트 샘플에 대해 Local Shapley 값을 계산하여 해당 샘플의 중요성을 평가합니다.
4. **결과 분석**: Local Shapley 값과 전통적인 Shapley 값 간의 상관관계를 분석하고, 데이터 선택 작업에서의 유용성을 평가합니다.

이러한 실험을 통해 Local Shapley 접근 방식이 데이터 가치 평가에서의 효율성과 정확성을 높일 수 있음을 입증합니다.

---




In this paper, a Local Shapley approach for data valuation is proposed, and experiments are conducted across various models and datasets. Here, we will explain the specific inputs and outputs for each model and dataset, as well as the tasks being performed.

#### 1. Models and Datasets

- **Model**: Weighted K-Nearest Neighbors (WKNN)
  - **Dataset**: MNIST
  - **Input**: 1,000 training images and 1,000 test images, each represented as a 1,024-dimensional feature vector.
  - **Output**: Predicted class labels (0-9) for each test image and the probability scores for those classes.
  - **Task**: For each test image, find the K nearest neighbors and make a final prediction based on their class labels.

- **Model**: Decision Tree
  - **Dataset**: Iris
  - **Input**: 105 training samples and 45 test samples, each sample consisting of 4 continuous attributes.
  - **Output**: Predicted class labels for each test sample (one of three classes).
  - **Task**: Use the trained decision tree to predict the class labels of the test samples.

- **Model**: RBF Kernel SVM
  - **Dataset**: Breast Cancer
  - **Input**: 398 training samples and 171 test samples, each sample consisting of 30 diagnostic features.
  - **Output**: Predicted class labels for each test sample (benign or malignant).
  - **Task**: Predict the class labels of the test samples using the trained SVM model with an RBF kernel.

- **Model**: Graph Neural Network (GNN)
  - **Dataset**: Cora
  - **Input**: 1,708 nodes and 1,000 test nodes, each node represented as a 1,433-dimensional bag-of-words feature.
  - **Output**: Predicted class labels for each test node (one of seven classes).
  - **Task**: Update the embeddings of each node through message passing based on the graph structure and predict the classes of the test nodes.

#### 2. Experimental Process

For each model, Local Shapley values are calculated to evaluate the importance of the data. The experiments are conducted in the following steps:

1. **Data Preparation**: Prepare the dataset suitable for each model and split it into training and testing sets.
2. **Model Training**: Train each model using the training data.
3. **Local Shapley Value Calculation**: Calculate the Local Shapley values for each test sample to assess its importance.
4. **Result Analysis**: Analyze the correlation between Local Shapley values and traditional Shapley values, and evaluate their utility in data selection tasks.

Through these experiments, it is demonstrated that the Local Shapley approach can enhance the efficiency and accuracy of data valuation.

<br/>
# 요약


이 논문에서는 Local Shapley를 통해 데이터 가치 평가의 효율성을 높이는 방법을 제안하며, 모델의 구조적 특성을 활용하여 훈련 데이터의 영향력을 평가하는 새로운 알고리즘 LSMR과 LSMR-A를 개발하였다. 실험 결과, 이 방법은 기존의 글로벌 샤플리 값 계산 방식에 비해 훈련 횟수를 3배 이상 줄이고, 높은 평가 정확도를 유지하는 것으로 나타났다. 다양한 모델과 데이터셋에서 이 접근법이 효과적임을 입증하였다.

---

This paper proposes a method to enhance the efficiency of data valuation through Local Shapley, developing new algorithms LSMR and LSMR-A that leverage the structural properties of models to assess the influence of training data. Experimental results show that this approach reduces the number of training iterations by over three times compared to traditional global Shapley value computation while maintaining high valuation accuracy. The effectiveness of this approach is demonstrated across various models and datasets.

<br/>
# 기타




1. **다이어그램 및 피규어**
   - **Scatter Plots (Figure 2)**: Local Shapley와 Global Shapley 간의 상관관계를 보여주는 산점도는 다양한 모델에서 Local Shapley가 Global Shapley와 강한 양의 상관관계를 가지며, 특히 WKNN 모델에서 가장 높은 상관관계를 보임을 나타냅니다. 이는 Local Shapley가 모델의 주요 영향 경로를 잘 포착하고 있음을 시사합니다.
   - **Data Selection Curves (Figure 3)**: 데이터 선택 성능을 평가한 결과, LSMR-A가 Global-MC보다 적은 샘플로 높은 정확도를 달성함을 보여줍니다. 이는 Local Shapley가 데이터 선택에서 효과적임을 나타내며, 특히 WKNN 모델에서 두드러진 성과를 보입니다.

2. **테이블**
   - **Table 1**: 각 모델에 대한 총 실행 시간과 모델 훈련 횟수를 비교한 결과, LSMR-A가 다른 방법들에 비해 훈련 횟수를 현저히 줄이며 빠른 수렴을 보임을 확인할 수 있습니다. 이는 LSMR-A의 효율성을 강조합니다.
   - **Table 2**: 지원 집합 크기 |N(t)|에 따른 성능 변화를 보여줍니다. 지원 집합 크기가 증가함에 따라 Local Shapley의 정확도가 향상되지만, 훈련 시간은 여전히 LSMR-A가 Global-MC보다 훨씬 빠름을 나타냅니다.

3. **어펜딕스**
   - 어펜딕스에서는 수학적 증명과 관련된 세부 사항이 제공되어, 제안된 알고리즘의 이론적 기초를 강화합니다. 특히, Local Shapley의 성질과 LSMR-A의 최적성에 대한 증명이 포함되어 있어, 이론적 근거를 통해 알고리즘의 신뢰성을 높입니다.




1. **Diagrams and Figures**
   - **Scatter Plots (Figure 2)**: The scatter plots illustrate the correlation between Local Shapley and Global Shapley, showing a strong positive correlation across various models, particularly in WKNN. This indicates that Local Shapley effectively captures the main influence pathways of the model.
   - **Data Selection Curves (Figure 3)**: The results demonstrate that LSMR-A achieves higher accuracy with fewer samples compared to Global-MC, highlighting the effectiveness of Local Shapley in data selection, especially in the WKNN model.

2. **Tables**
   - **Table 1**: The comparison of total runtime and model training counts across models shows that LSMR-A significantly reduces the number of trainings and converges faster than other methods, emphasizing its efficiency.
   - **Table 2**: The results indicate that as the support set size |N(t)| increases, the accuracy of Local Shapley improves, while LSMR-A remains much faster than Global-MC, demonstrating the balance between fidelity and computational efficiency.

3. **Appendix**
   - The appendix provides detailed mathematical proofs and theoretical foundations for the proposed algorithms, reinforcing the reliability of the results. Notably, it includes proofs of the properties of Local Shapley and the optimality of LSMR-A, which enhance the theoretical justification for the algorithms presented in the paper.

<br/>
# refer format:



```bibtex
@article{yang2020local,
  title={Local Shapley: Model-Induced Locality and Optimal Reuse in Data Valuation},
  author={Yang, Xuan and Chen, Hsi-Wen and Chen, Ming-Syan and Pei, Jian},
  journal={Proceedings of the VLDB Endowment},
  volume={14},
  number={1},
  pages={XXX--XXX},
  year={2020},
  publisher={VLDB Endowment},
  doi={XX.XX/XXX.XX}
}
```





Xuan Yang, Hsi-Wen Chen, Ming-Syan Chen, and Jian Pei. "Local Shapley: Model-Induced Locality and Optimal Reuse in Data Valuation." *Proceedings of the VLDB Endowment* 14, no. 1 (2020): XXX-XXX. doi:XX.XX/XXX.XX.
