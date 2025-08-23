---
layout: post
title:  "[2025]Revisiting Semi-Supervised Learning in the Era of Foundation Models"
date:   2025-08-23 11:29:40 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 비전 기초 모델(VFM)과 반지도 학습(SSL)의 상호작용을 탐구하며, 새로운 SSL 벤치마크 데이터셋을 도입하고, 다양한 PEFT(파라미터 효율적 미세 조정) 방법을 활용한 자기 학습 기반의 SSL 방법을 제안이라는데.. 

방법론적으로는.. 그냥 파인튜닝한걸로 레이블링한거 아닌가..? 그 레이블링된걸로 지도학습했다... 이정도 같은디..  


짧은 요약(Abstract) :



이 논문에서는 반지도 학습(Semi-Supervised Learning, SSL)이 풍부한 비지도 데이터와 제한된 지도 데이터를 활용하여 학습을 향상시키는 방법을 다룹니다. 비전 기초 모델(Vision Foundation Models, VFM)이 비전 응용 프로그램의 중추로 점점 더 많이 사용됨에 따라, SSL이 이러한 사전 훈련된 모델과 어떻게 상호작용하는지에 대한 불확실성이 존재합니다. 이를 해결하기 위해, 우리는 동결된 VFM이 성능이 저조한 새로운 SSL 벤치마크 데이터셋을 개발하고 대표적인 SSL 방법들을 체계적으로 평가합니다. 놀랍게도, 지도 데이터만을 사용한 파라미터 효율적인 미세 조정(PEFT)이 비지도 데이터를 활용하지 않고도 SSL 성능에 필적하는 경우가 많다는 것을 관찰했습니다. 이는 우리가 자가 훈련(self-training)이라는 개념적으로 간단한 SSL 기준선을 재검토하도록 유도했습니다. 여기서 우리는 지도 PEFT 모델을 사용하여 비지도 데이터에 대한 의사 레이블을 생성하고 추가 훈련을 진행합니다. 노이즈가 많은 의사 레이블 문제를 극복하기 위해, 우리는 여러 PEFT 접근 방식과 VFM 백본을 앙상블하여 더 강력한 의사 레이블을 생성하는 방법을 제안합니다. 실험 결과는 이 간단하면서도 강력한 접근 방식의 효과를 검증하며, VFM과 함께하는 SSL에 대한 실행 가능한 통찰력을 제공하고, 기초 모델 시대의 더 확장 가능하고 실용적인 반지도 학습을 위한 길을 열어줍니다.

---




This paper addresses how semi-supervised learning (SSL) leverages abundant unlabeled data alongside limited labeled data to enhance learning. As vision foundation models (VFMs) increasingly serve as the backbone of vision applications, there remains uncertainty about how SSL interacts with these pre-trained models. To address this gap, we develop new SSL benchmark datasets where frozen VFMs underperform and systematically evaluate representative SSL methods. We make a surprising observation: parameter-efficient fine-tuning (PEFT) using only labeled data often matches SSL performance, even without leveraging unlabeled data. This motivates us to revisit self-training, a conceptually simple SSL baseline, where we use the supervised PEFT model to pseudo-label unlabeled data for further training. To overcome the notorious issue of noisy pseudo-labels, we propose ensembling multiple PEFT approaches and VFM backbones to produce more robust pseudo-labels. Empirical results validate the effectiveness of this simple yet powerful approach, providing actionable insights into SSL with VFMs and paving the way for more scalable and practical semi-supervised learning in the era of foundation models.


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



이 논문에서는 비전 기초 모델(Vision Foundation Models, VFM)과 반지도 학습(Semi-Supervised Learning, SSL)의 상호작용을 탐구하고, 이를 통해 새로운 SSL 방법론을 제안합니다. 제안된 방법은 다음과 같은 단계로 구성됩니다.

1. **매개변수 효율적 미세 조정(PEFT)**: 먼저, 제한된 레이블 데이터로 VFM을 미세 조정합니다. 이 과정에서 LoRA(Low-Rank Adaptation)와 AdaptFormer와 같은 PEFT 기법을 사용하여 모델의 일부 매개변수만 업데이트합니다. 이는 전체 모델을 미세 조정하는 것보다 더 효율적이며, 레이블 데이터가 부족할 때 성능을 향상시킵니다.

2. **의사 레이블 생성**: 미세 조정된 VFM을 사용하여 레이블이 없는 데이터에 대한 의사 레이블을 생성합니다. 이 단계에서는 모델이 높은 신뢰도로 예측한 클래스에 대해 의사 레이블을 할당합니다.

3. **의사 레이블 앙상블**: 여러 개의 PEFT 방법과 VFM을 사용하여 생성된 의사 레이블을 앙상블합니다. 이 과정에서 각 모델의 예측을 평균하여 더 신뢰할 수 있는 의사 레이블을 생성합니다. 이는 개별 모델의 예측이 서로 다를 수 있기 때문에, 다양한 예측을 결합함으로써 오류를 줄이고 성능을 향상시킵니다.

4. **자기 훈련(Self-Training)**: 최종적으로, 생성된 의사 레이블을 사용하여 모델을 다시 훈련합니다. 이 과정에서 모든 의사 레이블을 사용하여 모델을 업데이트하며, 복잡한 의사 레이블 선택 과정을 생략합니다. 이는 단일 라운드에서 자기 훈련을 완료할 수 있게 하여 효율성을 높입니다.

이러한 방법론은 기존의 SSL 방법보다 더 간단하면서도 효과적인 성능을 보여주며, VFM의 강점을 최대한 활용할 수 있는 기반을 제공합니다. 실험 결과, 제안된 방법은 다양한 데이터셋에서 기존 SSL 방법들보다 우수한 성능을 보였습니다.




This paper explores the interaction between Vision Foundation Models (VFMs) and Semi-Supervised Learning (SSL), proposing a new SSL methodology. The proposed method consists of the following steps:

1. **Parameter-Efficient Fine-Tuning (PEFT)**: Initially, the VFM is fine-tuned using limited labeled data. In this process, techniques such as Low-Rank Adaptation (LoRA) and AdaptFormer are employed to update only a subset of the model's parameters. This approach is more efficient than full model fine-tuning and enhances performance when labeled data is scarce.

2. **Pseudo-Label Generation**: The fine-tuned VFM is used to generate pseudo-labels for unlabeled data. In this step, pseudo-labels are assigned to classes predicted by the model with high confidence.

3. **Pseudo-Label Ensemble**: The pseudo-labels generated from multiple PEFT methods and VFMs are ensembled. By averaging the predictions from different models, more reliable pseudo-labels are created. This is important because individual model predictions can vary, and combining diverse predictions helps reduce errors and improve performance.

4. **Self-Training**: Finally, the model is retrained using the generated pseudo-labels. In this process, all available pseudo-labels are utilized to update the model, eliminating the need for complex pseudo-label selection. This allows self-training to be completed in a single round, enhancing efficiency.

This methodology demonstrates a simpler yet more effective performance compared to existing SSL methods, providing a foundation for maximizing the strengths of VFMs. Empirical results show that the proposed method outperforms existing SSL approaches across various datasets.


<br/>
# Results



이 논문에서는 비전 기초 모델(Vision Foundation Models, VFM)과 반지도 학습(Semi-Supervised Learning, SSL)의 상호작용을 탐구하고, 새로운 SSL 벤치마크 데이터셋을 제안하여 기존의 SSL 방법들이 VFM을 사용할 때 여전히 효과적인지 평가합니다. 연구 결과는 다음과 같습니다.

1. **경쟁 모델**: 연구에서는 FixMatch, FlexMatch, SoftMatch, FineSSL과 같은 여러 SSL 방법을 평가했습니다. 이들 방법은 기존의 SSL 알고리즘으로, 주로 라벨이 없는 데이터에서 신뢰할 수 있는 의사 라벨을 생성하고 선택하는 데 중점을 두고 있습니다.

2. **테스트 데이터**: 연구에서 사용된 데이터셋은 DTD, SUN397, Resisc45, Diabetic Retinopathy, Clevr-C, KITTI 등 다양한 도메인과 과제를 포함하고 있습니다. 각 데이터셋은 서로 다른 수의 라벨이 있는 샘플을 포함하고 있으며, 이는 SSL 방법의 성능을 평가하는 데 중요한 역할을 합니다.

3. **메트릭**: 성능 평가는 Top-1 정확도를 사용하여 이루어졌습니다. 이는 모델이 테스트 데이터에서 올바른 클래스를 예측한 비율을 나타냅니다. 각 SSL 방법의 성능은 라벨이 있는 데이터만을 사용한 경우와 비교하여 평가되었습니다.

4. **비교 결과**: 연구 결과에 따르면, 라벨이 있는 데이터만을 사용한 파라미터 효율적인 미세 조정(PEFT) 방법이 SSL 방법과 유사한 성능을 보였습니다. 특히, PEFT는 라벨이 있는 데이터가 제한적일 때 전체 미세 조정보다 더 나은 성능을 발휘하는 경우가 많았습니다. SSL 방법들은 VFM을 사용할 때 성능이 저하되는 경향이 있었으며, 이는 VFM의 내재된 일반화 능력이 손상될 수 있음을 시사합니다.

5. **제안된 방법**: 연구진은 PEFT와 VFM을 결합한 새로운 자기 학습 기반 SSL 방법을 제안했습니다. 이 방법은 여러 PEFT 접근 방식과 VFM 백본을 앙상블하여 더 강력한 의사 라벨을 생성하고, 이를 통해 모델 성능을 향상시킵니다. 실험 결과, 이 방법은 기존 SSL 방법들보다 우수한 성능을 보였습니다.

결론적으로, 이 연구는 VFM 시대의 SSL에 대한 새로운 통찰을 제공하며, 향후 연구를 위한 강력한 기초를 마련합니다.

---




This paper explores the interaction between Vision Foundation Models (VFMs) and Semi-Supervised Learning (SSL), proposing new SSL benchmark datasets to evaluate whether existing SSL methods remain effective when using VFMs. The findings are as follows:

1. **Competing Models**: The study evaluates several SSL methods, including FixMatch, FlexMatch, SoftMatch, and FineSSL. These methods focus on generating and selecting reliable pseudo-labels from unlabeled data.

2. **Test Data**: The datasets used in the study include DTD, SUN397, Resisc45, Diabetic Retinopathy, Clevr-C, and KITTI, covering a diverse range of domains and tasks. Each dataset contains varying numbers of labeled samples, which play a crucial role in assessing the performance of SSL methods.

3. **Metrics**: Performance evaluation was conducted using Top-1 accuracy, which indicates the proportion of correct class predictions made by the model on the test data. The performance of each SSL method was compared against that of labeled-only fine-tuning.

4. **Comparison Results**: The results indicate that parameter-efficient fine-tuning (PEFT) using only labeled data often matches the performance of SSL methods. Notably, PEFT frequently outperforms full fine-tuning when labeled data is limited. SSL methods tended to underperform when using VFMs, suggesting that allowing SSL to update all parameters in VFMs may inadvertently reduce their built-in generalizability.

5. **Proposed Method**: The authors propose a new self-training-based SSL method that combines PEFT and VFM. This method ensembles multiple PEFT approaches and VFM backbones to produce more robust pseudo-labels, leading to improved model performance. Empirical results show that this approach outperforms existing SSL methods.

In conclusion, this study provides new insights into SSL in the era of VFMs and lays a strong foundation for future research.


<br/>
# 예제



이 논문에서는 반지도 학습(Semi-Supervised Learning, SSL)과 비전 기초 모델(Vision Foundation Models, VFM)의 상호작용을 탐구하기 위해 새로운 SSL 벤치마크 데이터셋을 개발했습니다. 이 데이터셋은 다양한 태스크와 도메인에서의 성능을 평가하기 위해 설계되었습니다. 예를 들어, DTD(Describable Texture Dataset) 데이터셋은 47개의 서로 다른 텍스처 카테고리로 구성된 이미지들을 포함하고 있으며, 이는 텍스처 패턴 인식 태스크에 사용됩니다. 

#### 트레이닝 데이터와 테스트 데이터 예시
- **트레이닝 데이터**: DTD 데이터셋에서 3개의 카테고리(예: "바위", "나무", "천")에 대해 각 카테고리당 6개의 라벨이 있는 이미지를 사용합니다. 총 18개의 라벨이 있는 이미지를 트레이닝 데이터로 사용합니다.
- **테스트 데이터**: DTD 데이터셋에서 1880개의 이미지가 테스트 데이터로 사용됩니다. 이 데이터는 모델의 일반화 성능을 평가하는 데 사용됩니다.

#### 구체적인 태스크
- **태스크**: 주어진 이미지가 어떤 텍스처 카테고리에 속하는지를 분류하는 것입니다. 예를 들어, 모델이 "바위" 텍스처의 이미지를 입력받았을 때, "바위"라는 라벨을 예측해야 합니다.

이와 같은 방식으로, 연구자들은 다양한 SSL 알고리즘을 평가하고, VFM을 활용하여 반지도 학습의 성능을 향상시키기 위한 방법을 제안합니다. 이 과정에서, 라벨이 있는 데이터와 라벨이 없는 데이터를 함께 활용하여 모델의 학습을 최적화합니다.

---




In this paper, the authors explore the interaction between Semi-Supervised Learning (SSL) and Vision Foundation Models (VFM) by developing new SSL benchmark datasets. These datasets are designed to evaluate performance across various tasks and domains. For example, the DTD (Describable Texture Dataset) dataset consists of images from 47 distinct texture categories, which are used for texture pattern recognition tasks.

#### Example of Training and Testing Data
- **Training Data**: In the DTD dataset, 3 categories (e.g., "rock," "wood," "fabric") are selected, and for each category, 6 labeled images are used. This results in a total of 18 labeled images for training.
- **Testing Data**: The DTD dataset includes 1880 images that are used as testing data. This data is utilized to evaluate the generalization performance of the model.

#### Specific Task
- **Task**: The task is to classify the given image into its corresponding texture category. For instance, when the model receives an image of a "rock" texture, it should predict the label "rock."

Through this approach, the researchers evaluate various SSL algorithms and propose methods to enhance the performance of SSL by leveraging VFM. In this process, both labeled and unlabeled data are utilized together to optimize the model's learning.

<br/>
# 요약
이 논문에서는 비전 기초 모델(VFM)과 반지도 학습(SSL)의 상호작용을 탐구하며, 새로운 SSL 벤치마크 데이터셋을 도입하고, 다양한 PEFT(파라미터 효율적 미세 조정) 방법을 활용한 자기 학습 기반의 SSL 방법을 제안합니다. 실험 결과, PEFT와 VFM을 결합한 자기 학습 방법이 기존 SSL 방법보다 우수한 성능을 보였으며, 다양한 데이터셋에서 효과적인 성능 향상을 확인했습니다. 이 연구는 VFM 시대의 반지도 학습을 위한 간단하면서도 강력한 기준선을 설정하고, 향후 연구에 대한 통찰을 제공합니다.

---

This paper explores the interaction between vision foundation models (VFMs) and semi-supervised learning (SSL), introducing new SSL benchmark datasets and proposing a self-training-based SSL method leveraging various parameter-efficient fine-tuning (PEFT) techniques. Experimental results show that the self-training approach combining PEFT and VFMs outperforms existing SSL methods, demonstrating effective performance improvements across diverse datasets. This study establishes a simple yet powerful baseline for SSL in the era of foundation models and provides insights for future research.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: 제안된 SSL 파이프라인을 보여줍니다. 이 파이프라인은 (1) 감독된 파라미터 효율적 미세 조정, (2) 의사 레이블 생성, (3) 의사 레이블 앙상블, (4) 자기 훈련의 네 가지 단계로 구성됩니다. 이 구조는 라벨이 적고 비라벨 데이터가 풍부한 상황에서 효과적으로 학습할 수 있는 방법을 제시합니다.
   - **Figure 2**: 다양한 SSL 방법과 전체 미세 조정의 평균 정확도를 비교합니다. 이 결과는 라벨이 적은 경우에도 전체 미세 조정이 SSL 방법보다 우수한 성능을 보일 수 있음을 보여줍니다.
   - **Figure 4**: 다양한 의사 레이블을 앙상블할 때 성능이 향상되는 것을 보여줍니다. 이는 의사 레이블의 다양성이 모델 성능에 긍정적인 영향을 미친다는 것을 강조합니다.
   - **Figure 6**: PEFT 방법과 VFM을 사용한 앙상블의 성능을 보여줍니다. 다양한 PEFT 및 VFM을 통합할 때 성능이 일관되게 향상되는 것을 확인할 수 있습니다.

2. **테이블**
   - **Table 1**: 새로운 SSL 벤치마크 데이터셋에서의 성능을 보여줍니다. 이 표는 기존 SSL 데이터셋에서의 성능과 비교하여, 새로운 벤치마크에서의 성능이 낮음을 나타내며, 이는 SSL의 필요성을 강조합니다.
   - **Table 3**: 다양한 SSL 방법과 제안된 방법의 성능을 비교합니다. ST(PEFT + VFMs) 방법이 다른 방법들보다 평균적으로 우수한 성능을 보이는 것을 확인할 수 있습니다.
   - **Table 4**: 하이퍼파라미터 튜닝의 효과를 보여줍니다. 제안된 방법이 랜덤 선택 및 단일 기준 선택 방법보다 더 나은 성능을 보이는 것을 확인할 수 있습니다.
   - **Table 9**: 다양한 SSL 방법의 성능을 비교합니다. PEFT가 전체 미세 조정보다 일관되게 우수한 성능을 보이는 것을 확인할 수 있습니다.

3. **어펜딕스**
   - **Appendix A**: 실험에 사용된 데이터셋에 대한 세부 정보를 제공합니다. 다양한 도메인과 작업을 포함하여 SSL의 효과를 평가하는 데 적합한 데이터셋을 선택했습니다.
   - **Appendix B**: 실험에 사용된 하이퍼파라미터 및 계산 설정에 대한 추가 세부 정보를 제공합니다. 이는 실험의 재현성을 높이는 데 기여합니다.
   - **Appendix C**: 제안된 하이퍼파라미터 검색 알고리즘 및 실험에서 고려된 메트릭에 대한 자세한 설명을 제공합니다. 이는 SSL 방법의 성능을 평가하는 데 중요한 역할을 합니다.

---




1. **Diagrams and Figures**
   - **Figure 1**: Illustrates the proposed SSL pipeline, consisting of four phases: (1) Supervised Parameter-Efficient Fine-Tuning, (2) Pseudo-Label Generation, (3) Pseudo-Label Ensemble, and (4) Self-Training. This structure presents a method for effectively learning in scenarios with limited labels and abundant unlabeled data.
   - **Figure 2**: Compares the average accuracy of various SSL methods against full fine-tuning. The results indicate that even with limited labeled data, full fine-tuning can outperform SSL methods.
   - **Figure 4**: Shows the performance improvement when incorporating diverse pseudo-labels in the ensemble. This emphasizes the positive impact of pseudo-label diversity on model performance.
   - **Figure 6**: Displays the performance of ensembles using PEFT methods and VFMs. It confirms that integrating various PEFT and VFM consistently enhances performance.

2. **Tables**
   - **Table 1**: Presents performance metrics on the new SSL benchmark datasets. It indicates that performance is lower compared to existing SSL datasets, highlighting the need for SSL.
   - **Table 3**: Compares the performance of various SSL methods with the proposed method. The ST(PEFT + VFMs) method shows superior average performance compared to others.
   - **Table 4**: Demonstrates the effectiveness of hyperparameter tuning. The proposed method outperforms random selection and single-criterion selection methods.
   - **Table 9**: Compares the performance of various SSL methods. It shows that PEFT consistently outperforms full fine-tuning.

3. **Appendices**
   - **Appendix A**: Provides detailed information about the datasets used in the experiments. It includes a selection of datasets suitable for evaluating the effectiveness of SSL across various domains and tasks.
   - **Appendix B**: Offers additional details on hyperparameters and computational settings used in the experiments, contributing to the reproducibility of the results.
   - **Appendix C**: Contains a detailed description of the proposed hyperparameter search algorithms and metrics considered in the experiments. This plays a crucial role in assessing the performance of SSL methods.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@article{zhang2025revisiting,
  title={Revisiting Semi-Supervised Learning in the Era of Foundation Models},
  author={Zhang, Ping and Mai, Zheda and Nguyen, Quang-Huy and Chao, Wei-Lun},
  journal={arXiv preprint arXiv:2503.09707},
  year={2025}
}
```

### 시카고 스타일 인용
Zhang, Ping, Zheda Mai, Quang-Huy Nguyen, and Wei-Lun Chao. 2025. "Revisiting Semi-Supervised Learning in the Era of Foundation Models." arXiv preprint arXiv:2503.09707.
