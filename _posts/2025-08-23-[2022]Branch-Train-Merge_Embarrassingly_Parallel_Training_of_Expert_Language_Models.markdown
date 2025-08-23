---
layout: post
title:  "[2022]Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models"
date:   2025-08-23 20:25:04 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 

현재 세트에서 분기하여 새로운 도메인에서 추가 훈련을 거친 후, 결과 모델을 세트에 병합하여 향후 사용할 수 있도록해서 추론 시 앙상블하거나 평균화할 수 있습니다  
(Branch-Train-Merge (BTM) 알고리즘은 언어 모델(ELM)을 효율적으로 훈련하기 위함)  


짧은 요약(Abstract) :



Branch-Train-Merge (BTM)은 언어 모델(LM)을 효율적으로 훈련하기 위한 알고리즘입니다. BTM은 독립적인 전문가 언어 모델(ELM)을 학습하며, 각 모델은 과학적 또는 법률 텍스트와 같은 서로 다른 도메인에 특화되어 있습니다. 새로운 ELM은 현재 세트의 ELM에서 분기하여 새로운 도메인에서 추가 훈련을 거친 후, 결과 모델을 세트에 병합하여 향후 사용할 수 있도록 합니다. 이러한 ELM은 추론 시 앙상블하거나 평균화할 수 있습니다. 실험 결과, BTM은 계산 자원이 동일한 GPT 스타일의 트랜스포머 LM과 비교하여 도메인 내 및 도메인 외의 퍼플렉시티를 개선하는 것으로 나타났습니다. 우리의 결과는 극단적인 병렬 처리가 미래의 LM 확장에 효율적으로 사용될 수 있음을 시사합니다.




The abstract describes Branch-Train-Merge (BTM), an algorithm designed for efficient training of language models (LMs). BTM learns a set of independent Expert Language Models (ELMs), each specialized in different domains such as scientific or legal texts. New ELMs are created by branching from existing ELMs, further training on new domains, and then merging the resulting models back into the set for future use. These ELMs can be ensembled or averaged during inference. Experiments show that BTM improves in-domain and out-of-domain perplexities compared to compute-matched GPT-style transformer LMs. The results suggest that extreme parallelism could be used to efficiently scale LMs in future work.


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



Branch-Train-Merge (BTM) 알고리즘은 언어 모델(ELM)을 효율적으로 훈련하기 위한 새로운 방법론입니다. 이 알고리즘은 여러 개의 독립적인 전문가 언어 모델(ELM)을 학습시키며, 각 모델은 과학적 텍스트나 법률 문서와 같은 특정 도메인에 특화되어 있습니다. BTM은 다음과 같은 단계로 구성됩니다:

1. **브랜치(Branch):** 새로운 전문가 모델을 생성하기 위해 현재 존재하는 전문가 모델들의 파라미터를 가중 평균하여 초기화합니다. 이 단계에서는 기존 모델들의 파라미터를 활용하여 새로운 모델을 시작합니다.

2. **훈련(Train):** 새로 생성된 전문가 모델을 특정 도메인의 데이터로 추가 훈련합니다. 이 단계에서는 기존의 전문가 모델들이 훈련 과정에 관여하지 않으며, 새로운 도메인에 대한 전문성을 강화합니다.

3. **병합(Merge):** 훈련이 완료된 새로운 전문가 모델을 기존의 전문가 모델 집합에 병합합니다. 이를 통해 새로운 도메인에 대한 지식을 기존 모델 집합에 통합합니다.

4. **초기화(Initialization):** 첫 번째 전문가 모델을 생성할 때는 랜덤 초기화 대신, 다양한 도메인 데이터를 사용하여 시드 모델을 훈련합니다. 이 시드 모델은 이후 전문가 모델들이 도메인에 특화될 수 있도록 강력한 기초 표현을 제공합니다.

BTM 알고리즘은 각 전문가 모델이 독립적으로 훈련되기 때문에, 대규모 병렬 처리가 가능하며, 이는 언어 모델의 확장성을 높이는 데 기여합니다. 또한, 훈련된 전문가 모델들은 추론 시 앙상블하거나 파라미터를 평균화하여 단일 모델로 통합할 수 있습니다. 이러한 방법은 추론 비용을 일정하게 유지하면서도 성능을 향상시킬 수 있습니다.




The Branch-Train-Merge (BTM) algorithm is a novel methodology for efficiently training language models (ELMs). This algorithm involves training multiple independent Expert Language Models (ELMs), each specialized in a specific domain such as scientific texts or legal documents. The BTM process consists of the following steps:

1. **Branch:** To create a new expert model, initialize it with a weighted average of the parameters from existing expert models. This step leverages the parameters of current models to start the new model.

2. **Train:** Further train the newly created expert model on data from a specific domain. In this step, existing expert models do not participate in the training process, allowing the new model to enhance its expertise in the new domain.

3. **Merge:** After training, integrate the newly trained expert model into the existing set of expert models. This step incorporates the knowledge of the new domain into the existing model set.

4. **Initialization:** When creating the first expert model, instead of random initialization, train a seed model using diverse domain data. This seed model provides strong foundational representations that allow subsequent expert models to specialize in their respective domains.

The BTM algorithm allows for extreme parallelism as each expert model is trained independently, enhancing the scalability of language models. Additionally, the trained expert models can be ensembled or parameter-averaged into a single model during inference, maintaining constant inference costs while improving performance.


<br/>
# Results



이 논문에서는 Branch-Train-Merge (BTM) 알고리즘을 사용하여 ELM FOREST를 훈련한 결과를 제시합니다. ELM FOREST는 여러 개의 전문가 언어 모델(EXPERT LMs)로 구성되어 있으며, 각각은 특정 도메인에 특화되어 있습니다. 실험 결과, ELM FOREST는 계산량이 동일한 GPT 스타일의 Transformer-LM 및 DEMIX 레이어 기반 모델보다 우수한 성능을 보였습니다.

#### 경쟁 모델
- **Transformer-LM**: 전통적인 GPT-3 아키텍처를 기반으로 한 모델로, 분산 데이터 병렬 처리를 사용하여 훈련되었습니다.
- **DEMIX**: Transformer의 피드포워드 레이어를 도메인 전문가로 훈련시킨 모델입니다.

#### 테스트 데이터
- 8개의 다양한 훈련 도메인과 8개의 평가 도메인으로 구성된 데이터셋을 사용했습니다. 이 데이터셋은 주로 영어로 작성된 문서들로 구성되어 있습니다.

#### 메트릭
- **Perplexity (혼란도)**: 언어 모델의 성능을 평가하는 데 사용되는 지표로, 낮을수록 더 나은 성능을 의미합니다.

#### 비교
- **모델 크기**: 125M, 350M, 750M, 1.3B 파라미터 크기의 모델을 비교했습니다.
- **훈련 및 평가 결과**: ELM FOREST는 모든 모델 크기에서 Transformer-LM 및 DEMIX 모델보다 낮은 perplexity를 기록했습니다. 특히, ELM FOREST의 앙상블은 가장 낮은 perplexity를 보여주었습니다.
- **파라미터 평균화**: ELM FOREST의 파라미터를 평균화하여 단일 모델로 결합할 때, 도메인 후행 확률을 사용한 가중 평균이 가장 우수한 성능을 보였습니다. 이는 추가적인 추론 비용 없이도 Transformer-LM보다 성능이 우수했습니다.

#### 결론
ELM FOREST는 BTM 알고리즘을 통해 훈련되었으며, 이는 여러 개의 작은 전문가 모델을 독립적으로 훈련하여 대규모 언어 모델을 효율적으로 확장할 수 있는 가능성을 보여줍니다.

---




In this paper, the results of training an ELM FOREST using the Branch-Train-Merge (BTM) algorithm are presented. The ELM FOREST consists of multiple Expert Language Models (EXPERT LMs), each specialized in a specific domain. Experimental results show that ELM FOREST outperforms compute-matched GPT-style Transformer-LM and DEMIX layer-based models.

#### Competing Models
- **Transformer-LM**: A model based on the traditional GPT-3 architecture, trained using distributed data parallelism.
- **DEMIX**: A model where the feedforward layers of the Transformer are trained as domain experts.

#### Test Data
- The dataset consists of 8 diverse training domains and 8 evaluation domains, primarily composed of English-language documents.

#### Metric
- **Perplexity**: A metric used to evaluate the performance of language models, where lower values indicate better performance.

#### Comparison
- **Model Sizes**: Models with 125M, 350M, 750M, and 1.3B parameters were compared.
- **Training and Evaluation Results**: ELM FOREST recorded lower perplexity than both Transformer-LM and DEMIX models across all model sizes. Notably, the ensemble of ELM FOREST showed the lowest perplexity.
- **Parameter Averaging**: When combining ELM FOREST into a single model through parameter averaging, the weighted average using domain posterior probabilities showed the best performance, outperforming Transformer-LM without additional inference costs.

#### Conclusion
ELM FOREST, trained with the BTM algorithm, demonstrates the potential to efficiently scale large language models by independently training multiple smaller expert models.


<br/>
# 예제

논문 "Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models"에서는 언어 모델(ELM FOREST)을 훈련하기 위한 새로운 알고리즘인 Branch-Train-Merge(BTM)을 소개합니다. 이 알고리즘은 여러 개의 독립적인 전문가 언어 모델(EXPERT LMs)을 훈련하여 다양한 도메인에 특화된 모델을 생성합니다. 예를 들어, 과학 텍스트나 법률 텍스트와 같은 특정 도메인에 특화된 모델을 훈련합니다. 이 모델들은 추론 시 앙상블하거나 평균화하여 하나의 모델로 사용할 수 있습니다.   

### 예시 설명

#### 트레이닝 데이터
- **입력**: 다양한 도메인에서 수집된 텍스트 데이터. 예를 들어, 과학 논문, 법률 문서, 의료 연구 논문 등.
- **출력**: 각 도메인에 특화된 전문가 언어 모델(EXPERT LMs).

#### 테스트 데이터
- **입력**: 훈련에 사용되지 않은 새로운 도메인의 텍스트 데이터.
- **출력**: 해당 도메인에 대한 언어 모델의 퍼플렉시티(perplexity) 점수. 퍼플렉시티는 모델의 예측 성능을 평가하는 지표로, 낮을수록 더 나은 성능을 의미합니다.

#### 구체적인 테스크
1. **도메인 특화 모델 훈련**: 각 도메인에 대해 독립적인 전문가 언어 모델을 훈련합니다. 이 과정은 BTM 알고리즘의 "Branch"와 "Train" 단계에서 이루어집니다.
2. **모델 병합**: 훈련된 전문가 모델들을 "Merge" 단계에서 하나의 ELM FOREST로 병합합니다.
3. **추론**: 새로운 데이터에 대해 ELM FOREST를 사용하여 추론을 수행합니다. 이때, 앙상블 기법이나 파라미터 평균화를 사용하여 모델의 예측 성능을 향상시킵니다.




The paper "Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models" introduces a new algorithm called Branch-Train-Merge (BTM) for training language models (ELM FOREST). This algorithm trains multiple independent expert language models (EXPERT LMs) specialized in different domains, such as scientific or legal texts. These models can be ensembled or averaged into a single model during inference.

### Example Explanation

#### Training Data
- **Input**: Text data collected from various domains, such as scientific papers, legal documents, and medical research papers.
- **Output**: Expert language models (EXPERT LMs) specialized in each domain.

#### Test Data
- **Input**: New text data from domains not used in training.
- **Output**: Perplexity scores of the language model for the given domain. Perplexity is a metric for evaluating the predictive performance of the model, with lower scores indicating better performance.

#### Specific Tasks
1. **Domain-Specialized Model Training**: Train independent expert language models for each domain. This is done in the "Branch" and "Train" steps of the BTM algorithm.
2. **Model Merging**: Merge the trained expert models into a single ELM FOREST in the "Merge" step.
3. **Inference**: Perform inference on new data using the ELM FOREST. Use ensemble techniques or parameter averaging to enhance the model's predictive performance.

<br/>
# 요약


Branch-Train-Merge (BTM) 알고리즘은 독립적으로 훈련된 전문가 언어 모델(ELM)을 사용하여 다양한 도메인에 특화된 언어 모델을 병렬로 학습합니다. 실험 결과, BTM으로 훈련된 ELM FOREST는 GPT 스타일의 변환기 언어 모델보다 더 나은 성능을 보였으며, 특히 도메인 관련성을 고려한 파라미터 평균화가 효과적임을 확인했습니다. 이 방법은 모델의 크기를 확장하면서도 효율적인 학습과 추론을 가능하게 합니다.



The Branch-Train-Merge (BTM) algorithm trains independent Expert Language Models (ELMs) in parallel, each specialized for different domains. Experiments show that ELM FORESTs trained with BTM outperform GPT-style transformer language models, with parameter averaging based on domain relevance proving particularly effective. This approach enables efficient training and inference while scaling the model size.

<br/>
# 기타



1. **Figure 1: Seed Phase의 중요성**
   - **결과**: Seed phase에 할당된 컴퓨팅 자원의 비율에 따라 ELM FOREST의 성능이 어떻게 변하는지를 보여줍니다. 특히, 125M 및 350M 파라미터 모델에서 seed phase가 충분히 이루어지지 않으면 성능이 저하됩니다.
   - **인사이트**: Seed phase는 ELM FOREST의 파라미터 평균화에 매우 중요하며, 특히 작은 모델에서는 seed phase에 충분한 자원을 할당해야 최적의 성능을 발휘할 수 있습니다.

2. **Table 1: ELM FOREST의 성능 비교**
   - **결과**: ELM FOREST는 다양한 모델 크기에서 GPT 스타일의 Transformer-LM 및 DEMIX 모델보다 낮은 perplexity를 보이며 더 나은 성능을 발휘합니다.
   - **인사이트**: BTM 알고리즘을 사용한 ELM FOREST는 동일한 컴퓨팅 자원 하에서 더 나은 성능을 제공하며, 특히 다양한 도메인에 걸쳐 일반화 능력이 뛰어납니다.

3. **Table 2: ELM FOREST의 파라미터 평균화 기법 비교**
   - **결과**: 도메인 후행 확률을 사용한 파라미터 평균화가 가장 좋은 성능을 보이며, 이는 추가적인 추론 비용 없이 Transformer-LM보다 우수한 성능을 제공합니다.
   - **인사이트**: 도메인 후행 확률을 사용한 파라미터 평균화는 자원이 제한된 환경에서 효과적인 추론 기법이 될 수 있습니다.

4. **Appendix Table 7: 데이터 세부사항**
   - **결과**: 다양한 도메인에서의 훈련 및 평가 데이터 세부사항을 제공합니다.
   - **인사이트**: 다양한 도메인에서의 성능 평가를 통해 모델의 일반화 능력을 확인할 수 있습니다.

5. **Appendix Table 8: 훈련 도메인에서의 결과**
   - **결과**: 훈련 도메인에서도 ELM FOREST가 다른 모델들보다 우수한 성능을 보입니다.
   - **인사이트**: ELM FOREST는 훈련 도메인에서도 강력한 성능을 발휘하며, 이는 모델의 도메인 적응 능력을 보여줍니다.




1. **Figure 1: Importance of the Seed Phase**
   - **Results**: Shows how the performance of ELM FOREST varies with the proportion of compute resources allocated to the seed phase. Particularly, in 125M and 350M parameter models, insufficient seed phase leads to degraded performance.
   - **Insights**: The seed phase is crucial for parameter averaging in ELM FOREST, and sufficient resources must be allocated to the seed phase, especially for smaller models, to achieve optimal performance.

2. **Table 1: Performance Comparison of ELM FOREST**
   - **Results**: ELM FOREST demonstrates lower perplexity and better performance than GPT-style Transformer-LM and DEMIX models across various model sizes.
   - **Insights**: ELM FOREST trained with the BTM algorithm provides superior performance under the same computational resources, particularly excelling in generalization across diverse domains.

3. **Table 2: Comparison of Parameter Averaging Techniques for ELM FOREST**
   - **Results**: Parameter averaging using domain posterior weights shows the best performance, offering superior results over Transformer-LM without additional inference costs.
   - **Insights**: Parameter averaging with domain posterior weights can be an effective inference technique in resource-constrained environments.

4. **Appendix Table 7: Data Details**
   - **Results**: Provides details of training and evaluation data across various domains.
   - **Insights**: Performance evaluation across diverse domains helps verify the model's generalization capabilities.

5. **Appendix Table 8: Results on Training Domains**
   - **Results**: ELM FOREST also outperforms other models on training domains.
   - **Insights**: ELM FOREST demonstrates strong performance on training domains, indicating its domain adaptation capabilities.

<br/>
# refer format:



**BibTeX:**
```bibtex
@inproceedings{li2022branch,
  title={Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models},
  author={Li, Margaret and Gururangan, Suchin and Dettmers, Tim and Lewis, Mike and Althoff, Tim and Smith, Noah A. and Zettlemoyer, Luke},
  booktitle={Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS)},
  year={2022},
  organization={NeurIPS}
}
```

**Chicago Style:**
Li, Margaret, Suchin Gururangan, Tim Dettmers, Mike Lewis, Tim Althoff, Noah A. Smith, and Luke Zettlemoyer. 2022. "Branch-Train-Merge: Embarrassingly Parallel Training of Expert Language Models." In *Proceedings of the 36th Conference on Neural Information Processing Systems (NeurIPS)*.
