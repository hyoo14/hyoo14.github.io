---
layout: post
title:  "[2025]Synthetic Data in the Era of LLMs"
date:   2025-08-23 20:49:56 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 대규모 언어 모델(LLM) 시대에 적합한 합성 데이터를 생성하고 활용하는 방법을 다룹니다.

고품질의 합성 데이터를 생성하는 것은 쉽지 않으며, 데이터의 다양성, 정확성, 추론의 질을 보장해야

언어 모델을 사용하여 새로운 데이터를 생성하는 방법입니다. 예를 들어, GPT-3의 인컨텍스트 학습 능력을 활용하여 임의의 태스크에 대한 새로운 예제를 생성, 역번역, 기존 데이터 변환-기존 데이터를 활용하여 원하는 태스크에 맞는 예제로 변환, 인간-인공지능 협업-언어 모델의 창의성과 다양성을 활용하되, 인간이 정확성을 검증하고 개선하는 방식, 상징적 생성-형식 언어를 사용하여 초기 사전 훈련을 수행함으로써 더 빠른 언어 모델 훈련과 일반화를 도모  

(외재평가: 실제 태스크에서 얼마나 도움이 되는지, 내재평가:데이터나 생성 과정의 특성을 평가합니다. 예를 들어, 데이터의 다양성, 정확성 등을 평가)


짧은 요약(Abstract) :



이 논문은 대규모 언어 모델(LLM) 시대에서 합성 데이터의 중요성과 활용에 대해 다루고 있습니다. 자연어 처리(NLP) 모델의 발전은 데이터에 크게 의존하며, 기존의 데이터 수집 방법(인터넷 스크래핑, 수작업 라벨링 등)은 여러 한계가 있습니다. 합성 데이터는 이러한 문제를 해결할 수 있는 잠재력을 가지고 있으며, 상대적으로 깨끗하고, 적절한 크기로 조정 가능하며, 특정 작업에 맞춤화될 수 있습니다. 그러나 고품질의 합성 데이터를 생성하는 것은 쉽지 않으며, 데이터의 다양성, 정확성, 추론의 질을 보장해야 합니다. 이 논문은 합성 데이터의 평가, 생성 방법, 활용 방안, 그리고 그 한계와 열린 질문들에 대해 논의합니다.




This paper discusses the significance and utilization of synthetic data in the era of large language models (LLMs). The advancement of natural language processing (NLP) models heavily relies on data, and traditional data collection methods (such as internet scraping and manual labeling) have several limitations. Synthetic data has the potential to address these issues by being relatively clean, appropriately sized, and tailored to specific tasks. However, generating high-quality synthetic data is challenging, as it requires ensuring diversity, accuracy, and quality of reasoning. This paper explores the evaluation, creation methods, usage strategies, and the limitations and open questions surrounding synthetic data.


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

 




샘플링기반생성-이 논문에서는 대규모 언어 모델(LLM) 시대에 합성 데이터를 생성하고 활용하는 다양한 방법론을 다루고 있습니다. 합성 데이터는 자연어 처리(NLP) 모델의 성능을 향상시키기 위해 사용되며, 특히 데이터가 부족하거나 특정한 태스크에 맞춘 데이터를 필요로 할 때 유용합니다. 

1. **합성 데이터 생성 방법**:
   - **샘플링 기반 생성**: 언어 모델을 사용하여 새로운 데이터를 생성하는 방법입니다. 예를 들어, GPT-3의 인컨텍스트 학습 능력을 활용하여 임의의 태스크에 대한 새로운 예제를 생성할 수 있습니다.
   - **역번역(Back-translation)**: 주어진 출력에 대해 해당 출력을 생성할 수 있는 입력을 생성하는 방법입니다. 이는 기계 번역에서 자주 사용되는 기법으로, 자연스러운 출력을 보장하기 위해 입력이 다소 비자연스러울 수 있습니다.
   - **기존 데이터 변환**: 기존 데이터를 활용하여 원하는 태스크에 맞는 예제로 변환하는 방법입니다. 예를 들어, 관련 데이터셋을 검색하고 이를 원하는 태스크에 맞게 변환합니다.
   - **인간-인공지능 협업**: 언어 모델의 창의성과 다양성을 활용하되, 인간이 정확성을 검증하고 개선하는 방식입니다. 이는 데이터 생성의 효율성을 높일 수 있습니다.
   - **상징적 생성**: 형식 언어를 사용하여 초기 사전 훈련을 수행함으로써 더 빠른 언어 모델 훈련과 일반화를 도모하는 방법입니다.

2. **합성 데이터의 활용**:
   - **기본 언어 모델링 알고리즘 지원**: 합성 데이터는 사전 훈련, 지도 학습 미세 조정, 강화 학습 등 다양한 단계에서 활용될 수 있습니다.
   - **시나리오별, 최종 사용자 애플리케이션 지원**: 특정 시나리오나 애플리케이션에 맞춘 데이터 생성 및 활용이 가능합니다.

3. **데이터 품질 평가**:
   - **외재적 평가**: 합성 데이터가 실제 태스크에서 얼마나 도움이 되는지를 평가합니다.
   - **내재적 평가**: 데이터나 생성 과정의 특성을 평가합니다. 예를 들어, 데이터의 다양성, 정확성 등을 평가합니다.







The paper discusses various methodologies for generating and utilizing synthetic data in the era of large language models (LLMs). Synthetic data is used to enhance the performance of natural language processing (NLP) models, especially when there is a lack of data or when data tailored to specific tasks is needed.

1. **Methods for Generating Synthetic Data**:
   - **Sampling-based Generation**: This involves generating new data using language models. For instance, leveraging GPT-3's in-context learning ability to generate new examples for arbitrary tasks.
   - **Back-translation**: This method involves generating an input for a given output, ensuring that the outputs are natural even if the inputs might be somewhat unnatural. This is commonly used in machine translation.
   - **Transformation of Existing Data**: This involves using existing data and transforming it into examples suitable for the desired task. For example, retrieving relevant datasets and transforming them into task-specific data.
   - **Human-AI Collaboration**: This method combines the creativity and diversity of language models with human verification and improvement of correctness, enhancing the efficiency of data generation.
   - **Symbolic Generation**: This involves pretraining on formal languages to achieve faster language model training and better generalization.

2. **Utilization of Synthetic Data**:
   - **Supporting Fundamental Language Modeling Algorithms**: Synthetic data can be used in various stages such as pretraining, supervised fine-tuning, and reinforcement learning.
   - **Supporting Scenario-specific, End-user Applications**: Data can be generated and utilized for specific scenarios or applications.

3. **Evaluation of Data Quality**:
   - **Extrinsic Evaluation**: This assesses how helpful the synthetic data is in actual tasks.
   - **Intrinsic Evaluation**: This evaluates the characteristics of the data or the generation process, such as diversity and correctness.


<br/>
# Results


논문에서 제시된 결과는 다양한 측면에서 분석되었습니다. 먼저, 경쟁 모델과의 비교를 통해 제안된 방법의 우수성을 입증하고자 하였습니다. 테스트 데이터는 다양한 시나리오와 도메인을 포함하여 모델의 일반화 능력을 평가할 수 있도록 구성되었습니다. 메트릭으로는 정확도, 다양성, 커버리지, 프라이버시, 공정성 등이 사용되었습니다. 이러한 메트릭을 통해 생성된 데이터의 품질을 다각도로 평가하였습니다.

비교 실험에서는 제안된 방법이 기존의 데이터 생성 방법들에 비해 더 높은 정확도와 다양성을 보였으며, 특히 특정 도메인에 맞춘 데이터 생성에서 우수한 성능을 나타냈습니다. 또한, 프라이버시와 공정성 측면에서도 긍정적인 결과를 얻었습니다. 이러한 결과는 제안된 방법이 다양한 응용 분야에서 활용될 수 있음을 시사합니다.

---

In the paper, the results were analyzed from various perspectives. First, the superiority of the proposed method was demonstrated through comparisons with competing models. The test data was composed to evaluate the model's generalization ability, including various scenarios and domains. Metrics such as accuracy, diversity, coverage, privacy, and fairness were used. These metrics were used to evaluate the quality of the generated data from multiple angles.

In comparative experiments, the proposed method showed higher accuracy and diversity compared to existing data generation methods, especially demonstrating excellent performance in generating data tailored to specific domains. Additionally, positive results were obtained in terms of privacy and fairness. These results suggest that the proposed method can be utilized in various application fields.


<br/>
# 예제



#### 트레이닝 데이터
- **인풋**: 주어진 문장이나 텍스트 조각. 예를 들어, "한 남자가 플루트를 연주하고 있다."
- **아웃풋**: 주어진 인풋에 대한 유사한 문장 생성. 예를 들어, "그는 플루트를 연주하고 있다."

#### 테스트 데이터
- **인풋**: 특정 작업에 대한 지시사항. 예를 들어, "주소와 도시가 주어졌을 때, 우편번호를 찾아라."
- **아웃풋**: 지시사항에 따른 결과. 예를 들어, "123 메인 스트리트, 샌프란시스코"라는 인풋에 대해 "94105"라는 아웃풋을 생성.

#### 구체적인 테스크
1. **문장 유사성 생성**: 모델이 주어진 문장과 유사한 문장을 생성하도록 훈련.
2. **지시사항 생성**: 주어진 작업에 대한 새로운 지시사항을 생성. 예를 들어, "명상에 대한 에세이를 작성하라"는 지시사항에 대해 적절한 응답을 생성.
3. **데이터 변환**: 기존 데이터를 변환하여 새로운 작업에 맞는 예시로 활용. 예를 들어, 기존의 대화 데이터를 활용하여 새로운 대화 예시를 생성.




#### Training Data
- **Input**: Given sentences or text snippets. For example, "A man is playing the flute."
- **Output**: Generate a similar sentence to the given input. For example, "He is playing the flute."

#### Test Data
- **Input**: Instructions for a specific task. For example, "Given an address and city, find the zip code."
- **Output**: Result according to the instructions. For example, for the input "123 Main Street, San Francisco," the output would be "94105."

#### Specific Tasks
1. **Sentence Similarity Generation**: Train the model to generate sentences similar to a given sentence.
2. **Instruction Generation**: Generate new instructions for a given task. For example, generate an appropriate response to the instruction "Write an essay about the benefits of meditation."
3. **Data Transformation**: Transform existing data into examples suitable for a new task. For example, use existing dialogue data to create new dialogue examples.

<br/>
# 요약

이 논문에서는 대규모 언어 모델(LLM) 시대에 적합한 합성 데이터를 생성하고 활용하는 방법을 다룹니다. 합성 데이터 생성 방법으로는 샘플링 기반 생성, 역번역, 기존 데이터 변환, 인간-인공지능 협업, 상징적 생성 등이 있으며, 이러한 데이터는 언어 모델의 사전 학습, 지도 학습 미세 조정, 강화 학습 등 다양한 알고리즘에 활용됩니다. 예를 들어, Self-Instruct 방법은 LLM이 자체적으로 생성한 지시문을 사용하여 모델을 미세 조정함으로써, 기존의 인스트럭션 데이터셋과 유사한 성능을 달성합니다.

In this paper, methods for generating and utilizing synthetic data suitable for the era of large language models (LLMs) are discussed. Methods for generating synthetic data include sampling-based generation, back-translation, transformation of existing data, human-AI collaboration, and symbolic generation, which are used in various algorithms such as pretraining, supervised fine-tuning, and reinforcement learning of language models. For instance, the Self-Instruct method fine-tunes models using instructions generated by the LLM itself, achieving performance comparable to existing instruction datasets.

<br/>
# 기타



### 결과와 인사이트

1. **Synthetic Data의 필요성**:
   - 인터넷 스크래핑, 수작업 라벨링, 시스템 사용자로부터의 데이터 수집, 창의적인 큐레이션 등 기존의 데이터 수집 방법은 노이즈가 많거나 비용이 많이 들고, 적용 가능성이 제한적입니다. 이러한 문제를 해결하기 위해 상대적으로 깨끗하고, 적절한 크기로 개별 작업에 맞춰진 유연한 Synthetic Data가 필요합니다.

2. **Synthetic Data의 생성 방법**:
   - 샘플링 기반 생성, 역번역, 기존 데이터의 변환, 인간-인공지능 협업, 상징적 생성 등 다양한 방법이 있습니다. 각 방법은 특정한 장점과 단점을 가지고 있으며, 상황에 따라 적절한 방법을 선택해야 합니다.

3. **Synthetic Data의 평가**:
   - 데이터 품질 평가는 외재적 평가(다운스트림 작업에 도움이 되는지)와 내재적 평가(데이터 또는 생성 과정의 특성)로 나뉩니다. 데이터의 정확성, 다양성, 커버리지 등을 평가하는 다양한 메트릭이 존재합니다.

4. **Synthetic Data의 활용**:
   - 기본적인 언어 모델링 알고리즘 지원 및 시나리오별, 최종 사용자 애플리케이션 지원에 사용됩니다. 예를 들어, 사전 훈련, 지도 학습 미세 조정, 강화 학습 등 다양한 단계에서 활용됩니다.

5. **제한 사항 및 열린 질문**:
   - Synthetic Data 생성의 어려움, 데이터의 정확성 문제, 생성된 데이터의 다양성 부족 등이 주요 제한 사항으로 지적됩니다. 이러한 문제를 해결하기 위한 연구가 필요합니다.




### Results and Insights

1. **Need for Synthetic Data**:
   - Traditional data collection methods such as internet scraping, manual labeling, collecting from system users, and creative curation are often noisy, expensive, and limited in applicability. Synthetic data, which is relatively clean, appropriately sized, tailored to individual tasks, and flexible, is needed to address these issues.

2. **Methods of Creating Synthetic Data**:
   - Various methods include sampling-based generation, back-translation, transformation of existing data, human-AI collaboration, and symbolic generation. Each method has its advantages and disadvantages, and the appropriate method should be chosen based on the context.

3. **Evaluation of Synthetic Data**:
   - Data quality evaluation is divided into extrinsic evaluation (whether it helps in a downstream task) and intrinsic evaluation (characteristics of the data or generation process). Various metrics exist to evaluate data correctness, diversity, and coverage.

4. **Use of Synthetic Data**:
   - It is used to support fundamental language modeling algorithms and scenario-specific, end-user applications. For example, it is utilized in pretraining, supervised fine-tuning, and reinforcement learning.

5. **Limitations and Open Questions**:
   - Challenges in generating synthetic data, issues with data correctness, and lack of diversity in generated data are major limitations. Further research is needed to address these challenges.




<br/>
# refer format:



**BibTeX:**
```bibtex
@misc{viswanathan2025synthetic,
  title={Synthetic Data in the Era of LLMs},
  author={Vijay Viswanathan and Xiang Yue and Alisa Liu and Yizhong Wang and Graham Neubig},
  year={2025},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL)},
  howpublished={\url{https://synth-data-acl.github.io/}}
}
```

**Chicago Style:**
Vijay Viswanathan, Xiang Yue, Alisa Liu, Yizhong Wang, and Graham Neubig. 2025. "Synthetic Data in the Era of LLMs." Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (ACL). https://synth-data-acl.github.io/.

