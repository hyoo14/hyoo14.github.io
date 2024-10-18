---
layout: post
title:  "[2024]Advances in Human Event Modeling From Graph Neural Networks to Language Models"  
date:   2024-10-18 01:51:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    




이 논문은 인간 사건 모델링의 최근 발전을 다루며, 특히 그래프 신경망(GNN)과 대형 언어 모델(LLM)을 이용한 정치적 사건 예측에 초점을 맞추고 있습니다. 인간 사건, 예를 들어 병원 방문, 시위, 전염병 발생 등은 개인과 사회에 큰 영향을 미칩니다. 이러한 사건들은 경제, 정치, 공공 정책 등 다양한 사회적 요인에 의해 영향을 받습니다. 인터넷 상의 다양한 데이터 소스(예: 소셜 네트워크, 뉴스 기사, 개인 블로그)를 통해 이러한 사건들을 기록하고 AI 모델을 개발하는 데 기여하고 있습니다.

이 논문에서는 사건 예측 및 해석을 위한 기존 기술들을 체계적으로 정리하며, 특히 정치적 사건에 대한 예측과 해석에 중점을 둡니다. 그래프 신경망은 관계형 데이터를 다루는 데 강점을 보이며, 언어 모델은 사건 추론에서 유용하게 사용됩니다. 논문은 이 두 가지 기술이 인간 사건 모델링에 미치는 영향과 앞으로의 연구 방향을 논의하고 있습니다.



This paper discusses recent advances in human event modeling, focusing particularly on predicting political events using Graph Neural Networks (GNNs) and Large Language Models (LLMs). Human events, such as hospital visits, protests, and epidemic outbreaks, significantly affect individuals and societies. These events are influenced by various societal factors, such as economics, politics, and public policies. Online data sources like social networks, news articles, and personal blogs chronicle these events, contributing to the development of AI models.

The paper systematically organizes existing technologies for event forecasting and interpretation, emphasizing political event prediction and interpretation. Graph Neural Networks are effective in handling relational data, while language models are useful for event reasoning. The paper discusses how these two technologies influence human event modeling and outlines future research directions.


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



이 논문에서 사용된 주요 방법론은 그래프 신경망(GNN)과 대형 언어 모델(LLM)을 중심으로 한 인간 사건 모델링입니다. 특히, GNN은 사건 간의 관계와 시공간적 의존성을 효과적으로 모델링하기 위해 사용되었으며, 사건의 복잡한 상호작용을 파악하는 데 적합합니다. GNN은 기본적인 그래프 학습, 맥락 정보를 포함한 그래프 학습, 인과 추론을 포함한 그래프 학습의 세 가지 범주로 나뉘어 연구되었습니다. GNN을 활용한 예측 모델은 그래프의 노드와 엣지를 업데이트하고, 시계열 분석을 통해 과거 데이터를 학습하여 미래의 사건을 예측합니다.

대형 언어 모델(LLM)은 대규모 텍스트 데이터를 분석하여 인간 사건 예측에 유용한 정보를 추출하는 데 사용되었습니다. LLM은 뉴스 기사나 소셜 미디어 데이터를 바탕으로 사건 예측에 필요한 텍스트를 생성하거나 질문에 답변하는 방식으로 활용됩니다. 이 논문에서는 GPT-3 같은 LLM을 이용해 사건의 원인을 추론하고, 과거 사건과 패턴을 매칭하여 보다 정확한 예측을 제공하는 방법을 연구했습니다.



The primary methodology used in this paper centers on human event modeling through Graph Neural Networks (GNNs) and Large Language Models (LLMs). Specifically, GNNs are employed to model the relationships and spatiotemporal dependencies between events, making them suitable for capturing complex event interactions. The GNN approaches in the paper are categorized into three groups: vanilla graph learning, graph learning with contextual information, and graph learning with causal reasoning. The predictive models using GNNs update node and edge representations based on the event graph and leverage time-series analysis to learn from past data for forecasting future events.

Large Language Models (LLMs), such as GPT-3, are used to extract useful insights from vast amounts of textual data to aid in event prediction. LLMs generate relevant text based on news articles or social media data, answering questions and reasoning about the causes of events. The paper explores methods where LLMs match causes with past events to provide more accurate predictions.


<br/>
# Results  



이 논문의 결과는 그래프 신경망(GNN)과 대형 언어 모델(LLM)을 사용한 인간 사건 예측 모델의 성능을 평가한 것입니다. 논문에서 제안된 모델은 기존의 방법들과 비교하여 사건 예측의 정확도와 해석 가능성에서 향상된 결과를 보였습니다.

**모델 비교**: 제안된 GNN 및 LLM 기반 모델은 기존의 기계 학습 및 통계적 방법(예: 랜덤 포레스트, 로지스틱 회귀)과 비교되었습니다. 또한, 과거의 사건 데이터를 기반으로 시계열 분석을 수행하는 RNN(순환 신경망) 모델들과도 비교되었습니다. 제안된 GNN 모델은 그래프 구조 내에서 노드와 엣지 간의 상호작용을 더 잘 포착하여 예측 성능을 높였습니다. LLM 모델은 텍스트 데이터의 해석과 사건 추론을 효과적으로 수행할 수 있음을 입증했습니다.

**벤치마크 데이터셋**: 연구에서는 다양한 사건 예측 벤치마크 데이터셋이 사용되었습니다. 특히, 정치적 사건을 포함한 여러 사회적 사건들이 포함된 **GSR 데이터셋**과, **ACLED**, **ICEWS** 같은 데이터셋들이 사용되었습니다. 이러한 데이터셋은 각 사건의 시간, 장소, 유형 등의 정보를 제공하며, 모델이 미래의 사건을 예측하는 데 중요한 역할을 했습니다.

**메트릭**: 성능 평가는 **정확도(Accuracy)**, **F1 점수**, **정밀도(Precision)**, **재현율(Recall)** 등의 지표를 통해 이루어졌습니다. 특히, 여러 사건 예측 모델들 간의 비교를 위해 다중 클래스 분류에서 중요한 **F1 점수**가 중점적으로 사용되었습니다. 제안된 GNN 및 LLM 기반 모델은 대부분의 메트릭에서 기존 모델을 능가하는 성능을 보여주었습니다.




The results of this paper evaluate the performance of human event prediction models using Graph Neural Networks (GNNs) and Large Language Models (LLMs). The proposed models demonstrate improved accuracy and interpretability compared to existing approaches.

**Model Comparison**: The proposed GNN and LLM-based models were compared with traditional machine learning and statistical methods, such as Random Forest and Logistic Regression. Additionally, they were compared with RNN (Recurrent Neural Network) models that perform time-series analysis based on past event data. The GNN models outperformed others by capturing the interactions between nodes and edges within the graph structure more effectively. The LLM models demonstrated strong capabilities in interpreting text data and reasoning about events.

**Benchmark Datasets**: The study used several benchmark datasets for event prediction. Key datasets included the **GSR dataset**, along with others like **ACLED** and **ICEWS**, which provided crucial information about the time, location, and type of events. These datasets played a significant role in enabling the models to predict future events accurately.

**Metrics**: Performance was evaluated using metrics such as **Accuracy**, **F1 Score**, **Precision**, and **Recall**. The **F1 Score**, particularly important for multi-class classification, was a central metric for comparing different event prediction models. The proposed GNN and LLM-based models outperformed previous models across most metrics.



<br/>
# 예시  

 

이 논문에서 사용된 주요 벤치마크 데이터셋은 GSR, ACLED, ICEWS와 같은 정치적, 사회적 사건 데이터셋입니다. 각 데이터셋은 특정 시점과 장소에서 발생한 사건들에 대한 기록을 포함하고 있으며, 제안된 모델과 비교 모델들이 이러한 데이터를 바탕으로 미래 사건을 예측하는 방식으로 평가되었습니다.

예를 들어, **GSR 데이터셋**의 경우, 사회적 불안 사건(예: 시위, 폭동) 예측에서 제안된 **GNN 기반 모델**은 기존의 기계 학습 모델보다 더 높은 정확도를 보였습니다. GNN은 사건들 간의 상호 관계를 효과적으로 모델링했기 때문에, 사회적 불안이 특정 시기에 발생할 가능성이 높다는 점을 잘 예측할 수 있었습니다. 반면에 기존의 **로지스틱 회귀 모델**은 단순한 선형 관계를 사용했기 때문에 사건들 간의 복잡한 상호작용을 제대로 반영하지 못해, 예측 정확도가 낮아지는 경향을 보였습니다.

또한 **ICEWS 데이터셋**에서는 **LLM 기반 모델**이 텍스트 기반 정보를 해석하고, 특정 정치적 사건이 발생할 가능성을 더 잘 예측했습니다. 예를 들어, 뉴스 기사나 소셜 미디어 데이터를 바탕으로 특정 시점에 정치적 불안이 고조되고 있음을 인식하고, 미래에 발생할 사건을 예측하는 데 강점을 보였습니다. 반면, 기존의 **RNN 모델**은 시계열 정보에 집중했지만 텍스트 해석 능력이 부족하여, 사건의 원인이나 결과를 제대로 반영하지 못하는 경우가 있었습니다.

따라서 제안된 GNN 및 LLM 모델은 기존 모델에 비해 복잡한 사건 관계와 텍스트 기반의 정보를 더 잘 처리했으며, 벤치마크 데이터셋에 따라 예측의 정확도가 달랐습니다.


구체적인 예로, **ICEWS 데이터셋**을 사용하여 정치적 사건을 예측할 때, **LLM 기반 모델**은 텍스트 데이터를 통해 특정 국가에서의 정치적 불안이 증가하는 것을 더 잘 감지할 수 있었습니다. 예를 들어, 한 국가에서 다수의 뉴스 기사와 소셜 미디어 게시물에서 "정부에 대한 비판"과 "대규모 시위 계획"에 대한 언급이 급증하는 것을 포착한 모델은, 그 다음 주에 정치적 시위나 폭동이 발생할 가능성이 높다고 예측했습니다. 이때, LLM 모델은 뉴스 기사에서 언급된 '정부 비판'이라는 텍스트적 신호와 소셜 미디어에서의 감정적 반응(예: 분노, 불만)을 분석하여 미래 사건을 정확히 예측했습니다.

반면, 기존의 **RNN 모델**은 단순히 과거 사건들에 기반한 시계열 데이터를 사용하여 예측을 했기 때문에, 이러한 뉴스나 소셜 미디어 상의 미묘한 변화나 감정적 신호를 충분히 반영하지 못했습니다. 결과적으로, 정치적 불안이 고조되는 현상을 놓치거나 정확히 예측하지 못하는 경우가 있었습니다.

또 다른 예로, **ACLED 데이터셋**에서 사회적 불안(예: 시위, 소요)을 예측할 때, **GNN 모델**은 과거 사건들 간의 관계를 분석하여 특정 도시에서 발생한 작은 규모의 시위가 인근 도시로 확산될 가능성이 크다고 예측했습니다. 예를 들어, 한 도시에서 시작된 시위가 주변 지역으로 확산된 과거 패턴을 GNN이 학습했기 때문에, 새롭게 발생하는 작은 시위가 동일한 경로로 확산될 수 있다는 예측을 정확하게 할 수 있었습니다.




The primary benchmark datasets used in this paper were GSR, ACLED, and ICEWS, which record political and social events, including the time and location of each event. The proposed models, as well as comparison models, were evaluated on their ability to predict future events based on these datasets.

For instance, in the **GSR dataset** for social unrest events (such as protests and riots), the proposed **GNN-based model** showed higher accuracy than traditional machine learning models. Since GNN effectively modeled the interactions between events, it could predict the likelihood of social unrest occurring at specific times more accurately. On the other hand, the traditional **logistic regression model**, which relies on simple linear relationships, failed to capture the complex interactions between events, resulting in lower prediction accuracy.

In the **ICEWS dataset**, the **LLM-based model** excelled in interpreting text-based information and predicting the likelihood of specific political events. For example, by analyzing news articles and social media data, the model recognized rising political tensions and predicted future events more effectively. In contrast, the existing **RNN model** focused primarily on time-series information and lacked the capability to interpret text, which sometimes led to an inaccurate representation of event causes or outcomes.

Thus, the proposed GNN and LLM models outperformed traditional models by better handling complex event relationships and text-based data, leading to varying prediction accuracy depending on the benchmark dataset used.






A concrete example comes from the **ICEWS dataset**, where the **LLM-based model** was able to better detect rising political tensions in a specific country by analyzing text data. For instance, when news articles and social media posts from a particular country showed a sudden increase in mentions of "criticism of the government" and "planned mass protests," the model predicted a high likelihood of political protests or riots occurring in the following week. The LLM model captured these textual signals—such as "government criticism" from news articles and emotional reactions (e.g., anger, dissatisfaction) from social media—and used them to accurately forecast future events.

In contrast, the existing **RNN model** relied solely on time-series data from past events and failed to account for subtle shifts in sentiment or emerging signals from news and social media. As a result, it sometimes missed or inaccurately predicted the escalation of political unrest.

Another example is from the **ACLED dataset**, where the **GNN model** predicted social unrest (such as protests or riots) by analyzing the relationships between past events. For instance, the model identified that a small protest in one city had a high likelihood of spreading to nearby cities based on previous patterns of unrest spreading across regions. The GNN model learned from these past event relationships, allowing it to accurately predict that a new small protest might follow a similar trajectory and expand to adjacent areas.



<br/>  
# 요약 




이 논문에서는 **GNN**과 **LLM**을 사용하여 정치적 사건과 사회적 불안 예측을 수행했습니다. **ICEWS** 데이터셋에서 **LLM 기반 모델**은 뉴스 기사와 소셜 미디어 게시물을 분석하여 정부 비판과 같은 신호를 포착해, 미래의 시위 발생을 정확하게 예측했습니다. 반면, **RNN 모델**은 시계열 데이터에만 의존해 이러한 텍스트 신호를 놓쳐 정확도가 낮았습니다. **ACLED** 데이터셋에서는 **GNN 모델**이 과거 사건 간의 관계를 학습해 작은 규모의 시위가 주변 지역으로 확산될 가능성을 더 잘 예측했습니다. 성능 평가는 **정확도(Accuracy)**와 **F1 점수** 등의 메트릭을 사용해 이루어졌으며, GNN과 LLM 모델이 기존 모델보다 높은 성과를 보였습니다.




This paper applied **GNN** and **LLM** models to predict political events and social unrest. In the **ICEWS** dataset, the **LLM-based model** analyzed news articles and social media posts to detect signals like government criticism, accurately predicting future protests. In contrast, the **RNN model** relied solely on time-series data and missed these textual signals, resulting in lower accuracy. In the **ACLED** dataset, the **GNN model** learned from past event relationships and better predicted the likelihood of small protests spreading to neighboring areas. Performance was evaluated using metrics such as **Accuracy** and **F1 Score**, with the GNN and LLM models outperforming traditional models.


# 기타  




<br/>
# refer format:     


@inproceedings{deng2024advances,
  title={Advances in Human Event Modeling: From Graph Neural Networks to Language Models},
  author={Deng, Songgaojun and de Rijke, Maarten and Ning, Yue},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24)},
  pages={11 pages},
  year={2024},
  organization={ACM},
  doi={10.1145/3637528.3671466},
  location={Barcelona, Spain},
  publisher={ACM},
  isbn={979-8-4007-0490-1}
}




Deng, Songgaojun, Maarten de Rijke, and Yue Ning. 2024. "Advances in Human Event Modeling: From Graph Neural Networks to Language Models." Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '24). Barcelona, Spain: ACM. https://doi.org/10.1145/3637528.3671466.