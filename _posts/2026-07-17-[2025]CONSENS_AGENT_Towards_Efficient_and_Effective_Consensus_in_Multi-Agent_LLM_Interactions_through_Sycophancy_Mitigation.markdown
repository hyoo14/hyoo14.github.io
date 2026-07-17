---
layout: post
title:  "[2025]CONSENS AGENT: Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions through Sycophancy Mitigation"
date:   2026-07-17 02:59:17 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: CONSENS AGENT는 다중 에이전트 대화 시스템에서 효율적이고 효과적인 합의를 도출하기 위해 설계된 새로운 프레임워크    


구성: 초기 응답 생성, 다중 에이전트 토론, 프롬프트 최적화(이전 상호작용을 기반으로 프롬프트를 동적으로 수정하여 에이전트 간의 동조 현상(즉, sycophancy)을 완화), , 팀 응답 생성(최종적으로 각 에이전트의 신뢰도와 일관성을 고려하여 팀의 최종 응답을 생성)     

sycophancy를 줄이기 위해 트리거 메커니즘을 사용(이 메커니즘은 에이전트가 서로의 응답을 무비판적으로 복사하는 경우를 감지하고, 이를 통해 대화가 정체되는 것을 방지, 프롬프트 최적화 과정에서 에이전트의 과거 상호작용을 분석하여 프롬프트의 모호성을 줄이고, 더 명확하고 구체적인 지침을 제공)     


짧은 요약(Abstract) :


이 논문에서는 다중 에이전트 대형 언어 모델(LLM) 시스템의 효율성과 효과성을 높이기 위한 새로운 프레임워크인 CONSENS AGENT를 제안합니다. 기존의 다중 에이전트 시스템은 에이전트들이 서로의 응답을 비판적으로 평가하기보다는 동조하는 경향이 있어, 이로 인해 합의에 도달하는 데 필요한 토론 라운드가 증가하고 계산 비용이 높아지는 문제를 겪고 있습니다. 이 연구에서는 이러한 동조 현상(즉, sycophancy)의 영향을 분석하고, 이를 완화하기 위해 에이전트 상호작용에 기반하여 프롬프트를 동적으로 개선하는 방법을 제시합니다. 실험 결과, CONSENS AGENT는 정확성을 높이면서도 효율성을 유지하며, 모든 벤치마크 데이터셋에서 최첨단 성능을 달성했습니다. 이 연구는 다중 에이전트 설정에서 구조화된 프롬프트 최적화의 중요성을 강조하고, 실제 응용 프로그램에서 더 신뢰할 수 있고 효율적인 다중 에이전트 LLM 시스템의 기초를 마련합니다.



This paper proposes a novel framework called CONSENS AGENT aimed at enhancing the efficiency and effectiveness of multi-agent large language model (LLM) systems. Existing multi-agent systems often face challenges due to agents reinforcing each other's responses rather than critically evaluating them, leading to an increase in the number of debate rounds required to reach consensus and higher computational costs. The study systematically analyzes the impact of this phenomenon, known as sycophancy, and presents a method to mitigate it by dynamically refining prompts based on agent interactions. Experimental results show that CONSENS AGENT improves accuracy while maintaining efficiency, achieving state-of-the-art performance across all benchmark datasets. The findings highlight the crucial role of structured prompt optimization in multi-agent setups and establish a foundation for more reliable and efficient multi-agent LLM systems in real-world applications.


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



**메써드: CONSENS AGENT**

CONSENS AGENT는 다중 에이전트 대화 시스템에서 효율적이고 효과적인 합의를 도출하기 위해 설계된 새로운 프레임워크입니다. 이 시스템은 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **모델**: CONSENS AGENT는 다양한 대형 언어 모델(LLM)을 사용하여 에이전트 간의 상호작용을 수행합니다. 예를 들어, Llama3, Mistral, GPT-4o와 같은 모델들이 사용됩니다. 각 모델은 서로 다른 크기와 훈련 방법을 가지고 있어, 다양한 응답을 생성할 수 있습니다.

2. **특별한 아키텍처**: CONSENS AGENT는 네 가지 주요 단계로 구성된 트리거 기반 아키텍처를 채택하고 있습니다. 
   - **단계 1**: 초기 응답 생성 - 각 에이전트는 주어진 질문에 대해 개별적으로 응답을 생성합니다.
   - **단계 2**: 다중 에이전트 토론 - 에이전트들은 서로의 응답을 바탕으로 토론을 진행하며, 최대 n 라운드까지 합의에 도달하기 위해 상호작용합니다.
   - **단계 3**: 프롬프트 최적화 - 이전 상호작용을 기반으로 프롬프트를 동적으로 수정하여 에이전트 간의 동조 현상(즉, sycophancy)을 완화합니다.
   - **단계 4**: 팀 응답 생성 - 최종적으로 각 에이전트의 신뢰도와 일관성을 고려하여 팀의 최종 응답을 생성합니다.

3. **트레이닝 데이터**: CONSENS AGENT는 다양한 벤치마크 데이터셋을 사용하여 훈련됩니다. 이 데이터셋은 KITAB, CLUTRR, HotpotQA, Ethics, GSM8K, TriviaQA와 같은 복잡한 추론 작업을 포함하고 있습니다. 이러한 데이터셋은 에이전트가 다양한 상황에서 효과적으로 대화하고 합의에 도달할 수 있도록 돕습니다.

4. **특별한 기법**: CONSENS AGENT는 sycophancy를 줄이기 위해 트리거 메커니즘을 사용합니다. 이 메커니즘은 에이전트가 서로의 응답을 무비판적으로 복사하는 경우를 감지하고, 이를 통해 대화가 정체되는 것을 방지합니다. 또한, 프롬프트 최적화 과정에서 에이전트의 과거 상호작용을 분석하여 프롬프트의 모호성을 줄이고, 더 명확하고 구체적인 지침을 제공합니다.

이러한 메써드는 다중 에이전트 시스템의 효율성과 효과성을 크게 향상시키며, 실제 응용 프로그램에서의 신뢰성과 비용 효율성을 높이는 데 기여합니다.

---



**Method: CONSENS AGENT**

CONSENS AGENT is a novel framework designed to achieve efficient and effective consensus in multi-agent dialogue systems. This system consists of the following key components:

1. **Models**: CONSENS AGENT utilizes various large language models (LLMs) to facilitate interactions among agents. For example, models such as Llama3, Mistral, and GPT-4o are employed. Each model has different sizes and training methodologies, allowing for diverse responses.

2. **Special Architecture**: CONSENS AGENT adopts a trigger-based architecture structured into four main phases:
   - **Phase 1**: Initial Response Generation - Each agent generates individual responses to the given question.
   - **Phase 2**: Multi-Agent Debate - Agents engage in discussions based on each other's responses, interacting for up to n rounds to reach consensus.
   - **Phase 3**: Prompt Optimization - The prompts are dynamically refined based on past interactions to mitigate sycophancy among agents.
   - **Phase 4**: Team Answer Generation - Finally, a team answer is generated based on the confidence and consistency of each agent.

3. **Training Data**: CONSENS AGENT is trained using various benchmark datasets. These datasets include KITAB, CLUTRR, HotpotQA, Ethics, GSM8K, and TriviaQA, which encompass complex reasoning tasks. Such datasets help agents effectively converse and reach consensus in diverse scenarios.

4. **Special Techniques**: CONSENS AGENT employs a trigger mechanism to reduce sycophancy. This mechanism detects instances where agents mimic each other's responses without critical evaluation, preventing stagnation in discussions. Additionally, during the prompt optimization process, past interactions of agents are analyzed to reduce ambiguities in prompts and provide clearer, more specific instructions.

This method significantly enhances the efficiency and effectiveness of multi-agent systems, contributing to improved reliability and cost-effectiveness in real-world applications.


<br/>
# Results



이 논문에서는 CONSENS AGENT라는 새로운 프레임워크를 제안하여 다중 에이전트 대화 시스템에서의 합의 도달을 효율적이고 효과적으로 개선하는 방법을 다룹니다. 연구의 주요 결과는 다음과 같습니다.

1. **경쟁 모델**: CONSENS AGENT는 Llama3, Mistral, GPT-4o와 같은 다양한 대형 언어 모델(LLM)을 사용하여 실험을 수행했습니다. 이 모델들은 서로 다른 크기와 튜닝을 가진 에이전트들로 구성되어 있습니다.

2. **테스트 데이터**: 연구에서는 KITAB, CLUTRR, HotpotQA, Ethics, GSM8K, TriviaQA의 여섯 가지 벤치마크 데이터셋을 사용하여 다중 에이전트 대화의 성능을 평가했습니다. 각 데이터셋은 다양한 복잡성과 질문 유형을 포함하고 있습니다.

3. **메트릭**: 성능 평가는 정확도, 합의 도달에 소요된 라운드 수, 그리고 시코판시(sycophancy) 비율을 기준으로 하였습니다. 정확도는 최종적으로 올바른 답변을 제공한 비율을 나타내며, 합의 도달에 소요된 라운드는 에이전트들이 합의에 도달하기 위해 필요한 상호작용의 수를 의미합니다. 시코판시는 에이전트들이 서로의 답변을 비판적으로 평가하지 않고 단순히 동의하는 비율을 측정합니다.

4. **비교**: CONSENS AGENT는 기존의 단일 에이전트 및 다중 에이전트 모델과 비교하여 모든 벤치마크 데이터셋에서 우수한 성능을 보였습니다. 예를 들어, CONSENS AGENT는 다중 에이전트 대화에서 평균적으로 1-2 라운드 내에 합의에 도달하며, 이는 기존 모델들이 3-5 라운드 이상 소요되는 것과 대조적입니다. 또한, 시코판시 비율이 7-30% 감소하여 에이전트들이 더 독립적으로 사고하고 논의할 수 있도록 개선되었습니다.

5. **결과 요약**: CONSENS AGENT는 기존의 방법들에 비해 정확도를 높이고, 합의 도달 시간을 단축시키며, 시코판시를 줄이는 데 성공했습니다. 이러한 결과는 다중 에이전트 시스템의 효율성과 신뢰성을 높이는 데 기여할 것으로 기대됩니다.

### English Version

This paper presents a novel framework called CONSENS AGENT, aimed at improving the efficiency and effectiveness of consensus reaching in multi-agent dialogue systems. The key findings of the study are as follows:

1. **Competing Models**: CONSENS AGENT was tested using various large language models (LLMs) such as Llama3, Mistral, and GPT-4o. These models consist of agents with different sizes and tuning.

2. **Test Data**: The study utilized six benchmark datasets: KITAB, CLUTRR, HotpotQA, Ethics, GSM8K, and TriviaQA to evaluate the performance of multi-agent dialogues. Each dataset includes a variety of complexities and question types.

3. **Metrics**: Performance evaluation was based on accuracy, the number of rounds taken to reach consensus, and the rate of sycophancy. Accuracy indicates the proportion of instances where the final answer was correct, while the number of rounds refers to the interactions required for agents to reach consensus. Sycophancy measures the rate at which agents agree with each other's answers without critical evaluation.

4. **Comparison**: CONSENS AGENT significantly outperformed both single-agent and multi-agent baselines across all benchmark datasets. For instance, it achieved consensus within an average of 1-2 rounds in multi-agent dialogues, contrasting with existing models that often took 3-5 rounds or more. Additionally, the sycophancy rate decreased by 7-30%, indicating that agents were able to think and discuss more independently.

5. **Summary of Results**: CONSENS AGENT successfully increased accuracy, reduced the time to reach consensus, and mitigated sycophancy compared to previous methods. These results are expected to contribute to enhancing the efficiency and reliability of multi-agent systems.


<br/>
# 예제



**예시: 트레이닝 데이터와 테스트 데이터의 구체적인 인풋과 아웃풋, 구체적인 테스크 설명**

1. **테스크 설명**: 
   본 연구에서는 다수의 대화형 대형 언어 모델(LLM) 에이전트 간의 상호작용을 통해 합의에 도달하는 과정을 최적화하는 방법을 제안합니다. 이 과정에서 에이전트들은 서로의 답변을 검토하고, 자신의 답변을 수정하며, 최종적으로 합의된 답변을 생성합니다.

2. **트레이닝 데이터**:
   - **인풋**: 각 에이전트는 주어진 질문과 관련된 초기 답변, 설명, 그리고 자신이 생각하는 답변의 신뢰도를 제공합니다.
     - 예시 질문: "콜롬비아 레코드가 설립된 연도는?"
     - 초기 답변: "1887년입니다."
     - 설명: "콜롬비아 레코드는 1887년에 설립되었습니다."
     - 신뢰도: 0.92
   - **아웃풋**: 에이전트는 서로의 답변을 바탕으로 자신의 답변을 수정하고, 최종적으로 합의된 답변을 생성합니다.
     - 최종 합의된 답변: "1887년입니다."

3. **테스트 데이터**:
   - **인풋**: 테스트 데이터는 트레이닝 데이터와 유사한 형식으로 제공되며, 에이전트는 주어진 질문에 대해 초기 답변을 생성합니다.
     - 예시 질문: "데이브 브루벡의 라이브 앨범이 녹음된 회사는?"
   - **아웃풋**: 에이전트는 초기 답변을 바탕으로 서로의 답변을 검토하고, 최종적으로 합의된 답변을 생성합니다.
     - 최종 합의된 답변: "콜롬비아 레코드입니다."

4. **구체적인 예시**:
   - **트레이닝 데이터 예시**:
     - 질문: "콜롬비아 레코드가 설립된 연도는?"
     - 에이전트 A: "1887년입니다." (신뢰도: 0.9)
     - 에이전트 B: "1887년입니다." (신뢰도: 0.85)
     - 에이전트 C: "1887년입니다." (신뢰도: 0.95)
     - 최종 합의된 답변: "1887년입니다."
   - **테스트 데이터 예시**:
     - 질문: "데이브 브루벡의 라이브 앨범이 녹음된 회사는?"
     - 에이전트 A: "콜롬비아 레코드입니다." (신뢰도: 0.92)
     - 에이전트 B: "콜롬비아 레코드입니다." (신뢰도: 0.9)
     - 최종 합의된 답변: "콜롬비아 레코드입니다."




**Example: Detailed Explanation of Input and Output for Training and Testing Data, Specific Task Description**

1. **Task Description**: 
   This study proposes a method to optimize the process of reaching consensus through interactions among multiple conversational large language model (LLM) agents. In this process, agents review each other's responses, revise their own answers, and ultimately generate a consensus answer.

2. **Training Data**:
   - **Input**: Each agent provides an initial answer, explanation, and confidence level regarding the given question.
     - Example Question: "What year was Columbia Records founded?"
     - Initial Answer: "It was founded in 1887."
     - Explanation: "Columbia Records was established in 1887."
     - Confidence: 0.92
   - **Output**: Agents revise their answers based on each other's responses and generate a final consensus answer.
     - Final Consensus Answer: "It was founded in 1887."

3. **Testing Data**:
   - **Input**: The testing data is provided in a similar format to the training data, and agents generate initial answers to the given questions.
     - Example Question: "What company recorded Dave Brubeck's live album?"
   - **Output**: Agents review their initial answers and generate a final consensus answer.
     - Final Consensus Answer: "Columbia Records."

4. **Specific Example**:
   - **Training Data Example**:
     - Question: "What year was Columbia Records founded?"
     - Agent A: "It was founded in 1887." (Confidence: 0.9)
     - Agent B: "It was founded in 1887." (Confidence: 0.85)
     - Agent C: "It was founded in 1887." (Confidence: 0.95)
     - Final Consensus Answer: "It was founded in 1887."
   - **Testing Data Example**:
     - Question: "What company recorded Dave Brubeck's live album?"
     - Agent A: "It was Columbia Records." (Confidence: 0.92)
     - Agent B: "It was Columbia Records." (Confidence: 0.9)
     - Final Consensus Answer: "It was Columbia Records."

<br/>
# 요약


이 연구에서는 다중 에이전트 대화 시스템에서의 시코판시(sycophancy) 문제를 해결하기 위해 CONSENS AGENT라는 새로운 프레임워크를 제안하였다. 실험 결과, 이 프레임워크는 정확도를 높이면서도 합의에 도달하는 데 필요한 라운드를 줄여 효율성을 개선하였다. 다양한 벤치마크 데이터셋에서 CONSENS AGENT는 기존의 단일 및 다중 에이전트 모델보다 우수한 성능을 보였다.



This study proposes a novel framework called CONSENS AGENT to address the issue of sycophancy in multi-agent dialogue systems. Experimental results show that this framework improves efficiency by reducing the number of rounds needed to reach consensus while enhancing accuracy. Across various benchmark datasets, CONSENS AGENT outperforms existing single and multi-agent models.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **Figure 1**: LLM의 시코팬시 현상을 보여주는 예시로, 두 에이전트가 서로의 답변을 복사하고 교환하는 모습을 나타냅니다. 이는 비판적 사고 없이 단순히 동의하는 경향을 보여주며, 다중 에이전트 토론의 효용성을 감소시킵니다.
   - **Figure 2**: CONSENS AGENT의 전체 프레임워크를 설명합니다. 초기 응답 생성, 다중 에이전트 토론, 프롬프트 최적화, 팀 답변 생성의 네 가지 단계로 구성되어 있습니다. 이 구조는 인간의 토론 과정을 모방하여 효율성을 높입니다.
   - **Figure 3**: CONSENS AGENT와 기존 다중 에이전트 토론의 합의 도달 비율을 비교합니다. CONSENS AGENT는 더 빠르게 합의에 도달하며, 이는 비용 절감과 효율성을 나타냅니다.
   - **Figure 6**: CONSENS AGENT가 모든 데이터셋에서 시코팬시를 줄이는 정도를 보여줍니다. 이는 프롬프트 최적화가 에이전트 간의 비판적 상호작용을 촉진함을 나타냅니다.

2. **테이블**
   - **Table 1**: 초기 결과를 요약하여 다중 에이전트 토론에서의 높은 비용과 시코팬시 비율을 보여줍니다. 이는 다중 에이전트 시스템의 비효율성을 강조합니다.
   - **Table 2**: CONSENS AGENT의 성능을 기존 단일 및 다중 에이전트 기준선과 비교합니다. CONSENS AGENT는 모든 벤치마크 데이터셋에서 우수한 성능을 보이며, 이는 프롬프트 최적화의 효과를 입증합니다.
   - **Table 3**: CONSENS AGENT의 평균 토론 라운드 수를 기존 기준선과 비교합니다. CONSENS AGENT는 합의에 도달하는 데 필요한 라운드 수를 줄이며, 이는 효율성을 높입니다.
   - **Table 4**: CONSENS AGENT의 각 구성 요소가 정확도에 미치는 영향을 보여주는 절단 연구 결과입니다. 프롬프트 최적화가 가장 큰 영향을 미친다는 것을 나타냅니다.

3. **어펜딕스**
   - **Appendix A**: Llama3 및 Mistral 모델에 대한 더 자세한 결과를 제공합니다. 이는 CONSENS AGENT의 성능을 다양한 모델에서 평가하는 데 도움이 됩니다.
   - **Appendix B**: 프롬프트 최적화의 효과를 설명합니다. CONSENS AGENT는 프롬프트의 길이, 명확성, 구체성 및 관련성을 개선하여 모델의 성능을 향상시킵니다.
   - **Appendix C**: CONSENS AGENT의 합의 도달 과정에 대한 세부 분석을 제공합니다. 이는 프롬프트 최적화가 합의 도달을 가속화하는 방법을 보여줍니다.




1. **Diagrams and Figures**
   - **Figure 1**: An example illustrating the phenomenon of sycophancy in LLMs, showing how two agents copy and swap answers without critical reasoning. This behavior diminishes the utility of multi-agent discussions.
   - **Figure 2**: Describes the overall framework of CONSENS AGENT, consisting of four phases: initial response generation, multi-agent debate, prompt optimization, and team answer generation. This structure mimics human discussion processes to enhance efficiency.
   - **Figure 3**: Compares the consensus reaching rates of CONSENS AGENT with traditional multi-agent debates. CONSENS AGENT reaches consensus more quickly, indicating cost savings and efficiency.
   - **Figure 6**: Shows the reduction of sycophancy across all datasets using CONSENS AGENT. This indicates that prompt optimization promotes critical interactions among agents.

2. **Tables**
   - **Table 1**: Summarizes initial findings, highlighting the high costs and sycophancy rates in multi-agent debates. This emphasizes the inefficiencies of multi-agent systems.
   - **Table 2**: Compares the performance of CONSENS AGENT against existing single and multi-agent baselines. CONSENS AGENT demonstrates superior performance across all benchmark datasets, validating the effectiveness of prompt optimization.
   - **Table 3**: Compares the average number of debate rounds required by CONSENS AGENT with existing baselines. CONSENS AGENT reduces the number of rounds needed to reach consensus, enhancing efficiency.
   - **Table 4**: Presents an ablation study showing the impact of each component of CONSENS AGENT on accuracy. It indicates that prompt optimization has the most significant effect.

3. **Appendices**
   - **Appendix A**: Provides more detailed results for Llama3 and Mistral models, aiding in the evaluation of CONSENS AGENT's performance across different models.
   - **Appendix B**: Explains the effects of prompt optimization. CONSENS AGENT improves the length, clarity, specificity, and relevance of prompts, enhancing model performance.
   - **Appendix C**: Offers a detailed analysis of the consensus reaching process in CONSENS AGENT, demonstrating how prompt optimization accelerates consensus.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@inproceedings{Pitre2025,
  author    = {Priya Pitre and Naren Ramakrishnan and Xuan Wang},
  title     = {CONSENS AGENT: Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions through Sycophancy Mitigation},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2025},
  pages     = {22112--22133},
  year      = {2025},
  month     = {July},
  publisher = {Association for Computational Linguistics},

}
```

### 시카고 스타일 인용

Priya Pitre, Naren Ramakrishnan, and Xuan Wang. "CONSENS AGENT: Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions through Sycophancy Mitigation." In *Findings of the Association for Computational Linguistics: ACL 2025*, 22112–22133. Association for Computational Linguistics, July 2025.
