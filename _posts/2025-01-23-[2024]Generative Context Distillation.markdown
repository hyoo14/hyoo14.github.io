---
layout: post
title:  "[2024]Generative Context Distillation"  
date:   2025-01-23 22:23:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


디스틸레이션-> 티처<->스튜던트   
이를 위한 더욱 효과적인 프롬프트 구조로 Think-Act 구조 제안  
더 나은 디스틸레이션 달성(프롬프트 없는 작업 수행과 기존 모델 대비 효율성 및 성능 개선)  




짧은 요약(Abstract) :    


현대 대형 언어 모델(LLM) 기반 애플리케이션에서 사용되는 프롬프트는 고정되고 길이가 긴 경우가 많아 계산 비용이 높아지는 문제가 있습니다. 이를 해결하기 위해, 저자들은 **생성적 컨텍스트 증류(Generative Context Distillation, GCD)**라는 경량 프롬프트 내재화 방법을 제안했습니다. GCD는 모델이 프롬프트 입력 없이도 동작을 복제할 뿐만 아니라, 프롬프트 내용과 모델 동작이 변경되어야 하는 이유까지 생성합니다. 이 접근법은 에이전트 기반 애플리케이션 시나리오에서 복잡한 프롬프트를 효과적으로 내재화하며, 프롬프트 없이도 높은 성능과 효율적인 추론을 가능하게 합니다. 또한, 대화형 데이터셋이 부족한 경우, 프롬프트와 환경 역할을 교체하여 대화형 데이터셋을 자동으로 수집하는 데이터 합성 기술을 소개합니다. GCD는 OS 상호작용 및 웹 에이전트 작업 등 다양한 테스트에서 높은 성능과 효율성을 입증했습니다.  

---

Prompts used in recent large language model-based applications are often fixed and lengthy, leading to significant computational overhead. To address this challenge, we propose **Generative Context Distillation (GCD)**, a lightweight prompt internalization method that employs a joint training approach. This method not only replicates the behavior of models with prompt inputs but also generates the content of the prompt along with reasons for why the model’s behavior should change accordingly. We demonstrate that our approach effectively internalizes complex prompts across various agent-based application scenarios. For effective training without interactions with the dedicated environments, we introduce a data synthesis technique that autonomously collects conversational datasets by swapping the roles of the agent and environment. This method is especially useful in scenarios where only a predefined prompt is available without a corresponding training dataset. By internalizing complex prompts, Generative Context Distillation enables high-performance and efficient inference without the need for explicit prompts.




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




논문에서 제안된 **생성적 컨텍스트 증류(Generative Context Distillation, GCD)** 방법은 고정된 프롬프트를 내재화하여 모델이 프롬프트 없이도 동일한 동작을 수행하도록 학습하는 것을 목표로 합니다. 이를 위해 다음과 같은 주요 요소와 기법이 사용되었습니다:

1. **데이터 생성**: 
   - 기존의 프롬프트를 기반으로 **자가 역할 전환 대화(Self Role-Playing Conversation)** 기술을 활용해 가상의 사용자 입력과 대화 데이터를 생성합니다. 여기서 에이전트와 환경의 역할을 교환하며 대화 데이터를 수집합니다.
   - 또한, 프롬프트와 관련된 이유(reason)를 생성하기 위해 대형 언어 모델(LLM)을 사용하여 모델 출력의 변화 이유를 설명하는 데이터를 추가로 생성합니다.

2. **모델 아키텍처 및 손실 함수**:
   - 기본적으로 **LLaMA-3-8B-Instruct** 모델을 사용하며, 이를 **QLoRA**로 경량화하여 특정 프롬프트에 적응하도록 미세 조정합니다.
   - 손실 함수는 두 가지로 구성됩니다:
     1. **SFT Loss**: 기존의 시퀀스 수준 지식 증류(Sequence-Level Knowledge Distillation) 기법을 사용해 교사 모델의 동작을 학생 모델이 모방하도록 학습합니다.
     2. **Prompt Generation Loss (PG Loss)**: 학생 모델이 프롬프트 내용과 출력 변화의 이유를 생성하도록 학습합니다. 이를 통해 모델은 프롬프트의 정보를 직접적으로 학습하게 됩니다.
   - 두 손실은 하이퍼파라미터 \( \lambda \)를 통해 조합하여 최적화됩니다.

3. **학습 데이터**:
   - 약 1,000개의 가상 사용자 입력 및 다중 턴 대화 데이터를 생성하여 학습 데이터로 사용합니다.
   - 이러한 데이터는 에이전트 기반 애플리케이션의 프롬프트를 내재화하기 위한 것으로, 환경 상호작용 없이도 효과적으로 학습할 수 있도록 설계되었습니다.

4. **효율성**:
   - GCD는 프롬프트 입력을 필요로 하지 않기 때문에 추론 시 토큰 수를 줄이고, 계산 비용을 절감합니다. 이를 통해 기존의 압축 또는 증류 기법보다 39% 효율성을 개선하였습니다.

---



The **Generative Context Distillation (GCD)** method aims to internalize fixed prompts, enabling the model to perform identically without explicit prompt inputs. The methodology is based on the following key components:

1. **Data Generation**:
   - A **Self Role-Playing Conversation** technique is used to generate pseudo user inputs and conversational datasets by swapping the roles of the agent and environment in a prompt.
   - Additionally, a large language model (LLM) is employed to generate "reasons" explaining why the output of the student model should align with the teacher model’s output.

2. **Model Architecture and Loss Functions**:
   - The base model used is **LLaMA-3-8B-Instruct**, fine-tuned with **QLoRA** to adapt to specific prompts efficiently.
   - Two types of loss functions are combined:
     1. **SFT Loss**: Adapts the sequence-level knowledge distillation approach to mimic the teacher model's behavior.
     2. **Prompt Generation Loss (PG Loss)**: Trains the student model to generate both the content of the prompt and the reasoning behind output changes, allowing for direct learning of the prompt.
   - These losses are combined using a hyperparameter \( \lambda \) to optimize the training process.

3. **Training Data**:
   - Approximately 1,000 pseudo user inputs and multi-turn conversational datasets are generated for training.
   - This data is designed to internalize agent-based prompts effectively without requiring interaction with a live environment.

4. **Efficiency**:
   - By eliminating the need for prompt inputs during inference, GCD reduces the number of tokens, significantly lowering computational overhead. This approach improves efficiency by 39% compared to existing compression or distillation methods.



   
 
<br/>
# Results  




논문에서는 **AgentBench**라는 벤치마크 데이터셋을 활용하여 GCD의 성능을 평가했습니다. 이 벤치마크는 다음 세 가지 에이전트 작업으로 구성됩니다:

1. **OS 상호작용**:
   - Ubuntu Docker 환경에서 Bash 명령어를 실행하는 작업으로, 성공률(Success Rate, SR) 메트릭으로 평가되었습니다.
   - GCD는 **100% 성공률**을 달성하며 기존 모델 대비 최상의 성능을 보였습니다.

2. **웹 브라우징**:
   - HTML 기반의 다중 선택 문제로 구성된 작업으로, 선택 정확도와 동작 정확도를 평가하는 성공률(SR) 메트릭을 사용했습니다.
   - GCD는 **82.35%**의 성능을 기록하며 기존 방법보다 높은 정확도를 달성했습니다.

3. **웹 쇼핑**:
   - Amazon 웹사이트에서 상품을 탐색하고 구매하는 작업으로, 예상 상품 속성과 실제 선택된 상품 속성의 유사도를 측정하는 **보상(Rewards)** 메트릭으로 평가되었습니다.
   - GCD는 **82.09%**의 보상 점수를 기록하며 다른 방법보다 25% 이상의 성능 향상을 보여주었습니다.

#### 비교 대상 모델:
- **프롬프트 압축 기반 모델**: LLMLingua-2 및 ICAE
  - 프롬프트를 압축하여 계산 비용을 줄이지만, 정보 손실로 인해 성능이 크게 저하되었습니다. 예를 들어, LLMLingua-2는 압축 비율이 30%를 초과하면 모든 작업에서 실패했습니다.
- **지식 증류 기반 모델**: SeqKD 및 SeqKD+KLD
  - SeqKD는 일부 작업(Web Browsing)에서 유망한 결과를 보였으나, GCD보다 낮은 성능을 기록했습니다.
- **프롬프트 전처리 모델**: 100% 또는 50% 확률로 프롬프트를 추가하는 방식
  - GCD보다 낮은 성능을 보였으며, 특히 긴 프롬프트에서 훈련-추론 불일치 문제로 성능 저하가 발생했습니다.

#### GCD의 성능 우위:
GCD는 프롬프트 입력 없이도 기존 프롬프트 기반 모델과 동일한 수준의 성능을 유지하거나 이를 초과했습니다. 또한, OS 상호작용 작업에서는 교사 모델(Upper Bound)과 동일한 최고 성능을 달성하며 효율성을 크게 향상시켰습니다. 특히 웹 쇼핑 작업에서는 **39% 효율성 개선**을 통해 계산 비용을 줄이면서도 높은 성능을 유지했습니다.

---



The paper evaluates the performance of GCD using the **AgentBench** benchmark dataset, which includes the following three agent tasks:

1. **OS Interaction**:
   - This task involves executing Bash commands in an Ubuntu Docker environment and is evaluated using the **Success Rate (SR)** metric.
   - GCD achieved a **100% success rate**, demonstrating superior performance compared to other models.

2. **Web Browsing**:
   - This task formulates HTML-based multi-choice problems, evaluated by accuracy in element selection and operation correctness using the **Success Rate (SR)** metric.
   - GCD recorded a **82.35% success rate**, outperforming other methods in accuracy.

3. **Web Shopping**:
   - This task involves navigating Amazon websites to find and purchase a product, evaluated using a **Rewards** metric that measures the similarity between the expected and selected product attributes.
   - GCD achieved a **82.09% rewards score**, showing a **25% performance improvement** over competing methods.

#### Baseline Models:
- **Prompt Compression Models**: LLMLingua-2 and ICAE
  - These models compress prompts to reduce computation but suffer from significant performance drops due to information loss. For example, LLMLingua-2 failed in all tasks when the compression rate exceeded 30%.
- **Knowledge Distillation Models**: SeqKD and SeqKD+KLD
  - SeqKD showed promising results in certain tasks (e.g., Web Browsing) but lagged behind GCD in overall performance.
- **Prompt Prepending Models**: Models using 100% or 50% probability of prepending prompts.
  - These models underperformed compared to GCD, particularly in handling long prompts due to train-test distribution mismatches.

#### GCD’s Advantage:
GCD maintained or exceeded the performance of prompt-based models without requiring explicit prompts. For the OS Interaction task, it achieved **upper bound-level performance**, matching the teacher model's results. Additionally, in the Web Shopping task, GCD improved computational efficiency by **39%**, significantly reducing overhead while maintaining high performance.





<br/>
# 예제  





논문에서 다룬 주요 테스트 데이터의 구체적인 예시와, GCD가 어떻게 제안된 작업을 잘 수행하고 비교 모델은 실패하는지를 아래와 같습니다.

---

##### **테스트 데이터 예시 (OS 상호작용)**:
1. **입력(prompt)**: 
   - "사용자는 `/usr/bin` 디렉토리 안의 파일 수를 알고 싶어 합니다. 당신은 Bash 명령어를 사용해 이 작업을 수행해야 합니다."
   - 프롬프트 예시: 
     ```
     Think: 파일의 수를 세기 위해 먼저 모든 파일을 나열하고 Bash 명령어를 실행해야 합니다.
     Act: bash
     ```bash
     ls -l /usr/bin | wc -l
     ```
     ```
2. **GCD의 출력**:
   - Think: `/usr/bin` 디렉토리의 파일 수를 확인하기 위해 Bash 명령어를 실행해야 합니다.
   - Act: bash
     ```bash
     ls -l /usr/bin | wc -l
     ```
   - Output: 1245 (파일 수 계산 결과).

3. **비교 모델 (LLMLingua-2)의 출력**:
   - Think: 사용자 질문에 따라 Bash 명령어를 제공해야 합니다.
   - Act: answer
     "파일을 세려면 `ls` 명령어와 `wc` 명령어를 사용해야 합니다."
   - Output: 구체적인 명령어 실행 없이 설명만 제공.

---

##### **테스트 데이터 예시 (웹 쇼핑)**:
1. **입력(prompt)**:
   - "Amazon에서 `Sony WH-1000XM5 헤드폰`을 검색하고 구매 버튼을 눌러야 합니다."
   - 프롬프트 예시:
     ```
     Think: 제품 이름을 검색하고 결과를 필터링하여 구매 버튼을 클릭해야 합니다.
     Act: search
     "Sony WH-1000XM5"
     ```

2. **GCD의 출력**:
   - Think: 검색 결과에서 원하는 제품이 나타나는지 확인해야 합니다.
   - Act: click
     "Buy Now"
   - Output: 성공적으로 제품 구매.

3. **비교 모델 (ICAE)의 출력**:
   - Think: 제품을 찾기 위해 검색해야 합니다.
   - Act: search
     "Sony 헤드폰"
   - Output: 결과가 너무 포괄적이고, "Buy Now" 버튼을 클릭하지 못함.

---

#### GCD의 강점:
- GCD는 프롬프트의 구조적 정보를 내재화하여 구체적인 행동을 취합니다(예: 정확한 Bash 명령어 실행, "Buy Now" 버튼 클릭).
- 비교 모델은 프롬프트 내용을 적절히 압축하지 못하거나 출력 행동이 프롬프트와 일치하지 않아 성능이 저하됩니다.

---



##### **Test Data Example (OS Interaction)**:
1. **Input(prompt)**:
   - "The user wants to know the number of files in the `/usr/bin` directory. Use Bash commands to perform this task."
   - Prompt example:
     ```
     Think: To count the number of files, I need to list all files and execute a Bash command.
     Act: bash
     ```bash
     ls -l /usr/bin | wc -l
     ```
     ```

2. **GCD Output**:
   - Think: To count the number of files in `/usr/bin`, I need to execute a Bash command.
   - Act: bash
     ```bash
     ls -l /usr/bin | wc -l
     ```
   - Output: 1245 (number of files calculated successfully).

3. **Baseline Model (LLMLingua-2) Output**:
   - Think: I need to provide a Bash command based on the user's query.
   - Act: answer
     "To count the files, you should use `ls` and `wc` commands."
   - Output: Provides an explanation but fails to execute the command.

---

##### **Test Data Example (Web Shopping)**:
1. **Input(prompt)**:
   - "Search for `Sony WH-1000XM5 headphones` on Amazon and click the Buy Now button."
   - Prompt example:
     ```
     Think: I need to search for the product and filter the results to click the Buy Now button.
     Act: search
     "Sony WH-1000XM5"
     ```

2. **GCD Output**:
   - Think: I need to verify if the desired product appears in the search results.
   - Act: click
     "Buy Now"
   - Output: Successfully completes the purchase of the product.

3. **Baseline Model (ICAE) Output**:
   - Think: I need to search for the product.
   - Act: search
     "Sony headphones"
   - Output: Too broad results; fails to click the "Buy Now" button.

---

#### GCD’s Strengths:
- GCD internalizes the structural information of prompts to perform precise actions (e.g., executing the exact Bash command, clicking the "Buy Now" button).
- Baseline models either fail to compress the prompt effectively or produce outputs misaligned with the prompt, resulting in performance degradation.


<br/>  
# 요약   



생성적 컨텍스트 증류(GCD)는 고정된 프롬프트를 모델 내부에 내재화하여 프롬프트 없이도 동일한 동작을 수행하도록 학습하는 방법론입니다. 이를 위해 GCD는 대화 데이터를 생성하기 위해 에이전트와 환경의 역할을 교환하는 "자가 역할 전환 대화" 기술을 활용하며, 프롬프트 내용과 출력 변화의 이유를 학습하는 손실 함수를 결합하여 최적화합니다. 결과적으로, GCD는 OS 상호작용, 웹 브라우징, 웹 쇼핑과 같은 작업에서 기존 모델을 능가하는 성능을 보여줍니다. 예를 들어, GCD는 `/usr/bin` 디렉토리의 파일 수를 계산할 때 정확한 Bash 명령어를 실행하거나, Amazon에서 특정 제품을 검색하고 구매하는 작업을 성공적으로 수행했습니다. 반면, 비교 모델은 명령어 실행 대신 단순 설명을 제공하거나, 검색 결과에서 구체적인 작업을 수행하지 못해 GCD와의 성능 차이가 명확히 드러났습니다.

---


Generative Context Distillation (GCD) is a methodology that internalizes fixed prompts within a model, enabling it to perform identical tasks without explicit prompts. To achieve this, GCD employs a "self-role-playing conversation" technique, which generates conversational data by swapping the roles of the agent and the environment, and optimizes a combined loss function to learn both the prompt content and the reasoning behind output changes. As a result, GCD outperforms baseline models in tasks such as OS interaction, web browsing, and web shopping. For instance, GCD successfully executed precise Bash commands to count files in the `/usr/bin` directory and completed a purchase on Amazon by accurately searching for a specific product. In contrast, baseline models either provided explanations instead of executing commands or failed to perform specific actions from the search results, highlighting GCD’s superior performance.


<br/>  
# 기타  





1. **Figure 1: GCD 개요**  
   그림 1은 생성적 컨텍스트 증류(GCD)의 전반적인 구조와 동작 원리를 보여줍니다. 이 그림은 에이전트와 환경 간의 상호작용을 나타내며, GCD가 어떻게 역할 전환(self-role-playing)을 통해 대화 데이터를 생성하고, 프롬프트 내용과 출력 이유를 학습하는지를 설명합니다. 또한, 프롬프트가 제거된 상태에서 모델이 동일한 작업을 수행하도록 학습하는 과정을 시각적으로 표현합니다.

2. **Table 1: AgentBench 결과 비교**  
   테이블 1은 AgentBench 벤치마크에서 GCD와 비교 모델의 성능을 보여줍니다. OS 상호작용, 웹 브라우징, 웹 쇼핑 작업에서의 성공률(SR) 및 보상(Rewards) 메트릭이 포함되어 있으며, GCD가 모든 작업에서 가장 높은 점수를 기록한 것을 확인할 수 있습니다. 특히, GCD는 OS 상호작용에서 100%의 성공률을 달성하고, 웹 쇼핑 작업에서는 비교 모델보다 약 25% 높은 보상 점수를 보여줍니다.

3. **Figure 2: GCD의 프롬프트 내재화 과정**  
   그림 2는 GCD가 복잡한 프롬프트를 내재화하는 과정을 단계적으로 보여줍니다. 입력 프롬프트를 바탕으로 데이터 생성을 시작하며, 모델이 프롬프트 내용뿐 아니라 행동 변화의 이유를 학습하도록 최적화됩니다. 또한, 생성된 데이터가 모델 학습에 어떻게 사용되는지 시각적으로 설명합니다.

4. **Table 2: 효율성 비교**  
   테이블 2는 GCD와 다른 프롬프트 압축 및 증류 기법의 효율성을 비교한 결과를 나타냅니다. GCD는 프롬프트가 없는 상태에서 추론 시간을 줄이고 계산 비용을 크게 절감했으며, 기존 모델 대비 39%의 효율성 향상을 달성했습니다.

5. **Figure 3: 웹 쇼핑 작업에서의 예제**  
   그림 3은 GCD와 비교 모델이 웹 쇼핑 작업에서 수행한 과정을 비교한 사례를 보여줍니다. GCD는 명확한 검색 결과를 제공하고, "Buy Now" 버튼을 성공적으로 클릭한 반면, 비교 모델은 너무 광범위한 검색 결과를 반환하거나 특정 동작을 수행하지 못했습니다.

---



1. **Figure 1: Overview of GCD**  
   Figure 1 illustrates the overall architecture and working principles of Generative Context Distillation (GCD). It depicts the interaction between the agent and the environment, showing how GCD uses self-role-playing to generate conversational data and learn the prompt content and reasoning behind the output. The figure visually explains how the model is trained to perform tasks without explicit prompts.

2. **Table 1: AgentBench Results Comparison**  
   Table 1 presents the performance of GCD and baseline models on the AgentBench benchmark. Metrics such as Success Rate (SR) and Rewards are provided for tasks like OS interaction, web browsing, and web shopping. GCD achieved the highest scores across all tasks, including a 100% success rate in OS interaction and a reward score approximately 25% higher than the baseline in the web shopping task.

3. **Figure 2: Prompt Internalization Process in GCD**  
   Figure 2 shows the step-by-step process of how GCD internalizes complex prompts. It starts with generating data based on input prompts and optimizes the model to learn both the prompt content and the reasoning behind behavior changes. The figure visually explains how the generated data is utilized during model training.

4. **Table 2: Efficiency Comparison**  
   Table 2 compares the efficiency of GCD with other prompt compression and distillation methods. GCD reduced inference time and computational costs significantly without prompts, achieving a 39% improvement in efficiency compared to baseline models.

5. **Figure 3: Example from Web Shopping Task**  
   Figure 3 provides an example comparing the processes of GCD and baseline models in the web shopping task. GCD delivered precise search results and successfully clicked the "Buy Now" button, while baseline models either returned overly broad search results or failed to perform specific actions.


<br/>
# 논문 구조 및 구성   



1. **서론 (Introduction)**  
   - 논문의 문제 제기와 연구 동기가 제시됩니다.  
   - 현대 대형 언어 모델(LLM) 기반 애플리케이션에서 고정된 프롬프트 사용으로 인한 계산 비용 문제를 강조하며, 이를 해결하기 위한 **Generative Context Distillation (GCD)** 기법의 필요성을 제안합니다.  
   - 연구의 주요 기여점과 성과를 간략히 소개하며, 이후 섹션에서 다룰 내용을 간략히 개요로 설명합니다.

2. **관련 연구 (Related Work)**  
   - 기존의 프롬프트 압축 및 증류 기법, 지식 증류 기반 방법, 그리고 LLM의 활용에 관한 연구들이 논의됩니다.  
   - GCD가 기존 연구들과 어떻게 차별화되고 이를 어떻게 개선하는지를 설명합니다.  

3. **방법론 (Methodology)**  
   - GCD의 핵심 아이디어와 구체적인 구현 방법이 소개됩니다.  
   - "자가 역할 전환 대화"를 통한 데이터 생성 과정, 프롬프트 내재화를 위한 모델 구조 및 손실 함수 설계, 그리고 학습 데이터 준비 과정이 상세히 설명됩니다.  
   - 모델의 학습 프로세스와 최적화 방법이 도식과 함께 제시됩니다.

4. **실험 설정 (Experimental Setup)**  
   - 실험에 사용된 벤치마크 데이터셋(AgentBench)과 평가 메트릭(Success Rate, Rewards 등)이 설명됩니다.  
   - 비교 대상으로 선택된 모델들과 실험 환경에 대한 세부 사항이 제공됩니다.

5. **결과 및 분석 (Results and Analysis)**  
   - GCD의 성능이 OS 상호작용, 웹 브라우징, 웹 쇼핑 작업에서 기존 모델을 어떻게 능가하는지를 정량적 결과를 통해 보여줍니다.  
   - 성공적인 예제와 실패한 비교 모델의 예제를 비교하며 GCD의 우수성을 강조합니다.  
   - GCD의 효율성 개선과 비용 절감 효과도 논의됩니다.

6. **한계 및 미래 작업 (Limitations and Future Work)**  
   - GCD의 한계와 아직 해결되지 않은 문제들을 논의합니다.  
   - 모델이 특정 시나리오에서 겪을 수 있는 제약과 이를 해결하기 위한 향후 연구 방향이 제시됩니다.

7. **결론 (Conclusion)**  
   - 연구의 주요 성과를 요약하며, GCD가 대형 언어 모델 기반 작업에 미치는 긍정적인 영향을 재강조합니다.  
   - 연구가 지닌 의의와 향후 발전 가능성을 간략히 언급하며 마무리합니다.  

이 구조는 연구의 필요성과 해결책, 성과를 논리적으로 연결하며 독자에게 명확한 이해를 제공  


<br/>
# refer format:     


@article{shin2024gcd,
  title={Generative Context Distillation},
  author={Shin, Haebin and Ji, Lei and Gong, Yeyun and Kim, Sungdong and Choi, Eunbi and Seo, Minjoon},
  journal={arXiv preprint arXiv:2411.15927},
  year={2024},
  url={https://arxiv.org/abs/2411.15927}
}





Haebin Shin, Lei Ji, Yeyun Gong, Sungdong Kim, Eunbi Choi, and Minjoon Seo. "Generative Context Distillation." arXiv preprint arXiv:2411.15927 (2024). Available at: https://arxiv.org/abs/2411.15927.  



