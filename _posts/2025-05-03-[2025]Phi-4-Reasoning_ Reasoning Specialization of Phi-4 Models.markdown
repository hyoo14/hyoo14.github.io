---
layout: post
title:  "[2025]Phi-4-Reasoning: Reasoning Specialization of Phi-4 Models"  
date:   2025-05-03 17:34:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 


think token 쓴게 신기하네.. 이걸 추가해서 생각 부분을 학습때 넣어준건가   
SFT(supervised fine tuning.. 지도학습), RL로 학습..  걍 지도학습/강화학습   
결론적으로 14B로 70B 버금가거나 능가했다고 주장.. 굿   



짧은 요약(Abstract) :    




이 논문에서는 **Phi-4-reasoning**이라는 140억 파라미터의 추론 특화 언어 모델을 소개합니다. 이 모델은 \*\*교사 모델(o3-mini)\*\*이 생성한 고품질 추론 데이터를 바탕으로 **선별된 '학습 가능한' 프롬프트**에 대해 \*\*감독 학습(SFT)\*\*을 통해 훈련되었습니다. 또한 **Phi-4-reasoning-plus**라는 버전은 \*\*보상 기반 강화 학습(RL)\*\*을 짧게 적용해 성능을 더욱 향상시켰습니다.

이 두 모델은 **수학, 과학 추론, 코딩, 계획 수립, 공간 이해** 등의 다양한 분야에서, 훨씬 더 큰 모델인 **DeepSeek-R1-Distill-70B**보다 높은 성능을 보이며, **671B 규모의 DeepSeek-R1** 모델과도 경쟁 가능한 결과를 보여줍니다.

감독 학습을 통한 정교한 데이터 큐레이션은 추론 언어 모델의 성능 향상에 효과적이며, RL을 추가로 적용하면 더욱 강화된 추론 능력을 얻을 수 있음을 실험적으로 보여줍니다. 이 보고서는 학습 데이터, 학습 방법론, 평가 결과를 포괄적으로 다루며, 향후 추론 모델의 평가 방식 개선의 필요성도 제안합니다.

---


We introduce Phi-4-reasoning, a 14-billion parameter reasoning model that achieves strong performance on complex reasoning tasks. Trained via supervised fine-tuning of Phi-4 on carefully curated set of “teachable” prompts–selected for the right level of complexity and diversity–and reasoning demonstrations generated using o3-mini, Phi-4-reasoning generates detailed reasoning chains that effectively leverage inference-time compute. We further develop Phi-4-reasoning-plus, a variant enhanced through a short phase of outcome-based reinforcement learning that offers higher performance by generating longer reasoning traces. Across a wide range of reasoning tasks, both models outperform significantly larger open-weight models such as DeepSeek-R1-Distill-Llama-70B model and approach the performance levels of full DeepSeek-R1 model. Our comprehensive evaluations span benchmarks in math and scientific reasoning, coding, algorithmic problem solving, planning, and spatial understanding. Interestingly, we observe a non-trivial transfer of improvements to general-purpose benchmarks as well. In this report, we provide insights into our training data, our training methodologies, and our evaluations. We show that the benefit of careful data curation for supervised fine-tuning (SFT) extends to reasoning language models, and can be further amplified by reinforcement learning (RL). Finally, our evaluation points to opportunities for improving how we assess the performance and robustness of reasoning models.

---




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



####  백본 모델 및 아키텍처 (Backbone Model & Architecture)

* **Phi-4-reasoning**은 **14B 파라미터**의 **Phi-4 모델**을 기반으로 **Supervised Fine-Tuning(SFT)** 방식으로 훈련되었습니다.
* **기본 아키텍처는 Phi-4와 동일**하되, 다음과 같은 **두 가지 주요 수정 사항**이 적용됨:

  1. `<think>` 및 `</think>` 토큰 도입 → reasoning block을 명시적으로 구분
  2. **토큰 길이 확장**: 기존 최대 길이 16K에서 32K로 확장 (RoPE base frequency를 2배로 증가시킴)

####  학습 데이터 (Training Data)

* 총 **140만 개 이상의 prompt-response pairs**, 약 **83억 개의 고유 토큰** 사용
* 주요 데이터 도메인:

  * 수학(MATH)
  * 코딩
  * 안전성과 Responsible AI 관련 alignment 데이터
* \*\*응답(Reasoning traces)\*\*은 OpenAI의 **o3-mini 모델**을 사용하여 생성한 고품질 synthetic data
* 데이터는 **자체 필터링 및 난이도 기준**을 기반으로 엄선된 문제(Seeds)로 구성됨

####  훈련 방식 (Training Setup)

* 총 약 **16,000 스텝** 동안 학습
* **Batch size**: 32
* **컨텍스트 길이**: 32K
* **Optimizer**: AdamW (learning rate: 1e-5, warmup steps: 450, weight decay: 1e-4)
* SFT 이후 별도의 **Reinforcement Learning(RL)** 단계(Phi-4-reasoning-plus)도 존재하나, 이는 별도 확장입니다.

---


####  Backbone Model & Architecture

* **Phi-4-reasoning** is a 14-billion parameter model obtained through **supervised fine-tuning (SFT)** of the base **Phi-4** model.
* The architecture remains the same as Phi-4, with two notable modifications:

  1. **Reasoning tokens**: Two placeholder tokens are repurposed as `<think>` and `</think>` to delimit the reasoning block.
  2. **Extended token context**: The token length was increased from 16K to **32K tokens** by doubling the RoPE base frequency.

####  Training Data

* The SFT dataset consists of **1.4M+ prompt-response pairs**, totaling around **8.3B unique tokens**.
* Major domains include:

  * Math (e.g., AIME, OmniMath)
  * Coding
  * Safety/alignment topics aligned with Responsible AI
* Responses are **synthetically generated using the o3-mini model**, and prompts are curated from filtered public and synthetic sources selected for their complexity and "teachability."

####  Training Configuration

* Training was conducted for **\~16,000 steps** using:

  * **Global batch size**: 32
  * **Context length**: 32K tokens
  * **Optimizer**: AdamW (LR = 1e-5, warmup = 450 steps, weight decay = 1e-4)
* Phi-4-reasoning can be further enhanced via **reinforcement learning**, resulting in Phi-4-reasoning-plus.

---





   
 
<br/>
# Results  




**1. 테스트 데이터셋**
모델은 다양한 분야의 추론 벤치마크에서 평가되었습니다. 주요 벤치마크는 다음과 같습니다:

* **AIME 2025** (미국 수학경시대회): 30문항
* **Omni-MATH**: 4,428개의 고난이도 수학 문제
* **HMMT Feb 2025**: 30문항
* **GPQA**: 198문항 (과학 추론)
* **LiveCodeBench**: 2024.08\~2025.01 기간의 코딩 문제
* **Codeforces**: ID 1505\~1536에 해당하는 143개 문제
* **BA-Calendar**, **3SAT**, **TSP**, **Maze**, **SpatialMap** 등 다양한 계획/공간/조합 문제 포함.

**2. 입력과 출력**

* 입력: 문제 설명 및 프롬프트 (주로 Chain-of-Thought 요구)
* 출력: 모델이 추론하여 생성한 풀이 또는 최종 정답
* 일부 테스크에서는 10\~64회의 generation을 통해 평균 정확도(pass\@1 등)를 측정함.

**3. 경쟁 모델**

* **DeepSeek-R1**, **DeepSeek-R1-Distill-Llama-70B**
* **OpenAI o1, o3-mini**
* **Claude 3.7 Sonnet**
* **Gemini 2 Flash Thinking**
  이 중 상당수가 70B 이상 크기의 모델이며, Phi-4 시리즈(14B)는 그보다 훨씬 작음에도 유사하거나 더 나은 성능을 보임.

**4. 성능 및 메트릭**

* **수학 분야**: AIME 및 Omni-MATH에서 Phi-4 대비 50% 정확도 향상
* **코딩 분야**: LiveCodeBench에서 25% 향상, Codeforces에서 Elo rating으로 평가
* **조합/계획 문제**: TSP, 3SAT, Calendar Planning에서 30\~60% 향상
* **일반 벤치마크**: IFEval, ArenaHard, FlenQA 등에서도 10\~20% 향상
* **정확도 기준**: pass\@1, Elo rating, instruction-following 정확도, 평균 응답 길이 대비 성능 등 다양한 측면에서 평가.

---


**1. Test Datasets**
The models were evaluated on a wide range of reasoning benchmarks, including:

* **AIME 2025** (30 math problems)
* **Omni-MATH** (4,428 olympiad-level problems)
* **HMMT February 2025**, **GPQA**, **LiveCodeBench**, **Codeforces**
* **3SAT**, **TSP**, **BA-Calendar**, **Maze**, and **SpatialMap**.

**2. Task Inputs/Outputs**

* **Input**: Prompts involving mathematical, logical, or planning tasks (often with Chain-of-Thought required).
* **Output**: The model's generated solution or final answer.
* Evaluation involves multiple generations per task (e.g., 50-64 for AIME and GPQA) to measure variance and accuracy.

**3. Competitor Models**

* **DeepSeek-R1**, **DeepSeek-R1-Distill-Llama-70B**
* **OpenAI's o1, o3-mini**
* **Claude 3.7 Sonnet**, **Gemini 2 Flash Thinking**

Despite being smaller (14B parameters), Phi-4-reasoning(-plus) performs comparably to or better than these larger models.

**4. Performance & Metrics**

* **Math Benchmarks**: +50% accuracy over Phi-4
* **Coding Tasks**: +25% on LiveCodeBench, Elo rating on Codeforces
* **Planning/Algorithmic Tasks**: +30-60% on TSP, 3SAT, and Calendar Planning
* **General Benchmarks**: +22 on IFEval, +10 on ArenaHard, +16 on FlenQA
* **Metrics Used**: pass\@1, Elo rating, instruction-following accuracy, and token-efficiency tradeoffs.





<br/>
# 예제  




**1. 학습 데이터 예시**

* 학습은 약 **140만 개의 prompt-response 쌍**으로 구성된 데이터셋을 기반으로 진행되었으며,
* 데이터는 수학, 과학, 코딩, 논리 퍼즐, 책임감 있는 AI 등의 도메인에서 수집되었고,
* 고품질의 **Chain-of-Thought (CoT) reasoning trace**가 포함된 응답이 사용되었습니다.
* 예시 문제는 웹에서 수집하거나, 기존 데이터셋 기반으로 재구성된 **합성 문제**(synthetic prompts)입니다.

  * 예: 수학 문제 “삼각형 ABC에서 AB=13, BC=10일 때... AC는 얼마인가?” 같이 변형됨.

**2. 테스트 데이터 및 태스크 예시**

* 모델은 다음과 같은 **복잡한 추론 태스크**에서 평가됨:

  * **수학:** AIME 2025, Omni-Math (4000+ 수학 경시대회 문제)
  * **코딩:** LiveCodeBench (실시간 코드 문제 해결)
  * **계획:** BA-Calendar (시간대, 우선순위 등을 고려한 일정 조율 문제)
  * **알고리즘:** 3SAT, TSP (NP-Hard 문제 해결)
  * **공간 추론:** Maze, SpatialMap (경로 탐색, 기하학적 관계 추론)
* 예시 문제:

  * “아래의 일정 제약 조건 하에 월요일 중 가능한 회의 시간대를 찾아라” (BA-Calendar)
  * “0번 노드에서 시작하여 목적지까지 가는 경로를 adjacency list 기반으로 찾아라” (Maze task).

---



**1. Training Data Examples**

* Phi-4-reasoning was trained on **1.4 million prompt-response pairs** across reasoning-heavy domains like math, coding, safety, and logic.
* Responses included **long chain-of-thought reasoning traces** generated using the o3-mini model.
* The prompts were either drawn from high-quality web and public datasets or were **synthetically rewritten** versions of existing questions.

  * Example: A raw web question about triangle perimeter was transformed into a verifiable short-answer question such as:
    *“In triangle ABC with AB=13 and BC=10, if perimeter(△AMC) = perimeter(△CNA) and perimeter(△ANB) = perimeter(△CMB), what is AC?”*.

**2. Test Data and Task Examples**

* The model was evaluated on the following **reasoning-intensive tasks**:

  * **Mathematics:** AIME 2025, Omni-MATH (4000+ olympiad-level problems)
  * **Coding:** LiveCodeBench (real-world programming problems)
  * **Planning:** BA-Calendar (finding valid meeting times with constraints)
  * **Algorithmic Reasoning:** 3SAT, TSP (NP-hard problems)
  * **Spatial Reasoning:** Maze, SpatialMap (pathfinding, geometric reasoning)
* Task Examples:

  * *“Find a feasible meeting slot on Monday given each participant’s time constraints and buffers”* (BA-Calendar)
  * *“Given a graph in adjacency list form, output a valid path from node 0 to the destination”* (Maze task).

---




<br/>  
# 요약   




Phi-4-reasoning은 Phi-4(14B)를 기반으로 `<think>` 토큰, 32K 컨텍스트 길이 등을 적용해 고난이도 프롬프트에 대해 SFT 및 RL로 학습된 추론 특화 모델이다.
수학, 코딩, 계획, 공간 추론 등 다양한 벤치마크에서 70B 모델들과 유사하거나 더 나은 성능을 보이며, pass\@1, Elo 등 다양한 메트릭으로 평가되었다.
학습 데이터는 o3-mini가 생성한 Chain-of-Thought 응답을 포함한 수학/코딩 중심의 프롬프트이며, 테스트에서는 AIME, BA-Calendar, Maze 등 실제 문제 기반 예시들이 사용되었다.

---



Phi-4-reasoning is a 14B-parameter model fine-tuned from Phi-4 with `<think>` token delimiters and extended 32K context, trained using SFT and RL on carefully selected reasoning prompts.
It outperforms or matches 70B-scale models like DeepSeek-R1 on math, coding, planning, and spatial reasoning benchmarks, using metrics such as pass\@1 and Elo rating.
The training set includes chain-of-thought responses generated by o3-mini, and evaluations include real-world examples like AIME math problems, BA-Calendar scheduling, and Maze pathfinding tasks.



<br/>  
# 기타  





* **부록(Apendix) 및 피규어(Figures)**:

  * Figure 8, 14, 15, 16 등 다양한 피규어를 통해 세부적인 벤치마크 결과(예: AIME, GPQA, SAT, TSP 등)를 시각화.
  * 예를 들어, Figure 15는 Omni-MATH의 주제별 정확도를 보여주며, 이산 수학과 기하학에서 모든 모델의 성능이 낮음을 강조함.
  * Figure 16은 GPQA의 분야별 성능(물리, 생물, 화학)과 토큰 사용량을 함께 보여주며 물리에서 가장 좋은 성능을 보였음을 시사.

* **에러 분석 및 다양한 세부 실험**:

  * 동일한 프롬프트에 대해 50\~64번씩 독립적으로 모델 출력을 생성해 분산(variance)을 분석. 이는 모델 성능이 한 번의 테스트 결과에 크게 영향을 받을 수 있다는 점을 보여줌(Figure 9).
  * 다양한 디코딩 전략 (예: 평균, 다수결, 최고 결과 등)에 따른 성능 차이도 비교(Figure 17 참조).

* **모델 출력 예시**:

  * “How many strawberries for 9 r’s?”, “Describe Seattle using only S-words” 같은 창의적 질문에 대한 Phi-4-reasoning의 응답이 예시로 제시됨.
  * 이는 모델이 학습 데이터 범위를 벗어난 질문에도 추론 능력을 잘 활용할 수 있음을 보여줌.

---



* **Appendix and Figures**:

  * Figures such as 8, 14, 15, and 16 present detailed benchmark performances, including task-level accuracy breakdowns.
  * For example, Figure 15 highlights domain-specific performance on Omni-MATH, showing that all models perform worse on Discrete Math and Geometry.
  * Figure 16 visualizes GPQA accuracy by domain (Physics, Biology, Chemistry) and token usage per model, indicating best results in Physics.

* **Error and Variance Analysis**:

  * The authors generate 50–64 independent outputs per prompt to analyze performance variance, demonstrating how single-run results can be unreliable (see Figure 9).
  * They also explore differences in performance across decoding strategies like average-of-N, majority vote, and best-of-N generations (e.g., Figure 17).

* **Model Output Examples**:

  * The appendix includes examples such as a riddle ("How many strawberries for 9 r’s?") and constrained generation ("Describe Seattle using only S-words").
  * These examples illustrate the model’s ability to generalize reasoning to out-of-distribution prompts not seen during training.



<br/>
# refer format:     



@misc{chen2024phi4reasoning,
  title={Phi-4-Reasoning: Reasoning Specialization of Phi-4 Models}, 
  author={Yen-Chun Chen and Subhojeet Pramanik and Subhajit Chaudhury and Mohammadamin Barekatain and Ofir Press and Manzil Zaheer and Jianshu Chen and Mandar Joshi and Elnaz Nouri and Christian Puhrsch and Sudha Rao and Thomas Scialom and Yujia Xie and Xiaodong Liu and Yichong Xu and Zhirui Zhang and Mohit Bansal and Peter Chin and Pradeep Dasigi and Chris DuBois and Xinyun Chen and Adam Fisch and Tatsunori B. Hashimoto and Vu Ha and Ehsan Imani and Nitish Shirish Keskar and Tushar Khot and Mike Lewis and Zhenqin Lu and Jesse Mu and Colin Raffel and Aakanksha Naik and Jeff Z. HaoChen and Xuezhi Wang and Sherry Yang and Eric Zelikman and Yuhuai Wu and Andrew M. Dai and Rattandeep Singh and Wen-tau Yih and Ece Kamar and Sudipta Sengupta and Sebastian Nowozin and Margaret Mitchell and Yoon Kim and Alex Smola and Sam Bowman and Hyung Won Chung and Yao Fu and Daniel Fried and Xinyu Hua and Jaewoo Kang and Faisal Ladhak and Jiachang Liu and Mark Neumann and Qianli Ma and Swaroop Mishra and Swaroop Ramaswamy and Eric Wallace and Tianyi Zhang and Dan Garrette and Noah A. Smith and William Wang},
  year={2024},
  eprint={2504.21318},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2504.21318}
}




Chen, Yen-Chun, Subhojeet Pramanik, Subhajit Chaudhury, Mohammadamin Barekatain, Ofir Press, Manzil Zaheer, Jianshu Chen, Mandar Joshi, Elnaz Nouri, Christian Puhrsch, et al. "Phi-4-Reasoning: Reasoning Specialization of Phi-4 Models." arXiv preprint arXiv:2504.21318 (2024). https://arxiv.org/abs/2504.21318.







