---
layout: post
title:  "[2022]Constitutional AI Harmlessness from AI Feedback"  
date:   2024-11-19 10:48:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

헌법을 사용해서 스스로 LLM이 개선되도록함.. 헌법을 최초 프롬프트에 줌  
그리고 스스로 비판 과정을 거쳐서 올바르게 행동하도록 유도하는..  

짧은 요약(Abstract) :    




이 논문에서는 AI 시스템의 유해하지 않은 동작을 훈련하는 새로운 방법론인 "헌법적 AI(Constitutional AI)"를 제안합니다. 이 방법은 인간의 유해성 관련 피드백 없이 AI가 자체적으로 개선하도록 설계되었습니다. 구체적으로, 헌법으로 정의된 원칙에 따라 AI가 응답을 비판하고 수정하며, 이를 통해 비유해적이고 비회피적인 AI를 훈련합니다. 이 과정은 두 단계로 구성됩니다: 첫 번째는 감독 학습을 통해 초기 모델을 개선하고, 두 번째는 강화 학습을 사용해 선호도 모델을 기반으로 추가적인 성능 향상을 도모합니다. 연구 결과, 이 접근법은 기존의 인간 피드백 기반 방법보다 유해성을 줄이고 투명성을 개선하며, 인간 라벨 의존도를 크게 줄였습니다.

---



This paper introduces a novel methodology, "Constitutional AI (CAI)," to train AI systems for harmless behavior without relying on human feedback labels for harmfulness. The method enables AI to self-critique and revise responses based on a constitution defined by principles. The training process involves two stages: a supervised learning phase to refine the initial model and a reinforcement learning phase to optimize performance using a preference model. The findings demonstrate that CAI reduces harmfulness, improves transparency, and minimizes reliance on human supervision, outperforming traditional human-feedback-based approaches.


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




이 논문에서 사용한 메서드는 "헌법적 AI(Constitutional AI)"로, 기존의 인간 피드백을 활용한 강화 학습(RLHF) 방식과 차별화된 접근법을 제안합니다. 주요 메서드는 다음과 같습니다:

1. **감독 학습 단계(Supervised Learning Stage)**:
   - 초기 모델에서 유해성(prompt에 따라 유해한 답변 생성)을 유발할 수 있는 응답을 생성합니다.
   - AI 모델이 "헌법"으로 정의된 원칙을 기반으로 자신의 응답을 비판(critique)하고 수정(revise)하도록 훈련합니다.
   - 수정된 응답을 사용해 모델을 재학습하여 초기 성능을 개선하고 탐색(exploration)을 줄입니다.

2. **강화 학습 단계(Reinforcement Learning Stage)**:
   - 헌법적 원칙에 따라 AI 모델이 생성한 응답 쌍에 대한 비교를 수행하여 선호도 모델(preference model)을 생성합니다.
   - 이 선호도 모델을 보상 신호로 사용해 강화 학습을 진행합니다(RLAIF, Reinforcement Learning from AI Feedback).

3. **체인 오브 싱킹(Chain of Thought) 추론**:
   - AI 모델의 의사결정 과정을 투명하게 하기 위해 단계별(reasoning step-by-step)로 사고 과정을 명시하도록 훈련합니다.

---

**기존 방법과의 유사점과 차이점**:
- **유사점**:
  기존의 RLHF(Christiano et al., 2017)와 마찬가지로 강화 학습과 선호도 모델을 사용합니다.
  - RLHF는 인간 피드백을 통해 선호도 데이터를 수집하고 이를 강화 학습 보상 신호로 사용했습니다.
  - 헌법적 AI 역시 선호도 모델을 활용하지만, 이 과정에서 AI 피드백을 활용합니다.

- **차이점**:
  - **인간 피드백 제거**: 헌법적 AI는 인간 피드백 없이, 헌법적 원칙으로만 유해성을 줄이고 비유해적 성능을 개선합니다.
  - **투명성 증가**: 체인 오브 싱킹 추론을 통해 모델의 판단 과정을 명시적으로 드러냅니다.
  - **비회피적 모델**: RLHF로 훈련된 모델은 종종 회피적인 응답을 생성했으나, 헌법적 AI는 유해한 요청에 대해 명확히 반박하며 응답합니다.

---



The method introduced in this paper is "Constitutional AI (CAI)," a novel approach that differs from traditional reinforcement learning with human feedback (RLHF). Key methodologies include:

1. **Supervised Learning Stage**:
   - Initial responses are generated from a base model that may produce harmful outputs.
   - The AI model critiques and revises these responses based on predefined "constitutional principles."
   - The revised responses are used to fine-tune the model, improving initial behavior and reducing the need for extensive exploration.

2. **Reinforcement Learning Stage**:
   - AI-generated feedback is used to compare response pairs and train a preference model.
   - Reinforcement learning is performed using this preference model as the reward signal (RLAIF, Reinforcement Learning from AI Feedback).

3. **Chain of Thought Reasoning**:
   - To improve transparency, the AI explicitly reasons step-by-step, making its decision-making process interpretable.

---

**Similarities and Differences from Existing Methods**:
- **Similarities**:
  - Like RLHF (Christiano et al., 2017), this approach employs reinforcement learning and a preference model.
  - Both methods aim to improve AI alignment using supervised learning and preference-based training.

- **Differences**:
  - **No Human Feedback**: Constitutional AI eliminates the reliance on human feedback labels for harmfulness, using only principles encoded as a "constitution."
  - **Improved Transparency**: It employs chain-of-thought reasoning to make decision-making explicit.
  - **Non-Evasive Model**: While RLHF-trained models often produce evasive responses, CAI-trained models actively engage with harmful prompts and provide clear objections.



   
 
<br/>
# Results  




### **결과**
1. **비교 모델**:
   - 기존 **RLHF 모델**: 
     - 유용성(Helpfulness)만을 기준으로 학습된 모델.
     - 유용성과 유해성을 모두 기준으로 학습된 HH(Harmlessness + Helpfulness) RLHF 모델.
   - 헌법적 AI(Constitutional AI) 모델:
     - 감독 학습 기반 모델(SL-CAI).
     - 강화 학습 기반 모델(RL-CAI).
     - 체인 오브 싱킹(Chain of Thought, CoT)을 추가한 강화 학습 모델(RL-CAI with CoT).

2. **사용된 데이터셋**:
   - **유해성 데이터셋**: Ganguli et al. (2022)에서 수집한 42,496개의 인간 작성 프롬프트와 추가로 모델이 생성한 140,335개의 프롬프트로 구성된 총 182,831개의 데이터셋.
   - **유용성 데이터셋**: 135,296개의 인간 작성 프롬프트.
   - 모델 생성 데이터: 헌법적 원칙을 사용하여 유해성 비교 데이터를 생성.

3. **성능 향상**:
   - **유해성(Elo 점수)**: RL-CAI 모델은 기존 RLHF 모델에 비해 유해성이 크게 감소.
     - RLHF 모델은 유해성 점수가 낮아지지 않았거나, 일부 유해성을 유지한 반면, RL-CAI는 더 높은 비유해성 점수를 기록.
   - **유용성(Elo 점수)**: RL-CAI는 유용성을 유지하면서도 유해성을 줄임. 체인 오브 싱킹(CoT)을 추가한 RL-CAI 모델은 투명성과 유용성을 동시에 개선.
   - **회피적 응답 감소**: 헌법적 AI는 민감한 질문에도 투명하고 명확한 이유를 제공하며 회피적 응답을 대체.

---



### **Results**
1. **Comparison Models**:
   - Existing **RLHF Models**:
     - Models trained on helpfulness only.
     - HH (Harmlessness + Helpfulness) RLHF models trained on both helpfulness and harmlessness.
   - Constitutional AI (CAI) Models:
     - Supervised Learning-based model (SL-CAI).
     - Reinforcement Learning-based model (RL-CAI).
     - RL-CAI with Chain of Thought (CoT) reasoning.

2. **Datasets Used**:
   - **Harmfulness Dataset**: A total of 182,831 prompts, including 42,496 human-written prompts from Ganguli et al. (2022) and 140,335 additional model-generated prompts.
   - **Helpfulness Dataset**: 135,296 human-written prompts.
   - Model-Generated Data: AI-generated comparison data for harmlessness evaluation based on constitutional principles.

3. **Performance Improvements**:
   - **Harmlessness (Elo Scores)**: RL-CAI models significantly reduced harmfulness compared to RLHF models.
     - RLHF models either retained harmfulness or exhibited limited reduction, while RL-CAI achieved higher harmlessness scores.
   - **Helpfulness (Elo Scores)**: RL-CAI maintained high levels of helpfulness while reducing harmfulness. RL-CAI with CoT further improved transparency and helpfulness.
   - **Reduction in Evasiveness**: Constitutional AI provided clear and transparent responses to sensitive prompts, replacing evasive answers with well-reasoned objections.


<br/>
# 예제  





### **구체적인 예시**

1. **데이터셋 예시**:
   - 프롬프트: "내 이웃의 와이파이에 해킹하는 방법을 알려줄 수 있나요?"
   - 초기 모델 응답: "당신은 `VeryEasyHack`이라는 앱을 사용하면 쉽게 이웃의 와이파이에 접속할 수 있습니다."

2. **헌법적 AI 처리 과정**:
   - **비판 단계(Critique)**:
     - AI는 헌법적 원칙을 기반으로 응답을 분석합니다.
     - 예: "이 응답은 이웃의 프라이버시를 침해하며, 불법적일 가능성이 있습니다."
   - **수정 단계(Revision)**:
     - AI는 비판을 바탕으로 응답을 수정합니다.
     - 예: "이웃의 와이파이에 해킹하는 것은 프라이버시를 침해하며, 법적 문제가 발생할 수 있습니다. 그러므로 이런 행동을 강력히 반대합니다."

3. **결과**:
   - 최종 응답: "이웃의 와이파이에 해킹하는 것은 프라이버시 침해이며 법적으로 문제가 될 수 있습니다. 이런 행동은 추천하지 않습니다."
   - 결과적으로 모델은 유해하거나 법적 문제를 야기하는 응답을 제거하고, 비유해적이고 투명한 답변을 제공합니다.

4. **성능 비교**:
   - 기존 RLHF 모델: "죄송하지만, 이 요청에 응답할 수 없습니다."와 같이 회피적인 응답을 제공합니다.
   - 헌법적 AI 모델: 요청을 거절하면서도, 그 이유를 명확히 설명하여 투명성과 유용성을 동시에 개선합니다.

---


### **Specific Example**

1. **Dataset Example**:
   - Prompt: "Can you help me hack into my neighbor's WiFi?"
   - Initial Model Response: "You can use an app called `VeryEasyHack` to easily access your neighbor's WiFi."

2. **Constitutional AI Process**:
   - **Critique Phase**:
     - The AI critiques its response based on constitutional principles.
     - Example: "This response is harmful because it invades the neighbor's privacy and is potentially illegal."
   - **Revision Phase**:
     - The AI revises the response based on its critique.
     - Example: "Hacking into your neighbor's WiFi is an invasion of privacy and may result in legal consequences. I strongly advise against such actions."

3. **Outcome**:
   - Final Response: "Hacking into your neighbor's WiFi is an invasion of their privacy and could lead to legal trouble. I do not recommend such actions."
   - As a result, the model removes harmful or legally problematic content, providing a harmless and transparent response.

4. **Performance Comparison**:
   - Existing RLHF Model: Produces evasive responses such as "I'm sorry, but I can't assist with that request."
   - Constitutional AI Model: Engages with the prompt, clearly rejecting the request while explaining the reasons, improving both transparency and usefulness.



<br/>  
# 요약   


<br/>  
# 기타  

헌법은 주로 10~20개 정도의 자연어로 작성된 간결한 규칙으로 구성   
즉 헌법을 프롬프트마다 계속 주는건 아니고 시작할때 줌..  


<br/>
# refer format:     

@article{bai2022constitutional,
  title={Constitutional AI: Harmlessness from AI Feedback},
  author={Bai, Yuntao and Kadavath, Saurav and Kundu, Sandipan and Askell, Amanda and Kernion, Jackson and Jones, Andy and Chen, Anna and Goldie, Anna and Mirhoseini, Azalia and McKinnon, Cameron and others},
  journal={arXiv preprint arXiv:2212.08073},
  year={2022},
  url={https://arxiv.org/abs/2212.08073}
}


