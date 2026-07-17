---
layout: post
title:  "[2026]Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce"
date:   2026-07-17 02:48:59 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문은 AI 에이전트의 자동화 및 증강 가능성을 평가하기 위해 1,500명의 도메인 근로자와 52명의 AI 전문가의 데이터를 수집하여 WORKBank 데이터베이스를 구축하였다.


짧은 요약(Abstract) :


이 논문은 복합 AI 시스템(즉, AI 에이전트)의 급속한 발전이 노동 시장을 어떻게 변화시키고 있는지를 다루고 있습니다. 특히, 일자리 대체, 인간의 자율성 감소, 자동화에 대한 과도한 의존에 대한 우려가 커지고 있지만, 현재의 기술적 능력과 노동자들이 AI 에이전트에게 자동화 또는 보강을 원하는 작업에 대한 체계적인 이해가 부족하다는 점을 지적합니다. 이를 해결하기 위해, 저자들은 노동자들이 AI 에이전트에게 자동화 또는 보강을 원하는 작업을 평가할 수 있는 새로운 감사 프레임워크를 제안합니다. 이 프레임워크는 오디오 강화 미니 인터뷰를 통해 노동자의 미세한 욕구를 포착하고, 인간의 개입 수준을 정량화하기 위한 인간 에이전시 척도(Human Agency Scale, HAS)를 도입합니다. 이 연구는 1,500명의 도메인 노동자와 AI 전문가의 능력 평가를 바탕으로 104개 직업에 걸쳐 844개의 작업에 대한 데이터를 수집하여, AI 에이전트 개발의 기회와 노동자의 기대를 조화롭게 맞추는 중요성을 강조합니다.



This paper addresses how the rapid rise of compound AI systems (i.e., AI agents) is reshaping the labor market, raising concerns about job displacement, diminished human agency, and overreliance on automation. However, there is a lack of systematic understanding of the evolving landscape regarding the current technological capabilities and the tasks that workers want AI agents to automate or augment. To address this gap, the authors propose a novel auditing framework to assess which occupational tasks workers desire AI agents to automate or augment. This framework features an audio-enhanced mini-interview to capture nuanced worker desires and introduces the Human Agency Scale (HAS) as a shared language to quantify the preferred level of human involvement. Using this framework, the study constructs a database based on responses from 1,500 domain workers and capability assessments from AI experts across 844 tasks spanning 104 occupations, highlighting the importance of aligning AI agent development with human desires and preparing workers for evolving workplace dynamics.


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



이 연구에서는 AI 에이전트의 자동화 및 증강 가능성을 평가하기 위해 새로운 감사 프레임워크를 개발했습니다. 이 프레임워크는 미국 노동부의 O*NET 데이터베이스에서 소싱한 직업별 작업을 기반으로 하며, 작업 수준에서의 감사 접근 방식을 채택하여 실제 작업의 복잡성과 맥락을 더 잘 포착할 수 있도록 설계되었습니다.

#### 1. 감사 프레임워크의 설계 원칙
- **작업의 복잡성 및 맥락**: 프레임워크는 특정 직업과 관련된 복잡한 다단계 작업에 초점을 맞추며, 단순한 저수준 활동이 아닌 실제 직무 책임을 반영합니다.
- **컴퓨터 호환 작업**: AI 에이전트가 영향을 미칠 가능성이 높은 컴퓨터에서 수행되는 작업에 한정합니다.
- **인간 에이전시 척도 (Human Agency Scale, HAS)**: 자동화와 증강의 스펙트럼을 평가하기 위해 H1(인간 개입 없음)에서 H5(인간 개입 필수)까지의 5단계 척도를 도입합니다. 이 척도는 작업 완료에 필요한 인간의 개입 정도를 정량화하는 데 사용됩니다.

#### 2. 데이터 수집 및 분석
- **설문조사 시스템**: 도메인 작업자들이 자신의 경험을 공유하고 작업에 대한 피드백을 제공할 수 있도록 오디오 지원 설문조사 시스템을 사용합니다. 각 작업에 대해 작업자들은 자동화 욕구(Aw(t))와 원하는 HAS 수준(Hw(t))을 5점 리커트 척도로 평가합니다.
- **AI 전문가 평가**: AI 연구 및 개발 경험이 있는 전문가들이 각 작업의 현재 자동화 능력(Ae(t))과 가능한 HAS 수준(He(t))을 평가합니다. 이 이중 접근 방식은 작업자와 전문가 간의 인식 차이를 드러내고, AI 에이전트 통합의 준비 상태를 이해하는 데 도움을 줍니다.

#### 3. WORKBank 데이터베이스 구축
- **데이터베이스 구성**: 2025년 1월부터 5월까지 수집된 데이터를 바탕으로, 1,500명의 작업자와 52명의 AI 전문가의 응답을 포함하는 WORKBank 데이터베이스를 구축했습니다. 이 데이터베이스는 104개 직업에 걸쳐 844개의 작업을 포함하고 있으며, 작업자 욕구와 AI 에이전트의 기술적 능력을 포괄적으로 반영합니다.

#### 4. 분석 방법
- **혼합 효과 모델**: 작업자의 자동화 욕구 평가가 작업의 고유한 특성에 의해 얼마나 영향을 받는지를 분석하기 위해 혼합 효과 모델을 사용합니다. 이 모델은 개인의 인구 통계학적 특성과 태도 변수를 통제하여 작업 수준에서의 신호를 추출합니다.
- **Jensen-Shannon Divergence (JSD)**: 작업자와 전문가의 HAS 수준 간의 불일치를 정량화하기 위해 JSD를 사용하여 두 분포 간의 차이를 측정합니다.

이러한 방법론을 통해 연구는 AI 에이전트의 자동화 및 증강 가능성을 체계적으로 평가하고, 작업자와 기술 간의 불일치를 파악하여 향후 AI 개발 방향을 제시합니다.

---




In this study, we developed a novel auditing framework to assess the automation and augmentation potential of AI agents. This framework is based on occupational tasks sourced from the U.S. Department of Labor's O*NET database and adopts a task-level auditing approach to better capture the complexity and context of real-world work.

#### 1. Design Principles of the Auditing Framework
- **Complexity and Context of Tasks**: The framework focuses on complex, multi-step tasks associated with specific occupations, reflecting actual job responsibilities rather than isolated low-level activities.
- **Computer-Compatible Tasks**: It is limited to tasks performed on computers, recognizing their susceptibility to AI agents.
- **Human Agency Scale (HAS)**: A five-level scale from H1 (no human involvement) to H5 (human involvement essential) is introduced to evaluate the spectrum between automation and augmentation. This scale quantifies the degree of human involvement required for task completion.

#### 2. Data Collection and Analysis
- **Survey System**: An audio-enhanced survey system allows domain workers to share their experiences and provide feedback on tasks. For each task, workers rate their automation desire (Aw(t)) and desired HAS level (Hw(t)) using a 5-point Likert scale.
- **Expert Assessments**: AI experts with experience in research and development assess the current automation capability (Ae(t)) and feasible HAS level (He(t)) for each task. This dual approach reveals discrepancies between worker and expert perceptions and helps understand the readiness for AI agent integration.

#### 3. Construction of the WORKBank Database
- **Database Composition**: Based on data collected from January to May 2025, we constructed the WORKBank database, which includes responses from 1,500 workers and 52 AI experts. This database encompasses 844 tasks across 104 occupations, reflecting worker desires and technological capabilities comprehensively.

#### 4. Analytical Methods
- **Mixed-Effects Model**: A mixed-effects model is employed to analyze the extent to which workers' automation desire ratings are influenced by intrinsic task properties, controlling for individual demographic characteristics and attitudinal variables.
- **Jensen-Shannon Divergence (JSD)**: JSD is used to quantify the divergence between worker-desired and expert-assessed HAS levels, measuring the differences between these two distributions.

Through this methodology, the study systematically evaluates the automation and augmentation potential of AI agents, identifying mismatches between worker desires and technological capabilities to inform future AI development directions.


<br/>
# Results



이 논문에서는 AI 에이전트의 자동화 및 증강 가능성을 평가하기 위해 WORKBank 데이터베이스를 구축하고, 이를 통해 도출된 결과를 바탕으로 여러 가지 중요한 발견을 제시합니다. 

1. **작업 자동화에 대한 근로자의 선호**: 연구에 따르면, 46.1%의 작업에 대해 근로자들은 AI 에이전트의 자동화를 긍정적으로 평가했습니다. 특히 반복적이고 저가치의 작업에 대한 자동화 욕구가 높았으며, 이는 근로자들이 더 높은 가치의 작업에 시간을 할애하고자 하는 욕구에서 비롯되었습니다.

2. **자동화 욕구-능력 경관**: WORKBank의 데이터를 통해 근로자들이 원하는 자동화 수준과 AI 전문가들이 평가한 기술적 능력을 비교하여 네 가지 영역으로 나누었습니다:
   - **자동화 "그린 라이트" 존**: 높은 자동화 욕구와 높은 기술적 능력을 가진 작업.
   - **자동화 "레드 라이트" 존**: 높은 기술적 능력에도 불구하고 낮은 자동화 욕구를 가진 작업.
   - **R&D 기회 존**: 높은 자동화 욕구를 가지지만 현재 기술적 능력이 낮은 작업.
   - **저우선 존**: 낮은 자동화 욕구와 낮은 기술적 능력을 가진 작업.

   이 경관은 AI 에이전트 개발에 있어 중요한 투자 우선순위를 제시합니다.

3. **인간 에이전시 스케일(HAS)**: 연구 결과, 근로자들은 AI 에이전트와의 협업을 선호하는 경향이 있으며, 45.2%의 직업에서 H3(동등한 파트너십)가 지배적인 수준으로 나타났습니다. 이는 AI의 발전이 인간의 역할에 대한 기대를 변화시킬 수 있음을 시사합니다.

4. **핵심 인간 기술의 변화**: AI 에이전트의 통합이 진행됨에 따라 정보 처리 중심의 기술에서 대인 관계 및 조직적 기술로의 전환이 나타나고 있습니다. 이는 AI가 인간의 역할을 대체하기보다는 보완하는 방향으로 나아가고 있음을 보여줍니다.

5. **비교 및 메트릭**: 연구에서는 Y Combinator의 투자 패턴과 AI 에이전트 연구 논문을 분석하여, 현재의 투자와 연구 노력이 근로자들의 욕구와 기술적 가능성 간의 불일치를 보여주고 있음을 강조합니다. 예를 들어, 많은 투자와 연구가 소프트웨어 개발 및 비즈니스 분석에 집중되어 있으며, 이는 "그린 라이트" 존과 기회 존의 작업에 대한 충분한 관심이 부족함을 나타냅니다.

이러한 결과들은 AI 에이전트의 개발 및 통합에 있어 근로자의 요구와 기술적 현실을 조화롭게 맞추는 것이 중요하다는 점을 강조합니다.

---




This paper presents findings based on the construction of the WORKBank database, which evaluates the automation and augmentation potential of AI agents. The results lead to several significant discoveries:

1. **Worker Preferences for Task Automation**: The study found that 46.1% of tasks were positively rated by workers for automation by AI agents. This desire for automation is particularly high for repetitive and low-value tasks, driven by workers' wish to allocate more time to higher-value work.

2. **Desire-Capability Landscape**: By comparing the automation desire of workers with the technological capabilities assessed by AI experts, tasks in WORKBank were categorized into four zones:
   - **Automation "Green Light" Zone**: Tasks with high automation desire and high capability.
   - **Automation "Red Light" Zone**: Tasks with high capability but low desire for automation.
   - **R&D Opportunity Zone**: Tasks with high desire but currently low capability.
   - **Low Priority Zone**: Tasks with both low desire and low capability.

   This landscape provides critical insights for prioritizing investments in AI agent development.

3. **Human Agency Scale (HAS)**: The results indicate that workers generally prefer higher levels of human agency in collaboration with AI agents, with 45.2% of occupations showing H3 (equal partnership) as the dominant level. This suggests that the integration of AI may shift expectations regarding human roles.

4. **Shift in Core Human Skills**: As AI agents are integrated, there is a noticeable shift from information-processing skills to interpersonal and organizational skills. This indicates that AI is moving towards complementing rather than replacing human roles.

5. **Comparisons and Metrics**: The study analyzed investment patterns from Y Combinator and AI agent research papers, highlighting a misalignment between current investments and the joint perspectives of technology developers and the workers the technology aims to support. For instance, much of the investment and research is concentrated in software development and business analysis, indicating a lack of attention to tasks in the "Green Light" and Opportunity Zones.

These findings underscore the importance of aligning AI agent development with worker desires and technological realities.


<br/>
# 예제



이 논문에서는 AI 에이전트의 자동화 및 증강 가능성을 평가하기 위해 WORKBank라는 데이터베이스를 구축했습니다. 이 데이터베이스는 1,500명의 도메인 작업자와 52명의 AI 전문가의 응답을 기반으로 하며, 104개의 직업에 걸쳐 844개의 작업을 포함하고 있습니다. 이 연구의 주요 목표는 작업자들이 AI 에이전트가 자동화하거나 증강하기를 원하는 작업을 이해하고, 이러한 선호가 현재의 기술적 능력과 어떻게 일치하는지를 평가하는 것입니다.

#### 예시: 트레이닝 데이터와 테스트 데이터

1. **트레이닝 데이터**:
   - **입력**: 각 작업에 대한 작업자들의 피드백. 예를 들어, "AI가 이 작업을 완전히 수행할 수 있다면 얼마나 원하십니까?"라는 질문에 대한 응답을 수집합니다. 응답은 1에서 5까지의 리커트 척도로 평가됩니다.
   - **출력**: 각 작업에 대한 평균 자동화 욕구 점수(Aw(t))와 원하는 인간 에이전시 수준(Hw(t)). 예를 들어, "세무사: 고객과의 약속을 잡는 작업"에 대해 작업자가 5점 만점에 4점을 주었다면, 이 작업의 Aw(t)는 4입니다.

2. **테스트 데이터**:
   - **입력**: AI 전문가들이 평가한 각 작업의 기술적 능력(Ae(t))과 가능한 인간 에이전시 수준(He(t)). 예를 들어, "AI가 이 작업을 수행하는 데 필요한 기술적 능력은 얼마나 됩니까?"라는 질문에 대한 전문가의 평가를 수집합니다.
   - **출력**: 각 작업에 대한 평균 기술적 능력 점수와 전문가가 평가한 인간 에이전시 수준. 예를 들어, "세무사: 고객과의 약속을 잡는 작업"에 대해 전문가가 3점을 주었다면, 이 작업의 Ae(t)는 3입니다.

이러한 데이터는 WORKBank 데이터베이스에 저장되어, 작업자와 전문가의 관점을 통합하여 AI 에이전트의 자동화 및 증강 가능성을 평가하는 데 사용됩니다.

---




In this paper, a database called WORKBank was constructed to assess the automation and augmentation potential of AI agents. This database is based on responses from 1,500 domain workers and 52 AI experts, covering 844 tasks across 104 occupations. The main goal of this research is to understand which tasks workers want AI agents to automate or augment and how these preferences align with current technological capabilities.

#### Example: Training Data and Test Data

1. **Training Data**:
   - **Input**: Feedback from workers on each task. For example, responses to the question, "If an AI can do this task completely, how much do you want it to do for you?" are collected. Responses are rated on a Likert scale from 1 to 5.
   - **Output**: Average automation desire score (Aw(t)) and desired Human Agency Scale level (Hw(t)) for each task. For instance, if a worker rated "Tax Preparers: Schedule appointments with clients" a 4 out of 5, then the Aw(t) for this task is 4.

2. **Test Data**:
   - **Input**: Assessments from AI experts regarding the technological capability (Ae(t)) and feasible Human Agency Scale level (He(t)) for each task. For example, responses to the question, "How capable is AI of performing this task?" are collected from experts.
   - **Output**: Average technological capability score and expert-assessed Human Agency Scale level for each task. For instance, if an expert rated "Tax Preparers: Schedule appointments with clients" a 3, then the Ae(t) for this task is 3.

This data is stored in the WORKBank database and is used to integrate the perspectives of both workers and experts to evaluate the automation and augmentation potential of AI agents.

<br/>
# 요약


이 논문은 AI 에이전트의 자동화 및 증강 가능성을 평가하기 위해 1,500명의 도메인 근로자와 52명의 AI 전문가의 데이터를 수집하여 WORKBank 데이터베이스를 구축하였다. 결과적으로, 근로자들은 반복적이고 저가치 작업에 대한 자동화를 선호하며, AI의 기술적 능력과 근로자의 욕구 간의 불일치가 발견되었다. 또한, Human Agency Scale(HAS)을 통해 다양한 직업에서 인간의 참여 수준에 대한 선호가 드러났다.

---

This paper developed the WORKBank database by collecting data from 1,500 domain workers and 52 AI experts to assess the automation and augmentation potential of AI agents. The results showed that workers prefer automation for repetitive and low-value tasks, revealing a mismatch between technological capabilities and worker desires. Additionally, the Human Agency Scale (HAS) highlighted diverse preferences for human involvement across various occupations.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **자동화 욕구-능력 풍경 (Figure 5)**: 이 다이어그램은 WORKBank의 작업을 네 가지 영역으로 나누어 보여줍니다: 
     - **자동화 "그린 라이트" 존**: 높은 자동화 욕구와 높은 기술 능력을 가진 작업.
     - **자동화 "레드 라이트" 존**: 높은 기술 능력에도 불구하고 낮은 자동화 욕구를 가진 작업.
     - **R&D 기회 존**: 높은 자동화 욕구를 가지지만 현재 기술 능력이 낮은 작업.
     - **저우선 존**: 낮은 욕구와 낮은 능력을 가진 작업.
   - **인간 에이전시 스케일 (Figure 6)**: 이 스케일은 작업의 인간 참여 수준을 평가하며, 대부분의 직업에서 H3(동등한 파트너십)가 지배적인 수준으로 나타났습니다. 이는 인간과 AI의 협업 가능성을 강조합니다.

2. **테이블**
   - **상위 20개 자동화 욕구 작업 (Table 5)**: 이 테이블은 자동화 욕구가 가장 높은 작업을 나열하고 있으며, 예를 들어 세금 준비자와 공공 안전 통신원과 같은 직업이 포함되어 있습니다. 이들은 반복적이고 낮은 가치의 작업을 자동화하고 싶어하는 경향이 있습니다.
   - **하위 20개 자동화 욕구 작업 (Table 6)**: 이 테이블은 자동화 욕구가 가장 낮은 작업을 나열하고 있으며, 예를 들어 편집자와 비디오 게임 디자이너와 같은 직업이 포함되어 있습니다. 이들은 창의적이고 인간적인 요소가 중요한 작업을 수행하고 있습니다.

3. **어펜딕스**
   - **설문 세부사항 (Appendix A)**: 설문은 오디오 지원 미니 인터뷰와 작업 평가 섹션으로 구성되어 있으며, 작업에 대한 자동화 욕구와 인간 에이전시 수준을 평가합니다.
   - **강건성 분석 (Appendix B)**: AI 전문가의 평가 일치도를 측정하여 데이터의 신뢰성을 높였습니다. 전문가의 평가가 일관되게 이루어졌음을 보여줍니다.
   - **WORKBank 통계 (Appendix D)**: 1,500명의 참가자가 104개 직업에 대해 평가한 결과를 보여주며, 다양한 직업군을 포괄하는 데이터베이스의 대표성을 강조합니다.

### Insights from Diagrams, Figures, Tables, and Appendices

1. **Diagrams and Figures**
   - **Desire-Capability Landscape (Figure 5)**: This diagram categorizes tasks in WORKBank into four zones:
     - **Automation "Green Light" Zone**: Tasks with high automation desire and high technological capability.
     - **Automation "Red Light" Zone**: Tasks with high capability but low desire for automation.
     - **R&D Opportunity Zone**: Tasks with high desire for automation but currently low capability.
     - **Low Priority Zone**: Tasks with both low desire and low capability.
   - **Human Agency Scale (Figure 6)**: This scale assesses the level of human involvement required for tasks, revealing that H3 (equal partnership) is the dominant level across most occupations, highlighting the potential for human-agent collaboration.

2. **Tables**
   - **Top 20 Tasks by Automation Desire (Table 5)**: This table lists tasks with the highest desire for automation, such as Tax Preparers and Public Safety Telecommunicators, indicating a preference for automating repetitive and low-value tasks.
   - **Bottom 20 Tasks by Automation Desire (Table 6)**: This table lists tasks with the lowest desire for automation, including Editors and Video Game Designers, suggesting that these roles value creativity and human elements in their work.

3. **Appendices**
   - **Survey Details (Appendix A)**: The survey consists of an audio-enhanced mini-interview and a task rating section, capturing workers' automation desires and desired levels of human agency.
   - **Robustness Analysis (Appendix B)**: Measures the agreement among AI experts' assessments, ensuring the reliability of the data collected.
   - **WORKBank Statistics (Appendix D)**: Shows that the database includes responses from 1,500 participants across 104 occupations, emphasizing its representativeness across various job sectors.

<br/>
# refer format:
### BibTeX 형식

```bibtex
@article{Shao2026,
  author = {Yijia Shao and Humishka Zope and Yucheng Jiang and Jiaxin Pei and David Nguyen and Erik Brynjolfsson and Diyi Yang},
  title = {Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce},
  journal = {arXiv preprint arXiv:2506.06576},
  year = {2026},




}
```

### 시카고 스타일

Shao, Yijia, Humishka Zope, Yucheng Jiang, Jiaxin Pei, David Nguyen, Erik Brynjolfsson, and Diyi Yang. 2026. "Future of Work with AI Agents: Auditing Automation and Augmentation Potential across the U.S. Workforce." arXiv preprint arXiv:2506.06576. 



