---
layout: post
title:  "[2025]SoRFT: Issue Resolving with Subtask-Oriented Reinforced Fine-Tuning"  
date:   2025-08-03 11:14:40 +0200
categories: study
---

{% highlight ruby %}


한줄 요약: 

강화학습(+파인튜닝)을 통해 LLM의 이슈해결력을 높임(이슈해결 소프트웨어 엔지니어링 데이터셋)  


짧은 요약(Abstract) :    



기존의 이슈 해결 시스템은 상용 대형언어모델(LLM)에 의존하고 있어 비용이 높고 개인정보 유출 우려가 큽니다. 또한 오픈소스 모델을 훈련시키는 기존 방법은 일반화 성능이 낮아 실용성이 떨어집니다. 이에 저자들은 \*\*SoRFT (Subtask-oriented Reinforced Fine-Tuning)\*\*이라는 새로운 학습 방법을 제안합니다. 이 방법은 이슈 해결 과정을 네 가지 하위 작업(파일 위치 찾기, 함수 위치 찾기, 코드 라인 찾기, 코드 수정 생성)으로 분해해 각각의 서브태스크에 대해 학습을 진행합니다.

SoRFT는 다음 두 단계로 구성됩니다:

1. **거절 샘플링 기반의 지도 학습(SFT)**: LLM이 생성한 중간 추론(CoT)을 정답 기반으로 필터링한 뒤 지도 학습 수행.
2. **규칙 기반 강화학습(RL)**: 정답과 비교해 점수를 매기는 규칙을 기반으로 PPO 알고리즘을 사용해 LLM을 업데이트.

SWE-Bench Verified와 SWE-Bench Lite 벤치마크에서 SoRFT로 훈련된 오픈소스 모델은 기존 최고 성능(open-source SOTA)을 달성하며 비용 효율적인 대안임을 입증합니다.

---


> Mainstream issue-resolving frameworks predominantly rely on commercial models, leading to high costs and privacy concerns. Existing training approaches for issue resolving struggle with poor generalization and fail to fully leverage open-source development resources. We propose Subtask-oriented Reinforced Fine-Tuning (SoRFT), a novel training approach to enhance the issue resolving capability of LLMs. We decomposes issue resolving into structured subtasks: file localization, function localization, line localization, and code edit generation. SoRFT consists of two training stages: (1) rejection-sampled supervised fine-tuning, Chain of Thought (CoT) data is filtered using ground-truth before fine-tuning the LLM, and (2) rule-based reinforcement learning, which leverages PPO with ground-truth based rewards. We evaluate the SoRFT-trained model on SWE-Bench Verified and SWE-Bench Lite, achieving state-of-the-art (SOTA) performance among open-source models (e.g., resolve 21.4% issues on SWE-Bench Verified with SoRFT-Qwen-7B). The experimental results demonstrate that SoRFT significantly enhances issue-resolving performance, improves model generalization, and provides a cost-efficient alternative to commercial models.

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






이 논문에서는 LLM 기반 이슈 해결 능력을 향상시키기 위해 **SoRFT (Subtask-oriented Reinforced Fine-Tuning)** 라는 새로운 학습 프레임워크를 제안합니다. 주요 구성 요소는 다음과 같습니다:

1. **서브태스크 분해(Subtask Decomposition)**
   전체 이슈 해결 과정을 다음의 네 가지 세부 작업으로 나눕니다:

   * **파일 위치 찾기 (File Localization)**: 문제 설명을 바탕으로 수정이 필요한 파일을 식별
   * **함수 위치 찾기 (Function Localization)**: 관련 함수나 클래스 등을 세부적으로 지정
   * **라인 위치 찾기 (Line Localization)**: 실제로 수정이 필요한 코드 라인 식별
   * **코드 수정 생성 (Code Edit Generation)**: 수정 전후 코드 블록을 생성 (Search/Replace 형식)

2. **거절 샘플링 기반 지도 학습 (Rejection-Sampled Supervised Fine-Tuning, SFT)**
   Claude-3.5를 활용해 CoT(Chain of Thought) 데이터를 생성하고, 각 서브태스크에 대해 정답과 비교해 잘못된 응답은 제거한 뒤, LLM을 미세 조정합니다.

3. **규칙 기반 강화학습 (Rule-based Reinforcement Learning, PPO 기반)**
   기존의 보상 모델 대신 정답과 비교하여 정의된 규칙 기반 점수(Fβ 점수)를 활용해 강화학습을 수행합니다. 이를 통해 보상 해킹을 방지하고, 일반화 성능을 향상시킵니다.

4. **기반 모델과 프레임워크**

   * 사용된 LLM: Qwen2.5-Coder-7B 및 32B (오픈소스 모델)
   * 프레임워크: Agentless (추론 단계에서 사용), FastChat과 DeepSpeed로 학습 구현
   * 데이터: 공개 GitHub 리포지터리에서 이슈-PR 쌍 30만 개를 기반으로 서브태스크 데이터 구성 (그중 6만\~9만 샘플로 SFT, 3만 샘플로 PPO 학습)

이러한 접근법을 통해 SoRFT는 기존 지도 학습만으로는 얻기 어려운 세밀한 구조 학습과 일반화 능력을 확보합니다.

---



The paper proposes **SoRFT (Subtask-oriented Reinforced Fine-Tuning)**, a novel framework to enhance LLMs’ issue-resolving ability. The method consists of:

1. **Subtask Decomposition**
   The issue-resolving task is broken down into four structured subtasks:

   * **File Localization**: Identify files needing edits based on the issue description.
   * **Function Localization**: Identify relevant functions, methods, or classes.
   * **Line Localization**: Pinpoint exact lines requiring modification.
   * **Code Edit Generation**: Produce code edits in a *Search/Replace* format using context.

2. **Rejection-Sampled Supervised Fine-Tuning (SFT)**
   Chain-of-Thought (CoT) reasoning traces are generated using Claude-3.5. Only samples with overlap to ground truth are kept, improving reasoning structure and output formatting for each subtask.

3. **Rule-based Reinforcement Learning (PPO)**
   Instead of using a learned reward model, the authors define **rule-based rewards** using the **Fβ score** between model outputs and ground truth, avoiding reward hacking and improving generalization.

4. **Model and Framework Setup**

   * **Base LLMs**: Qwen2.5-Coder-7B and 32B (open-source instruction-tuned models)
   * **Frameworks**: Agentless (inference), FastChat and DeepSpeed (training)
   * **Training Data**: 300k (issue, PR) pairs from open-source GitHub repos. Of these, 60–90k samples are used for SFT, and 30k for PPO fine-tuning.

This design allows SoRFT to effectively learn task structure, mitigate hallucinations, and generalize better than purely supervised methods.

---



   
 
<br/>
# Results  





이 논문은 SoRFT 학습을 거친 LLM이 **이슈 해결 성능**에서 기존 오픈소스 모델들보다 뛰어난 성과를 보인다고 보고합니다. 주요 실험 조건과 결과는 다음과 같습니다:

####  사용한 벤치마크

* **SWE-Bench Verified**: 사람이 검증한 고품질 GitHub 이슈 500개를 포함한 데이터셋 (도커 테스트 포함)
* **SWE-Bench Lite**: 기능성 버그 수정에 초점을 둔 300개 이슈 포함

####  평가 메트릭

* **%Resolved**: 생성된 코드 수정이 모든 테스트 케이스를 통과한 비율
* **%Applied**: 코드 수정이 리포지터리에 적용 가능한 형태로 생성된 비율

####  주요 비교 결과

| 모델                       | 프레임워크        | SWE-Bench Verified (%) | SWE-Bench Lite (%) |
| ------------------------ | ------------ | ---------------------- | ------------------ |
| **SoRFT-Qwen-7B (제안모델)** | Agentless    | **21.4**               | **14.0**           |
| SWE-Gym-Qwen-7B          | OpenHands    | 10.6                   | 10.0               |
| Lingma-SWE-GPT-7B        | SWE-SynInfer | 18.2                   | 12.0               |
| SWE-Fixer-Qwen-72B       | SWE-Fixer    | 30.2                   | 23.3               |
| **SoRFT-Qwen-32B**       | Agentless    | **30.8**               | **24.0**           |

* **SoRFT-Qwen-7B**는 **동일 파라미터 크기** 모델 중에서 최고 성능을 보였으며, 심지어 \*\*32B 모델(SWE-Gym-Qwen-32B)\*\*보다도 높은 성능을 기록했습니다.
* **SoRFT-Qwen-32B**는 \*\*72B 초대형 모델(Lingma-SWE-GPT-72B)\*\*보다도 성능이 높았습니다.

####  추가 분석 결과

* SoRFT는 단순한 지도학습보다 성능이 높고, 일반화 능력도 향상됨
* Fβ 스코어 기반의 보상 규칙이 가장 성능이 좋음
* 파일, 함수, 라인 로컬라이제이션 서브태스크 모두에서 정확도 상승

---



The paper evaluates the performance of SoRFT-trained LLMs on issue resolution tasks and compares them against various open-source and proprietary models.

####  Benchmarks Used

* **SWE-Bench Verified**: 500 high-quality GitHub issues with Docker-based test verification.
* **SWE-Bench Lite**: 300 functional bug-fix issues.

####  Evaluation Metrics

* **%Resolved**: Percentage of issues where generated code edits pass all test cases.
* **%Applied**: Percentage of issues where code edits were syntactically and semantically applicable.

####  Key Performance Comparison

| Model                    | Framework    | SWE-Bench Verified (%) | SWE-Bench Lite (%) |
| ------------------------ | ------------ | ---------------------- | ------------------ |
| **SoRFT-Qwen-7B (Ours)** | Agentless    | **21.4**               | **14.0**           |
| SWE-Gym-Qwen-7B          | OpenHands    | 10.6                   | 10.0               |
| Lingma-SWE-GPT-7B        | SWE-SynInfer | 18.2                   | 12.0               |
| SWE-Fixer-Qwen-72B       | SWE-Fixer    | 30.2                   | 23.3               |
| **SoRFT-Qwen-32B**       | Agentless    | **30.8**               | **24.0**           |

* **SoRFT-Qwen-7B** achieved **best-in-class performance among 7B models**, outperforming even some 32B models.
* **SoRFT-Qwen-32B** outperformed **Lingma-SWE-GPT-72B**, a much larger model, demonstrating the method's efficiency.

####  Additional Insights

* SoRFT yields higher %Resolved and %Applied compared to supervised fine-tuning alone.
* **Fβ-based reward rules** lead to better reinforcement learning outcomes.
* Accuracy on subtasks (file/function/line localization) improved significantly with SoRFT.

---




<br/>
# 예제  





이 논문에서는 LLM에게 GitHub 이슈를 해결하도록 학습시키기 위해, 하나의 문제를 4개의 \*\*서브태스크(subtask)\*\*로 분해합니다. 각각의 서브태스크에는 고유한 입력과 출력이 있으며, 모두 실제 오픈소스 GitHub 이슈와 PR 데이터를 기반으로 구성되었습니다.

####  트레이닝 데이터 (입력/출력 예시)

1. **파일 위치 찾기 (File Localization)**

   * **입력**: GitHub 이슈 설명 + 프로젝트의 파일 구조 (repository skeleton)
   * **출력**: 수정이 필요한 파일 경로 리스트 (예: `conda_build/metadata.py`, `conda_build/render.py`)

2. **함수 위치 찾기 (Function Localization)**

   * **입력**: 이슈 설명 + 해당 파일의 함수 스켈레톤 (함수/클래스 이름들)
   * **출력**: 관련 함수 또는 클래스 이름 (예: `function: sql_flush`, `class: BaseDatabaseOperations`)

3. **라인 위치 찾기 (Line Localization)**

   * **입력**: 이슈 설명 + 함수 내용 (소스 코드)
   * **출력**: 수정이 필요한 라인 번호 또는 코드 블록

4. **코드 수정 생성 (Code Edit Generation)**

   * **입력**: 이슈 설명 + 관련 코드
   * **출력**: 수정 전 코드와 수정 후 코드를 나타내는 **SEARCH/REPLACE** 블록

     ```python
     ### conda_build/metadata.py
     <<<<<< SEARCH
     version = ".1"
     =======
     version = "0.1"
     >>>>>> REPLACE
     ```

####  테스트 데이터

* **SWE-Bench Verified**와 **SWE-Bench Lite**에서 제공되는 이슈는 다음과 같이 구성됨:

  * **이슈 설명** (자연어로 된 버그 리포트 또는 기능 요청)
  * **전체 리포지터리** 코드 베이스
  * **정답 PR**: 실제로 문제를 해결한 수정 코드 (코드 라인, 함수, 파일 포함)
  * **도커 기반 테스트 스크립트**: 생성된 수정이 실제로 문제를 해결하는지 검증

---



The paper defines four structured **subtasks** for issue resolution, with each subtask designed using real-world GitHub issues and pull requests from open-source projects. Each subtask has distinct inputs and outputs.

####  Training Data (Input/Output Examples)

1. **File Localization**

   * **Input**: GitHub issue description + repository skeleton (list of file paths)
   * **Output**: List of modified file paths (e.g., `conda_build/metadata.py`, `conda_build/render.py`)

2. **Function Localization**

   * **Input**: Issue description + function/class skeleton of the file
   * **Output**: Relevant function or class names (e.g., `function: sql_flush`, `class: BaseDatabaseOperations`)

3. **Line Localization**

   * **Input**: Issue description + function body (source code)
   * **Output**: Specific line numbers or code locations to edit

4. **Code Edit Generation**

   * **Input**: Issue description + localized code snippet
   * **Output**: Code patch in a **SEARCH/REPLACE** format

     ```python
     ### conda_build/metadata.py
     <<<<<< SEARCH
     version = ".1"
     =======
     version = "0.1"
     >>>>>> REPLACE
     ```

####  Test Data

* Comes from **SWE-Bench Verified** and **SWE-Bench Lite**, each issue includes:

  * **Issue description**: A real bug report or feature request in natural language
  * **Full repository codebase**
  * **Ground-truth pull request**: The actual code changes made to resolve the issue
  * **Docker-based test suite**: Automatically verifies whether the generated patch resolves the issue

---




<br/>  
# 요약   





SoRFT는 이슈 해결을 파일/함수/라인/코드수정 네 가지 서브태스크로 나눠 지도학습과 규칙 기반 강화학습(PPO)을 결합해 LLM을 훈련하는 방식이다. SWE-Bench Verified와 Lite 데이터셋에서 SoRFT-Qwen은 동급 모델 중 최고 성능을 기록했으며, 일부 대형 모델보다도 높은 %Resolved 성능을 보였다. 예시는 GitHub 이슈 설명과 리포지터리 구조를 입력으로 받아, 수정할 파일 경로, 함수 이름, 라인 번호, 코드 패치를 생성하는 형식으로 구성된다.

---



SoRFT trains LLMs by decomposing issue resolution into four subtasks—file, function, line localization, and code edit generation—using a combination of supervised learning and rule-based reinforcement learning with PPO. On SWE-Bench Verified and Lite, SoRFT-Qwen outperforms peer models and even surpasses larger models in terms of %Resolved. Input examples consist of GitHub issue descriptions and repository structure, with outputs including file paths, function names, line numbers, and SEARCH/REPLACE style code edits.

---


<br/>  
# 기타  





####  Figure 2 (전체 SoRFT 구조 다이어그램)

* 이슈 해결을 **4단계 서브태스크**(파일→함수→라인→수정)로 나누고,
  각 단계별로 \*\*거절 샘플링 기반 지도학습 + 규칙기반 강화학습(PPO)\*\*을 적용하는 전체 구조를 시각화함.
* 인사이트: 복잡한 이슈 해결 과정을 다단계 학습으로 세분화함으로써, 각 단계에서의 정확도를 향상시킬 수 있다는 점을 강조.

####  Table 1 (경쟁 모델들과 성능 비교)

* SoRFT-Qwen-7B는 SWE-Gym-32B, Lingma-7B보다 높은 성능을 기록하고,
  SoRFT-Qwen-32B는 Lingma-72B보다도 %Resolved 기준으로 우수함.
* 인사이트: SoRFT는 모델 크기 대비 성능 효율이 매우 뛰어남을 보여주며, 오픈소스 모델 중 SOTA 달성.

####  Table 2 (훈련 전략별 성능 비교)

* SFT만 수행한 모델보다 SFT+RL을 적용한 SoRFT가 %Resolved와 %Applied 모두에서 성능이 향상됨.
* 인사이트: 단순한 지도학습보다 강화학습을 결합했을 때 일반화 성능 및 실제 적용 가능성이 모두 높아짐.

####  Table 3 (서브태스크별 정확도 향상)

* 파일/함수/라인 위치 식별 정확도가 SoRFT 학습을 통해 모두 향상됨 (예: 파일 히트율 59.8%→77.8%).
* 인사이트: 서브태스크 기반 학습 전략이 각 단계별 정확도를 개별적으로도 개선시킨다는 실증적 증거.

####  Table 4 (보상 규칙별 성능 비교)

* F1 또는 Fβ 점수 기반 보상 규칙이 binary hit/exact-match보다 강화학습 성능이 높음.
* 인사이트: 정밀도-재현율 균형을 고려한 보상 설계가 RL 안정성과 수렴 속도에 긍정적 영향을 줌.

####  Figure 3 (훈련 데이터 양과 전략에 따른 변화)

* SFT 데이터가 많아져도 %Resolved는 소폭 증가하지만 %Applied는 오히려 하락 → 과적합 우려.
* SoRFT는 더 적은 데이터로도 더 높은 성능 달성.
* 인사이트: SoRFT는 데이터 효율성과 일반화 면에서 우수함.

####  Appendix D (서브태스크별 프롬프트 예시)

* 각 서브태스크에 대해 문제 설명, 컨텍스트, 출력 포맷을 명확히 지정한 Chain-of-Thought 기반 프롬프트 예시 제공.
* 인사이트: 체계적 프롬프트 설계가 LLM의 추론 과정 학습에 핵심적인 역할을 함.

---


####  Figure 2 (SoRFT Architecture Diagram)

* Illustrates the full pipeline: decomposing issue resolution into four subtasks (file, function, line, code edit), followed by rejection-sampled SFT and rule-based RL for each.
* Insight: Breaking the task into subtasks enables precise reasoning and targeted learning for each stage.

####  Table 1 (Comparison with Baselines)

* SoRFT-Qwen-7B outperforms SWE-Gym-32B and Lingma-7B, while SoRFT-Qwen-32B surpasses Lingma-72B.
* Insight: SoRFT delivers state-of-the-art performance among open-source models, even beating larger models.

####  Table 2 (Training Strategy Comparison)

* Models trained with SoRFT (SFT + RL) show higher %Resolved and %Applied than SFT-only.
* Insight: Reinforcement learning improves both generalization and practical applicability of the model.

####  Table 3 (Subtask Accuracy)

* SoRFT improves accuracy on file/function/line localization subtasks significantly (e.g., file hit: 59.8%→77.8%).
* Insight: Subtask decomposition is not just conceptually sound but empirically effective for boosting precision.

####  Table 4 (Reward Rule Ablation)

* F1 and Fβ score-based rewards outperform binary hit or exact-match metrics.
* Insight: Balanced precision-recall based rewards yield better RL convergence and model stability.

####  Figure 3 (Training Data vs. Performance)

* Adding more data to SFT slightly improves %Resolved but decreases %Applied → possible overfitting.
* SoRFT achieves better results with fewer examples.
* Insight: SoRFT is more data-efficient and generalizable than plain SFT.

####  Appendix D (Prompt Templates for Subtasks)

* Provides detailed CoT-style prompts for each subtask, including instructions, inputs, and answer format.
* Insight: Structured prompting is key to enabling reasoning and accurate LLM fine-tuning.

---


<br/>
# refer format:     


@inproceedings{ma2025sorft,
  title     = {SoRFT: Issue Resolving with Subtask-oriented Reinforced Fine-Tuning},
  author    = {Zexiong Ma and Chao Peng and Pengfei Gao and Xiangxin Meng and Yanzhen Zou and Bing Xie},
  booktitle = {Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages     = {11427--11441},
  year      = {2025},
  month     = jul,
  address   = {Bangkok, Thailand},
  publisher = {Association for Computational Linguistics}
}



Ma, Zexiong, Chao Peng, Pengfei Gao, Xiangxin Meng, Yanzhen Zou, and Bing Xie. “SoRFT: Issue Resolving with Subtask-Oriented Reinforced Fine-Tuning.” In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 11427–11441. Bangkok, Thailand: Association for Computational Linguistics, July 2025.




