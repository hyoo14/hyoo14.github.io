---
layout: post
title:  "[2025]Recursive Language Models"
date:   2026-01-17 18:38:38 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 Recursive Language Models (RLMs)라는 새로운 추론 전략을 제안하여, 대형 언어 모델이 긴 프롬프트를 효과적으로 처리할 수 있도록 한다.


전략은 LLM이 프로그램적(파이썬을 사용하여)으로 프롬프트의 조각을 검사하고 분해하며 재귀적으로 자신을 호출할 수 있도록


짧은 요약(Abstract) :


이 논문에서는 대형 언어 모델(LLM)이 임의로 긴 프롬프트를 처리할 수 있도록 하는 방법을 연구합니다. 저자들은 Recursive Language Models(RLMs)라는 일반적인 추론 전략을 제안하며, 이는 긴 프롬프트를 외부 환경의 일부로 간주하고 LLM이 프로그램적으로 이를 검사하고 분해하며 재귀적으로 호출할 수 있도록 합니다. 연구 결과, RLM은 모델의 컨텍스트 윈도우를 두 배 이상 초과하는 입력을 성공적으로 처리할 수 있으며, 짧은 프롬프트에서도 기본 LLM 및 일반적인 긴 컨텍스트 스캐폴드보다 성능이 크게 향상됩니다. 이 방법은 네 가지 다양한 긴 컨텍스트 작업에서 검증되었으며, 비용 측면에서도 유사하거나 더 저렴한 쿼리 비용을 유지합니다.



This paper studies the ability of large language models (LLMs) to process arbitrarily long prompts through the lens of inference-time scaling. The authors propose Recursive Language Models (RLMs), a general inference strategy that treats long prompts as part of an external environment, allowing the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. The findings indicate that RLMs can successfully handle inputs up to two orders of magnitude beyond the model's context windows and, even for shorter prompts, dramatically outperform the quality of base LLMs and common long-context scaffolds across four diverse long-context tasks, while maintaining comparable (or cheaper) cost per query.


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



**메써드: 재귀 언어 모델 (Recursive Language Models, RLMs)**

재귀 언어 모델(RLMs)은 대규모 언어 모델(LLMs)이 임의의 길이의 프롬프트를 처리할 수 있도록 설계된 일반적인 추론 전략입니다. 이 방법은 긴 프롬프트를 외부 환경의 일부로 간주하고, LLM이 프로그램적으로 프롬프트의 조각을 검사하고 분해하며 재귀적으로 자신을 호출할 수 있도록 합니다. RLMs는 모델의 컨텍스트 윈도우를 초과하는 입력을 효과적으로 처리할 수 있으며, 짧은 프롬프트에 대해서도 기본 LLM 및 일반적인 긴 컨텍스트 스캐폴드보다 품질이 크게 향상됩니다.

RLMs의 핵심 아이디어는 긴 프롬프트를 신경망(예: Transformer)에 직접 입력하는 대신, LLM이 상징적으로 상호작용할 수 있는 외부 환경의 객체로 취급하는 것입니다. RLM은 Python REPL(Read-Eval-Print Loop) 환경을 초기화하고, 프롬프트를 변수로 설정하여 LLM이 이 변수를 통해 프롬프트를 탐색하고 조작할 수 있도록 합니다. 이 과정에서 LLM은 코드 작성을 통해 프롬프트의 세부 정보를 관찰하고, 이를 기반으로 재귀적으로 하위 작업을 생성하여 자신을 호출합니다.

RLMs는 다양한 복잡성을 가진 여러 작업에서 평가되었으며, 특히 정보 밀도가 높은 작업에서 뛰어난 성능을 보였습니다. RLMs는 입력 컨텍스트의 길이에 따라 성능이 저하되는 기존 LLM의 한계를 극복하고, 더 긴 입력을 처리할 수 있는 능력을 제공합니다. 이 방법은 LLM의 추론 능력을 확장하고, 긴 문서나 대규모 데이터셋을 처리하는 데 있어 효율성을 높이는 데 기여합니다.



**Method: Recursive Language Models (RLMs)**

Recursive Language Models (RLMs) are a general inference strategy designed to enable large language models (LLMs) to process arbitrarily long prompts. This method treats long prompts as part of an external environment, allowing the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. RLMs effectively handle inputs that exceed the model's context window and significantly improve the quality of responses compared to base LLMs and common long-context scaffolds, even for shorter prompts.

The key insight of RLMs is that long prompts should not be fed directly into the neural network (e.g., Transformer) but should instead be treated as objects in an external environment that the LLM can symbolically interact with. RLMs initialize a Python Read-Eval-Print Loop (REPL) environment, setting the prompt as a variable, which allows the LLM to explore and manipulate the prompt through this variable. In this process, the LLM writes code to peek into the details of the prompt and iteratively constructs sub-tasks on which it can invoke itself recursively.

RLMs have been evaluated across various tasks with differing levels of complexity, demonstrating exceptional performance, particularly on information-dense tasks. They overcome the limitations of existing LLMs, which degrade in performance as input length increases, and provide the capability to handle longer inputs. This method contributes to extending the reasoning capabilities of LLMs and enhances efficiency in processing long documents or large datasets.


<br/>
# Results



이 논문에서는 Recursive Language Models (RLMs)라는 새로운 추론 전략을 제안하고, 이를 통해 대규모 언어 모델(LLMs)이 임의의 길이의 프롬프트를 처리할 수 있는 가능성을 탐구합니다. RLM은 긴 프롬프트를 외부 환경의 일부로 간주하고, LLM이 프로그램적으로 프롬프트의 조각을 검사하고 분해하며 재귀적으로 자신을 호출할 수 있도록 합니다. 연구 결과, RLM은 모델의 컨텍스트 윈도우를 초과하는 입력을 성공적으로 처리할 수 있으며, 짧은 프롬프트에서도 기본 LLM 및 일반적인 긴 컨텍스트 스캐폴드보다 성능이 크게 향상되었습니다.

#### 결과 요약
1. **경쟁 모델**: RLM은 GPT-5와 Qwen3-Coder-480B-A35B와 같은 최신 모델을 사용하여 평가되었습니다. RLM은 이들 모델의 성능을 크게 초월하는 결과를 보였습니다.
  
2. **테스트 데이터**: RLM은 S-NIAH, OOLONG, OOLONG-Pairs와 같은 다양한 긴 컨텍스트 작업을 포함한 여러 벤치마크에서 평가되었습니다. 각 작업은 입력 길이에 따라 다르게 설계되었습니다.

3. **메트릭**: 성능은 정확도, F1 점수, 평균 API 비용 등 다양한 메트릭을 통해 평가되었습니다. RLM은 특히 OOLONG-Pairs와 같은 정보 밀도가 높은 작업에서 두드러진 성능을 보였습니다.

4. **비교**: RLM은 기본 LLM 및 다른 일반적인 긴 컨텍스트 처리 방법(예: 요약 에이전트, CodeAct + BM25)과 비교되었습니다. RLM은 모든 작업에서 성능이 두 배 이상 향상되었으며, 비용 측면에서도 유사하거나 더 저렴한 결과를 보였습니다.

5. **관찰 결과**: RLM은 10M+ 토큰 규모에서도 강력한 성능을 발휘하며, 긴 컨텍스트 작업에서 기존 방법들보다 두 자릿수 퍼센트의 성능 향상을 보였습니다. 특히, RLM은 복잡한 작업에서 성능 저하가 적고, 입력 길이에 따른 성능 저하가 느리게 나타났습니다.




This paper introduces a new inference strategy called Recursive Language Models (RLMs) and explores the potential for large language models (LLMs) to process arbitrarily long prompts. RLM treats long prompts as part of an external environment, allowing the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. The study finds that RLMs successfully handle inputs beyond the model's context window and significantly outperform base LLMs and common long-context scaffolds even for shorter prompts.

#### Summary of Results
1. **Competing Models**: RLM was evaluated using state-of-the-art models such as GPT-5 and Qwen3-Coder-480B-A35B. The results showed that RLM significantly surpassed the performance of these models.

2. **Test Data**: RLM was assessed across various long-context tasks, including S-NIAH, OOLONG, and OOLONG-Pairs. Each task was designed to vary based on input length.

3. **Metrics**: Performance was evaluated using various metrics, including accuracy, F1 scores, and average API costs. RLM demonstrated particularly strong performance on information-dense tasks like OOLONG-Pairs.

4. **Comparison**: RLM was compared against base LLMs and other common long-context processing methods (e.g., summarization agents, CodeAct + BM25). RLM showed performance improvements of up to two times across all tasks while maintaining comparable or lower costs.

5. **Observations**: RLMs exhibited strong performance even at the 10M+ token scale, outperforming existing methods in long-context processing by double-digit percentage gains. Notably, RLMs showed less severe degradation in performance for complex tasks and exhibited slower performance degradation as input length increased.


<br/>
# 예제



이 논문에서는 Recursive Language Models (RLMs)라는 새로운 추론 전략을 제안합니다. RLM은 긴 프롬프트를 외부 환경의 일부로 처리하여 LLM이 프로그램적으로 프롬프트의 조각을 검사하고 분해하며 재귀적으로 자신을 호출할 수 있도록 합니다. 이 방법은 LLM이 최대 10M+ 토큰의 입력을 처리할 수 있게 하며, 다양한 긴 컨텍스트 작업에서 기존 LLM 및 일반적인 긴 컨텍스트 스캐폴드보다 성능이 뛰어납니다.

#### 예시: OOLONG-Pairs 작업

**테스크 설명**: OOLONG-Pairs는 정보 밀도가 높은 긴 입력을 처리하는 작업으로, 주어진 데이터에서 특정 조건을 만족하는 사용자 ID 쌍을 찾는 것입니다. 예를 들어, "두 사용자 모두 적어도 하나의 인스턴스가 설명 및 추상 개념 또는 약어를 포함하는 사용자 ID 쌍을 나열하라"는 질문이 주어질 수 있습니다.

**입력 데이터**: 
- 사용자 ID와 관련된 여러 질문이 포함된 데이터셋이 있습니다. 각 질문은 특정 카테고리(예: 설명 및 추상 개념, 인간, 숫자 값 등)로 분류될 수 있습니다.

**입력 예시**:
```
1. (user_id: 101, question: "이 사람은 누구인가?")
2. (user_id: 102, question: "이 사람의 직업은 무엇인가?")
3. (user_id: 103, question: "이 사람의 나이는 몇 살인가?")
```

**출력 데이터**: 
- 조건을 만족하는 사용자 ID 쌍의 리스트가 출력됩니다. 예를 들어, 사용자 101과 102가 조건을 만족한다면 출력은 다음과 같습니다.
```
(101, 102)
```

이와 같은 방식으로 RLM은 긴 입력을 처리하고, 재귀적으로 자신을 호출하여 필요한 정보를 추출하고, 최종적으로 조건을 만족하는 사용자 ID 쌍을 반환합니다.

---



This paper introduces a new inference strategy called Recursive Language Models (RLMs). RLM allows large language models (LLMs) to treat long prompts as part of an external environment, enabling the LLM to programmatically examine, decompose, and recursively call itself over snippets of the prompt. This method allows LLMs to handle inputs of up to 10M+ tokens and significantly outperforms existing LLMs and common long-context scaffolds across various long-context tasks.

#### Example: OOLONG-Pairs Task

**Task Description**: OOLONG-Pairs is a task that involves processing information-dense long inputs to find pairs of user IDs that meet specific conditions. For instance, a question might be posed as "List all pairs of user IDs where both users have at least one instance with a description and abstract concept or abbreviation."

**Input Data**: 
- A dataset containing multiple questions associated with user IDs. Each question can be categorized into specific categories (e.g., description and abstract concept, human being, numeric value, etc.).

**Input Example**:
```
1. (user_id: 101, question: "Who is this person?")
2. (user_id: 102, question: "What is this person's occupation?")
3. (user_id: 103, question: "How old is this person?")
```

**Output Data**: 
- A list of user ID pairs that satisfy the conditions. For example, if users 101 and 102 meet the criteria, the output would be:
```
(101, 102)
```

In this way, RLM processes long inputs, recursively calls itself to extract necessary information, and ultimately returns pairs of user IDs that meet the specified conditions.

<br/>
# 요약
이 논문에서는 Recursive Language Models (RLMs)라는 새로운 추론 전략을 제안하여, 대형 언어 모델이 긴 프롬프트를 효과적으로 처리할 수 있도록 한다. RLM은 입력 프롬프트를 외부 환경의 일부로 간주하고, 이를 통해 모델이 재귀적으로 자신을 호출하여 정보를 처리할 수 있게 한다. 실험 결과, RLM은 기존 모델보다 성능이 두 배 향상되었으며, 긴 문맥 처리에서 우수한 결과를 보였다.

---

This paper introduces a new inference strategy called Recursive Language Models (RLMs) that enables large language models to effectively handle long prompts. RLM treats the input prompt as part of an external environment, allowing the model to recursively call itself to process information. Experimental results show that RLM outperforms existing models by up to two times and demonstrates superior performance in long-context processing.

<br/>
# 기타



#### 다이어그램 및 피규어
1. **Figure 1**: RLM과 GPT-5의 성능 비교
   - RLM은 입력 길이가 증가함에 따라 성능 저하가 적고, 복잡한 작업에서도 안정적인 성능을 유지함.
   - GPT-5는 입력 길이와 작업 복잡성에 따라 성능이 급격히 저하됨.

2. **Figure 2**: RLM의 작동 방식
   - RLM은 입력 프롬프트를 외부 환경의 변수로 처리하고, 이를 통해 코드 실행 및 재귀 호출을 통해 문제를 해결함.

3. **Figure 3**: RLM과 다른 방법들의 비용 비교
   - RLM은 평균적으로 비슷하거나 더 낮은 비용으로 성능을 유지하며, 특히 긴 입력에 대해 더 효율적임.

4. **Figure 4**: RLM의 일반적인 작업 패턴
   - RLM은 코드 실행을 통해 입력 정보를 필터링하고, 재귀 호출을 통해 문제를 해결하는 경향이 있음.

#### 테이블
1. **Table 1**: 다양한 방법의 성능 비교
   - RLM은 모든 작업에서 기본 모델과 다른 방법들보다 우수한 성능을 보임.
   - 특히 OOLONG 및 OOLONG-Pairs 작업에서 RLM이 두 배 이상의 성능 향상을 보임.

#### 어펜딕스
- **Appendix C**: RLM의 런타임 및 비용 분석
  - RLM의 런타임은 비동기 호출을 통해 개선될 수 있으며, 높은 변동성을 보임.
  
- **Appendix D**: 실험에 사용된 프롬프트
  - RLM의 프롬프트는 모델에 따라 약간의 차이가 있으며, Qwen3-Coder의 경우 재귀 호출을 최소화하도록 경고하는 문구가 추가됨.



#### Diagrams and Figures
1. **Figure 1**: Performance comparison between RLM and GPT-5
   - RLM shows less performance degradation as input length increases and maintains stable performance even on complex tasks.
   - GPT-5 experiences significant performance drops based on input length and task complexity.

2. **Figure 2**: How RLM operates
   - RLM treats the input prompt as a variable in an external environment, allowing it to solve problems through code execution and recursive calls.

3. **Figure 3**: Cost comparison between RLM and other methods
   - RLM maintains performance at comparable or lower costs, especially for long inputs, demonstrating greater efficiency.

4. **Figure 4**: Common patterns in RLM operations
   - RLM tends to filter input information through code execution and solve problems using recursive calls.

#### Tables
1. **Table 1**: Performance comparison of various methods
   - RLM outperforms base models and other methods across all tasks.
   - Notably, RLM shows more than double the performance improvement on OOLONG and OOLONG-Pairs tasks.

#### Appendix
- **Appendix C**: Runtime and cost analysis of RLM
  - RLM's runtime can be improved through asynchronous calls, showing high variance.
  
- **Appendix D**: Prompts used in experiments
  - The prompts for RLM vary slightly by model, with an additional warning for Qwen3-Coder to minimize recursive calls.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@article{zhang2025recursive,
  title={Recursive Language Models},
  author={Zhang, Alex L. and Kraska, Tim and Khattab, Omar},
  journal={arXiv preprint arXiv:2512.24601},
  year={2025},
  url={https://arxiv.org/abs/2512.24601}
}
```

### 시카고 스타일
Alex L. Zhang, Tim Kraska, and Omar Khattab. "Recursive Language Models." arXiv preprint arXiv:2512.24601 (2025). https://arxiv.org/abs/2512.24601.
