---
layout: post
title:  "[2025]Deep Researcher with Test-Time Diffusion"  
date:   2025-08-13 03:00:56 -0000
categories: study
---

{% highlight ruby %}

한줄 요약:

반복적인 사이클을 통해 보고서를 작성합니다. TTD-DR은 초기 초안을 생성하고, 이 초안을 기반으로 외부 정보를 검색하여 점진적으로 "노이즈 제거"를 통해 초안을 개선  


짧은 요약(Abstract) :    


이 논문에서는 대형 언어 모델(LLM)을 기반으로 한 심층 연구 에이전트의 성능을 향상시키기 위해 테스트 시간 확산(Test-Time Diffusion)이라는 새로운 프레임워크를 제안합니다. 기존의 심층 연구 에이전트는 복잡하고 긴 형태의 연구 보고서를 생성할 때 성능이 정체되는 경향이 있습니다. 이를 해결하기 위해, 인간의 연구 과정에서 영감을 받아 연구 보고서 생성을 확산 과정으로 개념화한 Test-Time Diffusion Deep Researcher(TTD-DR)를 제안합니다. TTD-DR은 초기 초안을 생성하고, 이를 지속적으로 업데이트하여 연구 방향을 안내하는 기초로 사용합니다. 이 초안은 외부 정보를 통합하는 검색 메커니즘에 의해 동적으로 정보가 제공되며, "노이즈 제거" 과정을 통해 반복적으로 개선됩니다. 또한, 에이전트 워크플로우의 각 구성 요소에 자체 진화 알고리즘을 적용하여 확산 과정에 고품질의 컨텍스트를 생성하도록 보장합니다. 이 초안 중심의 설계는 보고서 작성 과정을 더 시기적절하고 일관되게 만들며, 반복적인 검색 과정에서 정보 손실을 줄입니다. TTD-DR은 집중적인 검색과 다중 단계 추론이 필요한 다양한 벤치마크에서 최첨단 결과를 달성하며, 기존의 심층 연구 에이전트를 크게 능가합니다.



In this paper, we propose a novel framework called Test-Time Diffusion to enhance the performance of deep research agents powered by Large Language Models (LLMs). Existing deep research agents often plateau in performance when generating complex, long-form research reports. To address this, we draw inspiration from the iterative nature of human research and conceptualize research report generation as a diffusion process, introducing the Test-Time Diffusion Deep Researcher (TTD-DR). TTD-DR initiates with a preliminary draft, serving as an evolving foundation to guide research direction. This draft is iteratively refined through a "denoising" process, dynamically informed by a retrieval mechanism that incorporates external information at each step. The core process is further enhanced by a self-evolutionary algorithm applied to each component of the agentic workflow, ensuring the generation of high-quality context for the diffusion process. This draft-centric design makes the report writing process more timely and coherent while reducing information loss during the iterative search process. We demonstrate that our TTD-DR achieves state-of-the-art results on a wide array of benchmarks that require intensive search and multi-hop reasoning, significantly outperforming existing deep research agents.


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


이 논문에서는 Test-Time Diffusion Deep Researcher (TTD-DR)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 복잡한 연구 보고서 생성을 확산 과정으로 개념화하여, 인간의 연구 과정과 유사한 반복적인 사이클을 통해 보고서를 작성합니다. TTD-DR은 초기 초안을 생성하고, 이 초안을 기반으로 외부 정보를 검색하여 점진적으로 "노이즈 제거"를 통해 초안을 개선합니다. 이 과정은 검색 도구를 통해 외부 정보를 통합하여 정확성과 포괄성을 높입니다. 또한, 각 구성 요소에 대해 자기 진화 알고리즘을 적용하여, 각 단계에서 고품질의 컨텍스트를 생성하도록 합니다. 이 프레임워크는 정보 손실을 줄이고, 더 시기적절하고 일관된 보고서를 생성할 수 있도록 설계되었습니다.



The paper proposes a novel framework called Test-Time Diffusion Deep Researcher (TTD-DR). This framework conceptualizes the generation of complex research reports as a diffusion process, mimicking the iterative cycles of human research. TTD-DR starts with generating an initial draft, which is progressively refined through a "denoising" process informed by external information retrieval. This process integrates external information via search tools to enhance accuracy and comprehensiveness. Additionally, a self-evolutionary algorithm is applied to each component to ensure the generation of high-quality context at each step. The framework is designed to reduce information loss and produce more timely and coherent reports.


<br/>
# Results  


이 논문에서는 Test-Time Diffusion Deep Researcher (TTD-DR)라는 새로운 프레임워크를 제안하여 복잡한 연구 보고서를 생성하는 데 있어 기존의 DR(Deep Research) 에이전트의 한계를 극복하고자 합니다. TTD-DR은 연구 보고서 생성을 확산 과정으로 개념화하여, 초기 초안을 작성하고 이를 반복적으로 "디노이징"하여 외부 정보를 통합하는 방식으로 최종 보고서를 완성합니다. 이 과정은 각 구성 요소에 대한 자기 진화 알고리즘을 통해 강화됩니다.

#### 주요 결과:
- **성능 비교**: TTD-DR은 다양한 벤치마크에서 기존의 DR 에이전트를 능가하는 성능을 보였습니다. 특히, OpenAI Deep Research와 비교했을 때 LongForm Research와 DeepConsult에서 각각 69.1%와 74.5%의 승률을 기록했습니다.
- **정확성**: HLE-Search와 HLE-Full 데이터셋에서 OpenAI Deep Research보다 각각 4.8%와 7.7% 높은 정확성을 보였습니다.
- **효율성**: TTD-DR은 테스트 시간 확장에 있어 효율적인 알고리즘으로, 더 많은 검색 및 수정 단계를 추가함으로써 성능을 향상시켰습니다.



This paper introduces a novel framework called Test-Time Diffusion Deep Researcher (TTD-DR) to overcome the limitations of existing DR (Deep Research) agents in generating complex research reports. TTD-DR conceptualizes the generation of research reports as a diffusion process, starting with an initial draft and iteratively "denoising" it by incorporating external information. This process is further enhanced by a self-evolutionary algorithm applied to each component.

#### Key Results:
- **Performance Comparison**: TTD-DR consistently outperformed existing DR agents across various benchmarks. Specifically, it achieved a win rate of 69.1% and 74.5% in LongForm Research and DeepConsult, respectively, compared to OpenAI Deep Research.
- **Accuracy**: It showed an improvement of 4.8% and 7.7% in accuracy over OpenAI Deep Research on the HLE-Search and HLE-Full datasets, respectively.
- **Efficiency**: TTD-DR demonstrated efficient algorithms for test-time scaling, achieving performance gains by adding more search and revision steps.


<br/>
# 예제  
논문 "Deep Researcher with Test-Time Diffusion"에서는 Test-Time Diffusion Deep Researcher (TTD-DR)라는 새로운 프레임워크를 제안합니다. 이 프레임워크는 복잡한 연구 보고서를 생성하는 과정을 확산 과정으로 개념화합니다. TTD-DR은 초기 초안을 생성하고, 이 초안을 기반으로 외부 정보를 검색하여 점진적으로 개선하는 "디노이징" 과정을 통해 최종 보고서를 완성합니다. 이 과정은 각 구성 요소에 대한 자기 진화 알고리즘을 통해 강화되어, 고품질의 맥락을 생성합니다.

예시 데이터는 다음과 같은 방식으로 구성됩니다:

1. **트레이닝 데이터**: 
   - **인풋**: 사용자가 제시한 연구 질문이나 주제.
   - **아웃풋**: 초기 초안, 검색 질문, 검색 결과, 최종 보고서.

2. **테스트 데이터**:
   - **인풋**: 새로운 연구 질문이나 주제.
   - **아웃풋**: 최종 보고서의 품질 평가 (도움이 되는지, 포괄적인지 등).

3. **구체적인 테스크**:
   - **초안 생성**: LLM을 사용하여 초기 초안을 생성합니다.
   - **검색 및 디노이징**: 검색 도구를 사용하여 외부 정보를 수집하고, 이를 통해 초안을 개선합니다.
   - **최종 보고서 작성**: 모든 정보를 종합하여 최종 보고서를 작성합니다.



The paper "Deep Researcher with Test-Time Diffusion" proposes a novel framework called Test-Time Diffusion Deep Researcher (TTD-DR). This framework conceptualizes the generation of complex research reports as a diffusion process. TTD-DR starts with an initial draft and iteratively refines it through a "denoising" process informed by external information retrieval. This process is further enhanced by a self-evolutionary algorithm applied to each component, ensuring the generation of high-quality context.

Example data is structured as follows:

1. **Training Data**:
   - **Input**: User-provided research questions or topics.
   - **Output**: Initial draft, search questions, search results, final report.

2. **Test Data**:
   - **Input**: New research questions or topics.
   - **Output**: Evaluation of the final report's quality (e.g., helpfulness, comprehensiveness).

3. **Specific Tasks**:
   - **Draft Generation**: Use LLM to generate an initial draft.
   - **Search and Denoising**: Use search tools to gather external information and refine the draft.
   - **Final Report Writing**: Synthesize all information to produce the final report.

<br/>  
# 요약   

이 논문에서는 인간의 연구 과정을 모방한 Test-Time Diffusion Deep Researcher (TTD-DR)라는 프레임워크를 제안하며, 초기 초안을 생성하고 이를 외부 정보 검색을 통해 반복적으로 개선하는 방법을 사용합니다. TTD-DR은 다양한 벤치마크에서 기존의 연구 에이전트를 능가하는 성능을 보이며, 특히 복잡한 연구 질문에 대한 포괄적인 보고서 생성에서 뛰어난 결과를 보여줍니다. 예를 들어, TTD-DR은 OpenAI Deep Research와 비교하여 LongForm Research와 DeepConsult에서 각각 69.1%와 74.5%의 승률을 기록했습니다.


This paper proposes a framework called Test-Time Diffusion Deep Researcher (TTD-DR), which mimics the human research process by generating an initial draft and iteratively refining it through external information retrieval. TTD-DR outperforms existing research agents across various benchmarks, particularly excelling in generating comprehensive reports for complex research questions. For instance, TTD-DR achieved win rates of 69.1% and 74.5% in LongForm Research and DeepConsult, respectively, compared to OpenAI Deep Research.

<br/>  
# 기타  


1. **Figure 1**: 인간의 글쓰기 과정을 계획, 초안 작성, 수정의 반복으로 설명하며, 이는 TTD-DR의 설계에 영감을 주었습니다. 인간의 글쓰기 패턴과 유사하게, TTD-DR은 초안을 작성하고 이를 반복적으로 수정하여 최종 보고서를 생성합니다.

2. **Figure 2**: TTD-DR 프레임워크의 개요를 보여줍니다. 사용자 쿼리가 초기 초안과 연구 계획을 시작하며, 이 초안은 검색 질문 생성과 정보 검색을 동적으로 안내합니다. 검색된 정보는 초안을 정제하고 개선하는 데 사용됩니다.

3. **Figure 3**: TTD-DR과 다른 오픈 소스 연구 에이전트의 비교를 보여줍니다. TTD-DR은 초안 정제 메커니즘을 도입하여, 각 섹션별로 분리된 검색을 피하고 전반적인 문맥을 유지합니다.

4. **Table 1**: TTD-DR이 다양한 데이터셋에서 다른 시스템과 비교하여 우수한 성능을 보임을 보여줍니다. 특히, OpenAI Deep Research와 비교하여 높은 승률을 기록했습니다.

5. **Table 2**: TTD-DR의 구성 요소별 성능을 보여주는 소거 연구 결과입니다. 기본 DR 에이전트에 비해 Self-evolution과 Diffusion with Retrieval을 추가했을 때 성능이 크게 향상되었습니다.

6. **Figure 7**: DR 에이전트의 성능과 지연 시간 간의 트레이드오프를 보여줍니다. TTD-DR의 두 가지 제안된 알고리즘이 테스트 시간 확장에 효율적임을 나타냅니다.

7. **Figure 8**: LongForm Research와 DeepConsult 벤치마크에서 TTD-DR이 OpenAI Deep Research보다 더 도움이 되고 포괄적인 보고서를 생성함을 보여줍니다.

8. **Figure 9**: Self-evolution이 검색 질문과 답변의 복잡성을 증가시켜 최종 보고서의 품질을 향상시킴을 보여줍니다.

9. **Figure 10**: Denoising with Retrieval이 Self-evolution에 비해 검색 쿼리의 참신성을 증가시키고, 초기 단계에서 정보를 효과적으로 활용하여 성능을 향상시킴을 보여줍니다.

10. **Appendix A.11**: HLE-Search 데이터셋에 대한 추가 분석 결과를 제공합니다. TTD-DR의 성능이 다른 DR 에이전트와 비교하여 유사하거나 더 나은 결과를 보임을 보여줍니다.

---



1. **Figure 1**: Describes the human writing process as a cycle of planning, drafting, and revising, which inspired the design of TTD-DR. Similar to human writing patterns, TTD-DR generates an initial draft and iteratively refines it to produce the final report.

2. **Figure 2**: Provides an overview of the TTD-DR framework. A user query initiates a preliminary draft and research plan, which dynamically guides the generation of search questions and information retrieval. The retrieved information is used to refine and improve the draft.

3. **Figure 3**: Compares TTD-DR with other open-source research agents. TTD-DR introduces a draft denoising mechanism, avoiding separate searches for each section and maintaining global context.

4. **Table 1**: Shows that TTD-DR outperforms other systems across various datasets, achieving high win rates compared to OpenAI Deep Research.

5. **Table 2**: Presents an ablation study of TTD-DR's components. Adding Self-evolution and Diffusion with Retrieval significantly improves performance over the basic DR agent.

6. **Figure 7**: Illustrates the trade-off between DR agent performance and latency. The proposed algorithms in TTD-DR are efficient for test-time scaling.

7. **Figure 8**: Demonstrates that TTD-DR produces more helpful and comprehensive reports than OpenAI Deep Research in LongForm Research and DeepConsult benchmarks.

8. **Figure 9**: Shows that Self-evolution increases the complexity of search queries and answers, enhancing the quality of the final report.

9. **Figure 10**: Indicates that Denoising with Retrieval increases the novelty of search queries and effectively leverages information in early stages, improving performance over Self-evolution.

10. **Appendix A.11**: Provides additional analysis results for the HLE-Search dataset, showing that TTD-DR achieves on-par or better results compared to competing DR agents.

<br/>
# refer format:     



```bibtex
@article{Han2025,
  title={Deep Researcher with Test-Time Diffusion},
  author={Rujun Han and Yanfei Chen and Zoey CuiZhu and Lesly Miculicich and Guan Sun and Yuanjun Bi and Weiming Wen and Hui Wan and Chunfeng Wen and Solène Maître and George Lee and Vishy Tirumalashetty and Emily Xue and Zizhao Zhang and Salem Haykal and Burak Gokturk and Tomas Pfister and Chen-Yu Lee},
  journal={arXiv preprint arXiv:2507.16075},
  year={2025},
  note={*Equal contribution; This work has no implications of any Google products.},
  url={https://arxiv.org/abs/2507.16075}
}
```


Han, Rujun, Yanfei Chen, Zoey CuiZhu, Lesly Miculicich, Guan Sun, Yuanjun Bi, Weiming Wen, Hui Wan, Chunfeng Wen, Solène Maître, George Lee, Vishy Tirumalashetty, Emily Xue, Zizhao Zhang, Salem Haykal, Burak Gokturk, Tomas Pfister, and Chen-Yu Lee. 2025. "Deep Researcher with Test-Time Diffusion." arXiv preprint arXiv:2507.16075. https://arxiv.org/abs/2507.16075.
