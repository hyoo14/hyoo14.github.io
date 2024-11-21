---
layout: post
title:  "[2024]Words Matter: Reducing Stigma in Online Conversations about Substance Use with Large Language Models"  
date:   2024-11-20 19:33:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 

낙인 표현이라는 새로운 토픽을 잡아서 이걸 LLM으로 완화해주는 연구..  



짧은 요약(Abstract) :    




이 연구는 물질 사용 장애(SUD)와 관련된 낙인(stigma)이 치료 참여를 방해하고 회복을 어렵게 한다는 점에 주목하여, Reddit의 소셜 미디어 데이터를 분석하였습니다. 총 120만 개 이상의 게시물을 분석하여 물질 사용자(PWUS)에 대한 낙인 표현이 포함된 3,207개의 게시물을 식별하였습니다. 이를 기반으로 GPT-4와 같은 대규모 언어 모델(LLM)을 활용하여 낙인 표현을 공감적 언어로 변환하는 모델을 개발하였고, 총 1,649개의 수정된 문장 쌍을 생성했습니다. 연구는 낙인의 언어적 특징을 탐구하고, 온라인 환경에서의 낙인 감소를 위한 실질적인 도구를 제안하며, SUD를 겪는 이들에게 보다 지지적인 디지털 환경을 조성하기 위한 방법론을 제시합니다.

---


This study addresses how stigma related to substance use disorders (SUD) acts as a barrier to treatment and recovery. Analyzing over 1.2 million Reddit posts, it identified 3,207 posts containing stigmatizing language towards people who use substances (PWUS). Leveraging large language models (LLMs) such as GPT-4, a model was developed to transform these expressions into empathetic language, resulting in 1,649 rephrased pairs. The study explores linguistic features of stigma, proposes tools for reducing stigma in online contexts, and aims to foster a more supportive digital environment for individuals affected by SUD.


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




이 연구의 독창적인 방법론은 다음과 같습니다:

1. **대규모 언어 모델(LLM)을 활용한 낙인 표현 변환**  
   기존 연구와 달리, 이 연구는 GPT-4와 같은 대규모 언어 모델을 사용하여 낙인 표현을 공감적이고 지원적인 언어로 변환하는 데 중점을 두었습니다. 특히, Link와 Phelan(2001)의 낙인 개념화 모델(라벨링, 고정관념, 분리, 차별)을 활용하여 낙인을 세부적으로 분석하고 이를 기반으로 수정된 문장을 생성했습니다.

2. **스타일과 감정 보존을 결합한 텍스트 변환**  
   단순히 낙인 표현을 제거하는 데 그치지 않고, 원래 게시물의 스타일과 감정을 유지하면서 수정된 텍스트를 생성하도록 설계되었습니다. 이는 기존의 단순한 언어 감정 분석 모델이 아닌, 더 자연스럽고 인간 중심적인 텍스트를 생성하게 합니다.

3. **데이터셋의 특이성**  
   연구에서는 1.2백만 개 이상의 Reddit 게시물을 분석하여 낙인 표현이 포함된 3,207개의 문장을 식별하고, 이를 수동으로 분류한 뒤 대규모 언어 모델을 활용해 자동화했습니다. 특히, 기존에 비해 더 큰 규모와 비약적으로 향상된 모델의 세밀함을 제공합니다.

**기존 접근 방식과의 차이점**
- 기존 연구는 주로 낙인의 표현을 식별하는 데 초점을 맞추고 있으며, 이를 줄이는 구체적인 전략을 제공하지 않는 경우가 많습니다.
- 이 연구는 단순히 낙인을 제거하는 것을 넘어, 스타일 보존 및 인간 친화적인 표현을 강조하여 수정된 문장의 품질을 높이고 사용자 경험을 개선하려 했습니다.
- 기존 방법론이 명시적으로 언급하지 않았던 감정적 및 심리적 요소를 텍스트 변환 과정에 통합함으로써, 생성된 텍스트의 자연스러움을 크게 향상시켰습니다.

---



This study's unique methodologies are as follows:

1. **Transforming Stigmatizing Language with Large Language Models (LLMs)**  
   Unlike existing research, this study leverages large language models like GPT-4 to transform stigmatizing language into empathetic and supportive expressions. It employs the stigma conceptualization framework of Link and Phelan (2001) (labeling, stereotyping, separation, discrimination) to systematically analyze stigma and generate revised sentences.

2. **Text Transformation with Style and Emotion Preservation**  
   The approach goes beyond simply removing stigmatizing expressions. It ensures that the revised text retains the original post's style and emotional tone, making the output more natural and human-centered compared to simpler sentiment analysis models.

3. **Dataset Specificity**  
   The study analyzed over 1.2 million Reddit posts, identifying 3,207 sentences with stigmatizing language. These sentences were manually annotated and then processed using LLMs for automation, offering unprecedented scale and precision compared to previous approaches.

**Differences from Existing Approaches**
- Previous studies primarily focus on identifying stigmatizing expressions without providing concrete strategies for reducing stigma.
- This study emphasizes not only removing stigma but also preserving style and human-like expressions to improve the quality of revised sentences and user experience.
- By integrating emotional and psychological elements into the text transformation process, this approach significantly enhances the naturalness of generated text, which is often overlooked in earlier methods.



   
 
<br/>
# Results  





**결과**  
이 연구에서 제안된 방법은 다른 방법들에 비해 낙인 표현 제거와 원문 스타일 보존 측면에서 더 우수한 결과를 보여주었습니다. 특히, Informed + Stylized GPT-4 모델은 가장 높은 품질의 수정된 텍스트를 생성했으며, 공감적 언어로의 변환에서 높은 충실성과 자연스러움을 유지했습니다. 인간 평가 결과, 이 모델이 "가장 높은 품질"과 "가장 충실한 변환"에서 최고 점수를 받았습니다. 반면, Baseline GPT-4 모델은 단순히 낙인 표현을 제거하는 데는 효과적이었지만, 텍스트의 맥락과 감정적인 자연스러움이 떨어졌습니다.

**평가 지표(Metrics)**  
1. **인간 평가**  
   - **전체 품질(Overall Quality):** 자연스러움, 응집성, 인간 친화도, 전체 논리성을 평가.
   - **낙인 제거 효과(Effectively De-Stigmatized):** 라벨링, 고정관념, 분리, 차별을 줄이는 정도.
   - **충실성(Faithfulness):** 원문 메시지의 감정과 스타일을 유지한 정도.

2. **자동 평가**  
   - **LIWC 분석:** 원문과 수정된 텍스트 간의 심리언어적 유사성을 비교.
   - BLEU나 ROUGE와 같은 기존 텍스트 생성 평가 지표는 이 작업의 특성상 적합하지 않음.

**데이터셋**  
- 총 1.2백만 개의 Reddit 게시물을 분석하여, 물질 사용자(PWUS)에 대한 낙인 표현이 포함된 3,207개의 문장을 식별.
- 이 중 1,649개의 수정된 문장 쌍이 최종적으로 생성되어 평가에 사용됨.

---



**Results**  
The proposed method outperformed other methods in both removing stigmatizing expressions and preserving the original style of the text. Notably, the Informed + Stylized GPT-4 model produced the highest quality rewritten texts, maintaining high faithfulness and naturalness in transforming stigmatizing language into empathetic expressions. Human evaluation showed that this model scored highest for "Overall Quality" and "Most Faithful Transformations." In contrast, the Baseline GPT-4 model was effective at removing stigmatizing language but failed to retain the contextual and emotional naturalness of the original posts.

**Evaluation Metrics**  
1. **Human Evaluation**  
   - **Overall Quality:** Assessed based on naturalness, cohesion, human-likeness, and overall logical coherence.
   - **Effectively De-Stigmatized:** Measured the reduction in labeling, stereotyping, separation, and discrimination.
   - **Faithfulness:** Evaluated the extent to which the revised text preserved the emotional tone and style of the original.

2. **Automatic Evaluation**  
   - **LIWC Analysis:** Compared psycholinguistic features between the original and revised texts.
   - Metrics like BLEU and ROUGE were deemed unsuitable due to the substantial differences in meaning between the original and revised texts.

**Dataset**  
- Analyzed 1.2 million Reddit posts, identifying 3,207 sentences with stigmatizing language.
- Of these, 1,649 revised sentence pairs were used for evaluation.



<br/>
# 예제  





**테스트 예시**  
다음은 연구에서 사용된 테스트 데이터와 경쟁 방법과의 비교를 보여주는 예시입니다:

1. **원문 (낙인 표현 포함)**  
   "마약 중독자들은 절대 변하지 않는다. 그들은 항상 중독 상태일 것이다."

2. **Baseline GPT-4 (기초 모델) 출력**  
   "마약 중독자들이 변화하는 것은 어려운 주제이다."  
   → **문제점:** 낙인 표현("중독자")은 제거되었지만, 원문의 맥락과 감정이 크게 손상되었습니다. 결과적으로 지나치게 일반적이고 비인간적인 텍스트가 생성되었습니다.

3. **Informed + Stylized GPT-4 (제안된 방법) 출력**  
   "물질 사용 장애를 겪는 사람들도 회복할 가능성이 있습니다."  
   → **우수점:** 라벨링("중독자")이 제거되었고, 고정관념("절대 변하지 않는다")이 공감적 언어로 대체되었습니다. 동시에 원문이 전달하려는 메시지의 맥락과 감정을 유지하며 더욱 인간적인 표현을 사용했습니다.

**비교 요약**  
제안된 방법(Informed + Stylized GPT-4)은 단순히 낙인 표현을 제거하는 것을 넘어, 원문의 스타일과 감정을 보존하며 공감적인 언어로 변환했습니다. 반면, Baseline GPT-4는 단순한 표현 제거에 초점이 맞춰져 원문과의 충실도가 낮았습니다.

---



**Test Example**  
Here is a test example comparing the proposed method to competitors:

1. **Original (with stigmatizing language)**  
   "Addicts will never change. They will always be addicts."

2. **Baseline GPT-4 Output**  
   "The topic of addicts changing is a difficult one."  
   → **Issue:** While the stigmatizing term "addicts" was removed, the context and emotional tone of the original were lost. The result was overly generic and lacked a human touch.

3. **Informed + Stylized GPT-4 (Proposed Method) Output**  
   "Individuals with substance use disorders also have the potential for recovery."  
   → **Strength:** The labeling term "addicts" was removed, and the stereotype ("never change") was replaced with empathetic language. At the same time, the context and emotional tone of the original message were preserved, making the output more human-centered.

**Comparison Summary**  
The proposed method (Informed + Stylized GPT-4) goes beyond simple removal of stigmatizing expressions by preserving the style and emotional tone of the original text while transforming it into empathetic language. In contrast, Baseline GPT-4 focused on expression removal, resulting in low faithfulness to the original.


<br/>  
# 요약   




이 연구는 물질 사용 장애(SUD)에 대한 낙인 표현을 공감적 언어로 변환하기 위해 대규모 언어 모델(GPT-4)을 활용하는 방법을 제안했습니다. 연구는 Link와 Phelan의 낙인 개념화를 기반으로 라벨링, 고정관념, 분리, 차별의 요소를 분석하고 이를 수정하는 프레임워크를 개발했습니다. 특히, Informed + Stylized GPT-4 모델은 낙인 제거뿐만 아니라 원문 스타일과 감정을 유지하며 가장 높은 품질의 결과를 보여주었습니다. 예를 들어, "중독자는 절대 변하지 않는다"는 표현을 "물질 사용 장애를 겪는 사람도 회복 가능성이 있다"로 공감적으로 변환하여 맥락과 메시지를 유지했습니다. 기존 접근 방식과 달리, 제안된 방법은 스타일과 감정을 보존하며 보다 인간 중심적인 텍스트를 생성하는 데 성공했습니다.  

---


This study proposed a method using large language models (GPT-4) to transform stigmatizing expressions about substance use disorders (SUD) into empathetic language. Based on Link and Phelan's stigma conceptualization, the study developed a framework to analyze and revise elements such as labeling, stereotyping, separation, and discrimination. The Informed + Stylized GPT-4 model demonstrated superior performance, excelling in stigma removal while preserving the original style and emotional tone. For instance, the phrase "Addicts will never change" was rephrased empathetically as "Individuals with substance use disorders also have the potential for recovery," maintaining the context and message. Unlike previous approaches, the proposed method successfully generates more human-centered texts by preserving style and emotion.  

<br/>  
# 기타  


<br/>
# refer format:     



@inproceedings{bouzoubaa2024words,
  author = {Layla Bouzoubaa and Elham Aghakhani and Rezvaneh Rezapour},
  title = {Words Matter: Reducing Stigma in Online Conversations about Substance Use with Large Language Models},
  booktitle = {Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year = {2024},
  month = {November},
  address = {Miami, Florida, USA},
  editor = {Yaser Al-Onaizan and Mohit Bansal and Yun-Nung Chen},
  publisher = {Association for Computational Linguistics},
  pages = {9139--9156},
  url = {https://aclanthology.org/2024.emnlp-main.516},
  language = {English},
  note = {}
}




Bouzoubaa, Layla, Elham Aghakhani, and Rezvaneh Rezapour. “Words Matter: Reducing Stigma in Online Conversations about Substance Use with Large Language Models.” In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, edited by Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen, 9139–9156. Miami, Florida, USA: Association for Computational Linguistics, 2024. https://aclanthology.org/2024.emnlp-main.516.




