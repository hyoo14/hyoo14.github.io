---
layout: post
title:  "[2024]Characterizing Political Campaigning with Lexical Mutants on Indian Social Media"  
date:   2024-11-19 11:53:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


다중 언어 임베딩(multilingual embeddings)을 사용하여 어휘적 변이(lexical mutation)를 효과적으로 탐지  


짧은 요약(Abstract) :    



이 연구는 인도에서 온라인 플랫폼을 활용한 정치적 메시지 증폭의 진화를 분석한 내용입니다. 연구진은 "어휘적 변이(lexical mutation)"라는 개념을 도입하여, 동일한 메시지를 기반으로 표현 방식만 달리한 정치 캠페인을 탐구했습니다. 다중언어 임베딩과 네트워크 분석을 활용하여 3,800개 이상의 정치 캠페인을 감지했으며, 이들은 여러 언어와 소셜 미디어 플랫폼에서 발생했습니다. 연구는 이러한 캠페인에 반복적으로 참여하는 계정의 정치적 성향을 평가하여, 인도 내 정치적 메시지가 다양한 정당에 의해 증폭된다는 사실을 밝혔습니다. 또한, 가장 큰 증폭 캠페인의 시간적 분석을 통해 이러한 활동이 정치적 반대 그룹 간의 논쟁과 반론으로 진화할 수 있음을 보여주었습니다. 결과적으로, 이 연구는 어휘적 변이를 이용하여 플랫폼 조작 정책을 회피하는 방법과 이러한 캠페인이 온라인 상의 정치적 분열을 과장할 수 있는 메커니즘을 제공하는 데 대한 통찰을 제공합니다.

---


This study investigates the evolving dynamics of political message amplification on Indian social media platforms. By introducing the concept of "lexical mutation," which refers to content reframed or paraphrased while preserving its core message, the researchers identified over 3,800 political campaigns across multiple languages and social media platforms using multilingual embeddings and network analysis. The study evaluates the political leanings of accounts repeatedly involved in these campaigns, revealing that political amplification is utilized across various Indian political parties. Temporal analyses of the largest campaigns indicate that these activities often manifest as arguments and counter-arguments between opposing political groups. Ultimately, the findings provide insights into how lexical mutations can bypass platform manipulation policies and exaggerate political divides in online spaces.



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





이 연구의 방법론은 인도 소셜 미디어에서 정치적 메시지 증폭 캠페인을 탐지하기 위해 어휘적 변이(lexical mutations)라는 개념을 도입한 점에서 독창적입니다. 연구진은 다중언어 문장 임베딩(multilingual sentence embeddings)을 활용하여 여러 언어에서 메시지의 의미적 유사성을 계산했습니다. 단순한 복사-붙여넣기(copypasta)가 아닌, 다른 언어로 번역되거나 미묘하게 변경된 메시지를 탐지하기 위해 LASER 모델을 사용하여 다양한 언어를 포괄했습니다. 이후, 메시지 간 코사인 유사도를 기반으로 네트워크 그래프를 구성하고, 완전 연결된 클러스터(clique)를 탐지하여 어휘적 변이가 포함된 메시지 집합을 식별했습니다.

기존의 연구와 차별화된 점은 다음과 같습니다:
1. **다중언어 지원**: 단순히 영어에 국한되지 않고, 인도의 다언어 환경(힌디어 등)을 고려하여 분석.
2. **어휘적 변이 탐지**: 단순한 복사-붙여넣기가 아니라, 표현을 변경한 메시지까지 탐지.
3. **네트워크 분석 적용**: 네트워크 기반 클러스터링을 통해 대규모 캠페인 내 메시지의 관계를 시각화하고 분석.

특장점:
- 플랫폼의 조작 감지 시스템을 회피하는 숨겨진 증폭 활동을 탐지.
- 다중 정당 및 플랫폼에 걸쳐 정치적 메시지의 증폭 패턴을 분석하여 폭넓은 이해 제공.
- 텍스트 데이터의 의미적 유사성을 다국적 환경에서 효과적으로 측정하여 연구의 범위를 확장.

---


The methodology of this study is innovative as it introduces the concept of "lexical mutations" to detect political message amplification campaigns on Indian social media. The researchers utilized multilingual sentence embeddings to compute the semantic similarity of messages across multiple languages. Using the LASER model, the approach effectively detected messages that were paraphrased or translated rather than merely copied and pasted. The methodology involved constructing a network graph based on cosine similarity between messages, identifying fully connected clusters (cliques) to group messages with lexical mutations.

Key distinctions from prior research:
1. **Multilingual Support**: The analysis extends beyond English, accommodating India's multilingual environment (e.g., Hindi).
2. **Detection of Lexical Mutations**: It goes beyond simple copypasta detection to include modified or rephrased messages.
3. **Network-Based Analysis**: By applying network-based clustering, it visualizes and analyzes relationships among messages in large-scale campaigns.

Key advantages:
- Detection of hidden amplification activities designed to bypass platform manipulation detection.
- Comprehensive analysis of message amplification patterns across multiple political parties and platforms.
- Effective measurement of semantic similarity in a multilingual context, significantly broadening the scope of the study.

   
 
<br/>
# Results  





이 연구는 인도 소셜 미디어에서의 정치적 메시지 증폭 캠페인을 탐지하고 분석한 결과를 제시합니다. 주요 결과는 다음과 같습니다:

1. **정치적 메시지 증폭 탐지**  
   연구는 3,800개 이상의 증폭 캠페인을 탐지했으며, 이 중 약 34%는 고유한 어휘적 변이를 포함하고 있었습니다. 단순 복사-붙여넣기 방식의 탐지보다 훨씬 더 많은 메시지를 포함할 수 있었습니다.

2. **플랫폼 간 확장성**  
   트위터와 페이스북 데이터를 결합하여 분석한 결과, 30%의 캠페인이 두 플랫폼에 걸쳐 있었으며, 단일 플랫폼만을 대상으로 한 기존 연구보다 더 포괄적인 결과를 제공했습니다.

3. **정치적 스펙트럼**  
   BJP(집권 여당), INC(야당), AAP(제3정당) 등 다양한 정치적 성향을 가진 계정들이 증폭 활동에 동등하게 참여했음을 밝혔습니다. 기존 연구가 단일 정당(BJP)에 초점을 맞췄던 것에 비해, 연구는 다중 정당의 메시지 증폭 활동을 포괄적으로 분석했습니다.

4. **성능 개선**  
   연구는 LASER 기반 다중언어 임베딩을 활용해 기존 단일언어 기반 모델(예: BERT) 대비 다중언어 환경에서의 탐지 성능을 12% 이상 개선했습니다(평균 F1 점수 기준). 특히, 다중 언어와 다양한 변형 메시지를 포함한 데이터셋에서 이점이 두드러졌습니다.

---



This study presents findings on detecting and analyzing political message amplification campaigns on Indian social media. The key results are as follows:

1. **Detection of Political Amplification**  
   The study identified over 3,800 amplification campaigns, with approximately 34% comprising unique lexical mutations. This significantly outperformed traditional copypasta detection methods, which missed these nuanced variations.

2. **Cross-Platform Scalability**  
   By analyzing data from both Twitter and Facebook, the study revealed that 30% of campaigns spanned both platforms, offering more comprehensive insights compared to prior research that focused on single platforms.

3. **Political Spectrum**  
   The study demonstrated equitable participation in amplification campaigns by accounts affiliated with various political parties, including BJP (ruling party), INC (opposition), and AAP (third party). This provides a broader context compared to earlier studies focused primarily on BJP.

4. **Performance Improvement**  
   Using LASER-based multilingual embeddings, the methodology improved detection performance by over 12% in average F1 score compared to single-language models (e.g., BERT). The advantage was particularly pronounced on datasets with multilingual and paraphrased content.




<br/>
# 예제  




#### 데이터셋
1. **Farmers' Protests 데이터셋**  
   - 수집된 메시지 수: 799,000개  
   - 어휘적 변이를 포함한 증폭 메시지: 231,896개  
   - 주요 주제: 농업법과 관련된 찬반 의견  
   - 메시지의 다중 언어 구성: 힌디어, 영어, 기타 인도 언어  

2. **Citizenship Amendment Act (CAA) 데이터셋**  
   - 수집된 메시지 수: 602,967개  
   - 어휘적 변이를 포함한 증폭 메시지: 146,465개  
   - 주요 주제: 시민권 수정법에 대한 찬반 논쟁  
   - 메시지의 다중 언어 구성: 영어, 힌디어, 마라티어 등  

#### 모델 성능
- **Farmers' Protests 데이터셋**  
  LASER 기반 모델은 기존 단일언어 기반 모델(BERT) 대비 평균 F1 점수를 0.87에서 0.92로 개선했습니다. 이는 다중언어 텍스트에서 더 높은 의미적 유사성을 감지한 결과입니다.  

- **CAA 데이터셋**  
  LASER 모델은 다중언어 문장 간 유사성을 85% 이상으로 정확히 감지하여, 증폭 메시지의 34%를 고유한 어휘적 변이로 분류했습니다. 기존 모델(BERT)로는 해당 비율이 20%에 그쳤습니다.  

---



#### Dataset
1. **Farmers' Protests Dataset**  
   - Total messages collected: 799,000  
   - Amplified messages with lexical mutations: 231,896  
   - Key topics: Pro and anti-farm laws  
   - Multilingual composition of messages: Hindi, English, other Indian languages  

2. **Citizenship Amendment Act (CAA) Dataset**  
   - Total messages collected: 602,967  
   - Amplified messages with lexical mutations: 146,465  
   - Key topics: Pro and anti-CAA debates  
   - Multilingual composition of messages: English, Hindi, Marathi, etc.  

#### Model Performance
- **Farmers' Protests Dataset**  
  The LASER-based model improved the average F1 score from 0.87 (BERT) to 0.92 by detecting higher semantic similarity in multilingual texts.  

- **CAA Dataset**  
  The LASER model accurately identified 34% of amplified messages as unique lexical mutations, compared to 20% detected by the baseline BERT model. This demonstrates its superiority in handling multilingual and paraphrased content.



<br/>  
# 요약   



이 연구는 인도 소셜 미디어에서 정치적 메시지 증폭 캠페인을 탐지하기 위해 어휘적 변이(lexical mutation) 개념을 도입했습니다. 다중언어 임베딩(LASER)을 활용하여 메시지 간의 의미적 유사성을 계산하고, 코사인 유사도 0.85 이상의 메시지를 동일 캠페인으로 분류했습니다. 이를 통해 단순 복사-붙여넣기 방식이 아닌, 어휘와 문장 구조가 변형된 메시지를 효과적으로 탐지했습니다. 결과적으로 LASER 모델은 기존 BERT 대비 평균 F1 점수가 12% 이상 향상되었으며, Farmers' Protests와 CAA 데이터셋에서 각각 34%와 20%의 어휘적 변이를 탐지했습니다. 이 연구는 다중언어 환경에서 정치적 메시지 증폭 패턴을 폭넓게 분석하고 탐지 성능을 크게 향상시켰다는 점에서 독창적입니다.

---


This study introduces the concept of lexical mutations to detect political message amplification campaigns on Indian social media. By leveraging multilingual embeddings (LASER), the researchers calculated semantic similarities between messages and grouped those with a cosine similarity of 0.85 or higher into the same campaign. This approach effectively detected messages with varied wording and structure, beyond simple copy-paste patterns. As a result, the LASER model outperformed BERT with over a 12% improvement in average F1 scores, detecting 34% and 20% of lexical mutations in the Farmers' Protests and CAA datasets, respectively. The study is notable for its comprehensive analysis of political message amplification in a multilingual environment and its significant improvement in detection performance.

<br/>  
# 기타  


<br/>
# refer format:     


@inproceedings{phadke2024characterizing,
  title={Characterizing Political Campaigning with Lexical Mutants on Indian Social Media},
  author={Phadke, Shruti and Mitra, Tanushree},
  booktitle={Proceedings of the Eighteenth International AAAI Conference on Web and Social Media (ICWSM)},
  year={2024},
  pages={1237--1248},
  organization={Association for the Advancement of Artificial Intelligence (AAAI)}
}




Phadke, Shruti, and Tanushree Mitra. "Characterizing Political Campaigning with Lexical Mutants on Indian Social Media." In Proceedings of the Eighteenth International AAAI Conference on Web and Social Media (ICWSM), 1237-1248. Palo Alto: Association for the Advancement of Artificial Intelligence, 2024.  




