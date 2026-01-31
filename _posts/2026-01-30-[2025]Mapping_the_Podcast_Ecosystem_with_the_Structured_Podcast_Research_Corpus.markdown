---
layout: post
title:  "[2025]Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus"
date:   2026-01-30 16:10:42 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: 이 논문에서는 2020년 5월과 6월에 수집된 110만 개의 팟캐스트 트랜스크립트를 기반으로 팟캐스트 생태계를 분석하였다.


짧은 요약(Abstract) :

이 논문에서는 팟캐스트 생태계를 대규모로 분석하기 위한 데이터 부족 문제를 해결하기 위해, 2020년 5월과 6월에 공개 RSS 피드를 통해 수집한 110만 개 이상의 팟캐스트 대본으로 구성된 방대한 데이터셋인 구조화된 팟캐스트 연구 코퍼스(SPORC)를 소개합니다. 이 데이터는 텍스트뿐만 아니라 메타데이터, 추론된 화자 역할, 오디오 특성 및 37만 개 에피소드의 화자 전환 정보를 포함하고 있습니다. 이 데이터를 활용하여 팟캐스트 생태계의 내용, 구조 및 반응성을 조사하였으며, 이를 통해 이 인기 있는 매체에 대한 지속적인 컴퓨터 연구의 기회를 열었습니다.



This paper addresses the issue of limited data for large-scale analysis of the podcast ecosystem by introducing the Structured Podcast Research Corpus (SPORC), a massive dataset of over 1.1 million podcast transcripts collected from public RSS feeds during May and June 2020. This data is not only textual but also includes metadata, inferred speaker roles, audio features, and speaker turns for a subset of 370,000 episodes. Using this data, we conduct a foundational investigation into the content, structure, and responsiveness of the podcast ecosystem, thereby opening the door for continued computational research into this popular medium.


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



이 논문에서는 팟캐스트 생태계를 연구하기 위해 **Structured Podcast Research Corpus (SPORC)**라는 대규모 데이터셋을 구축하고, 이를 통해 팟캐스트의 내용, 구조, 그리고 사회적 반응성을 분석하는 방법론을 제시합니다. 이 데이터셋은 2020년 5월과 6월에 공개 RSS 피드를 통해 수집된 110만 개 이상의 팟캐스트 에피소드의 전사본을 포함하고 있습니다. 이 데이터셋은 텍스트뿐만 아니라 메타데이터, 추론된 화자 역할, 오디오 특징 및 화자 전환 정보도 포함되어 있습니다.

#### 1. 데이터 수집
- **RSS 피드 활용**: 팟캐스트의 메타데이터를 수집하기 위해 Podcast Index라는 공개 데이터베이스를 사용하여 273,000개의 영어 팟캐스트 쇼를 식별했습니다. 이로부터 1.3백만 개의 에피소드와 관련된 오디오 파일 및 메타데이터를 다운로드했습니다.

#### 2. 전사
- **Whisper 모델 사용**: 오디오 파일의 내용을 텍스트로 변환하기 위해 Whisper라는 자동 음성 인식(ASR) 시스템을 사용했습니다. Whisper의 base.en 모델을 선택하여 품질과 속도의 균형을 맞췄습니다. 전사 과정에서 발생할 수 있는 오류를 줄이기 위해 n-그램 필터를 사용하여 반복적인 구문을 제거했습니다. 최종적으로 1.1백만 개의 에피소드 전사본을 확보했습니다.

#### 3. 오디오 특징 추출
- **openSMILE 툴킷 사용**: 오디오에서 화자의 발화 특성을 추출하기 위해 openSMILE 툴킷을 사용했습니다. 여기서는 기본 주파수(F0), 첫 번째 포먼트(F1), 그리고 첫 네 개의 멜 주파수 켑스트럼 계수(MFCCs 1-4)를 추출하여 화자의 음성 특성을 분석했습니다.

#### 4. 화자 식별
- **화자 다이어리제이션**: Whisper는 화자를 구분하지 않기 때문에, pyannote라는 도구를 사용하여 오디오 파일을 개별 화자 발화로 분리했습니다. 이 과정에서 각 발화 세그먼트를 특정 화자에 매핑하여 전사본과 연결했습니다.

#### 5. 역할 주석
- **호스트 및 게스트 식별**: 호스트와 게스트의 이름을 식별하기 위해, 팟캐스트의 설명 및 전사본에서 이름을 추출하고, 이를 HOST, GUEST, NEITHER로 분류하는 모델을 개발했습니다. 이 모델은 RoBERTa 아키텍처를 기반으로 하여 훈련되었습니다.

#### 6. 데이터 분석
- **주제 모델링**: LDA(Latent Dirichlet Allocation) 주제 모델을 사용하여 각 에피소드의 주제를 분석하고, 팟캐스트의 내용과 카테고리 간의 관계를 시각화했습니다. 이를 통해 팟캐스트의 주제적 커뮤니티와 네트워크 구조를 파악했습니다.

이러한 방법론을 통해 연구자들은 팟캐스트 생태계의 구조와 반응성을 분석하고, 향후 연구를 위한 기초 자료를 제공할 수 있게 되었습니다.

---




In this paper, the authors present a methodology for studying the podcast ecosystem by constructing a large-scale dataset called the **Structured Podcast Research Corpus (SPORC)**. This dataset includes over 1.1 million podcast episode transcripts collected from public RSS feeds during May and June 2020. The dataset encompasses not only text but also metadata, inferred speaker roles, audio features, and speaker turn information.

#### 1. Data Collection
- **Utilization of RSS Feeds**: To gather podcast metadata, the authors used Podcast Index, a public database, to identify 273,000 English-language podcast shows. From these feeds, they downloaded a total of 1.3 million episodes along with their associated audio files and metadata.

#### 2. Transcription
- **Use of Whisper Model**: The authors employed an automatic speech recognition (ASR) system called Whisper to convert audio files into text. They selected the base.en model of Whisper to balance quality and speed. To minimize transcription errors, they applied an n-gram filter to remove repetitive phrases. Ultimately, they secured 1.1 million episode transcripts.

#### 3. Audio Feature Extraction
- **Using openSMILE Toolkit**: To extract speaker characteristics from audio, the authors utilized the openSMILE toolkit. They focused on extracting the fundamental frequency (F0), the first formant (F1), and the first four Mel Frequency Cepstral Coefficients (MFCCs 1-4) to analyze vocal characteristics.

#### 4. Speaker Identification
- **Speaker Diarization**: Since Whisper does not distinguish between speakers, the authors used a tool called pyannote to split audio files into individual speaker turns. This process allowed them to map each audio segment to a specific speaker, linking it back to the transcripts.

#### 5. Role Annotation
- **Identifying Hosts and Guests**: To identify the names of hosts and guests, the authors developed a model that extracts names from podcast descriptions and transcripts, classifying them as HOST, GUEST, or NEITHER. This model was based on the RoBERTa architecture and was trained on annotated data.

#### 6. Data Analysis
- **Topic Modeling**: The authors employed Latent Dirichlet Allocation (LDA) to analyze the topics of each episode, visualizing the relationship between podcast content and categories. This analysis helped identify topical communities and the network structure of podcasts.

Through this methodology, the researchers were able to analyze the structure and responsiveness of the podcast ecosystem, providing foundational insights for future research.


<br/>
# Results



이 논문에서는 Structured Podcast Research Corpus (SPORC)라는 대규모 팟캐스트 데이터셋을 구축하고, 이를 통해 팟캐스트 생태계의 내용, 구조 및 반응성을 분석하였다. 연구의 주요 결과는 다음과 같다:

1. **경쟁 모델**: SPORC 데이터셋은 기존의 팟캐스트 데이터셋과 비교하여 훨씬 더 방대한 규모를 자랑한다. 예를 들어, Spotify에서 제공한 20만 개의 에피소드 데이터셋과 비교할 때, SPORC는 110만 개의 에피소드를 포함하고 있다. 이는 팟캐스트 생태계에 대한 보다 포괄적인 분석을 가능하게 한다.

2. **테스트 데이터**: SPORC는 2020년 5월과 6월에 공개 RSS 피드를 통해 수집된 모든 영어 팟캐스트 에피소드를 포함하고 있으며, 이 데이터는 텍스트, 메타데이터, 추론된 화자 역할 및 오디오 특성을 포함한다. 특히, 37만 개의 에피소드에 대해 화자 분리(diarization)와 화자 식별이 이루어졌다.

3. **메트릭**: 연구에서는 여러 메트릭을 사용하여 팟캐스트의 주제 분포, 네트워크 연결성 및 반응성을 분석하였다. 예를 들어, LDA 주제 모델을 사용하여 각 에피소드의 주제 분포를 추출하고, 이를 통해 팟캐스트의 주제 커뮤니티를 시각화하였다. 또한, 게스트의 공동 출연을 기반으로 한 소셜 네트워크 분석을 통해 팟캐스트 간의 연결성을 평가하였다.

4. **비교**: SPORC의 결과는 팟캐스트 생태계가 주요 사건에 어떻게 반응하는지를 보여준다. 예를 들어, 조지 플로이드 사건에 대한 팟캐스트의 반응을 분석한 결과, 팟캐스트 생태계는 사건 발생 후 약 10일 동안 주제에 대한 논의가 급증했으며, 이는 뉴스 미디어의 반응과 유사한 패턴을 보였다. 그러나 팟캐스트의 반응은 뉴스 미디어보다 느린 속도로 진행되었다.

5. **결론**: SPORC 데이터셋과 분석 결과는 팟캐스트 생태계에 대한 새로운 질문을 제기하며, 향후 연구에 대한 기초 자료를 제공한다. 특히, 팟캐스트의 주제와 커뮤니티 구조, 정보 확산 및 정치적 논의의 양상에 대한 심층적인 연구가 가능해질 것이다.

---




In this paper, a large-scale podcast dataset called the Structured Podcast Research Corpus (SPORC) was constructed, and an analysis of the content, structure, and responsiveness of the podcast ecosystem was conducted. The main results of the study are as follows:

1. **Competing Models**: The SPORC dataset boasts a significantly larger scale compared to existing podcast datasets. For instance, while Spotify provided a dataset of 200,000 episodes, SPORC includes 1.1 million episodes. This allows for a more comprehensive analysis of the podcast ecosystem.

2. **Test Data**: SPORC encompasses all English-language podcast episodes collected from public RSS feeds during May and June 2020, including text, metadata, inferred speaker roles, and audio features. Notably, speaker diarization and identification were performed on a subset of 370,000 episodes.

3. **Metrics**: Various metrics were employed to analyze the topic distribution, network connectivity, and responsiveness of podcasts. For example, an LDA topic model was used to extract the topic distribution for each episode, allowing for the visualization of podcast topic communities. Additionally, social network analysis based on guest co-appearances was conducted to assess the connectivity between podcasts.

4. **Comparisons**: The results from SPORC illustrate how the podcast ecosystem responds to major events. For instance, an analysis of the response to the George Floyd incident revealed that discussions related to the topic surged for about 10 days following the event, showing a pattern similar to that of news media. However, the response in the podcast ecosystem was slower compared to that of news media.

5. **Conclusion**: The SPORC dataset and its analyses raise new questions about the podcast ecosystem and provide foundational material for future research. In particular, it opens avenues for in-depth studies on the topics and community structures of podcasts, information diffusion, and the dynamics of political discourse.


<br/>
# 예제



이 논문에서는 팟캐스트의 호스트와 게스트를 자동으로 식별하기 위한 모델을 개발하는 과정을 설명합니다. 이 모델은 주어진 팟캐스트 에피소드의 텍스트에서 호스트와 게스트의 이름을 추출하고, 이들이 어떤 역할을 하는지를 분류하는 작업을 수행합니다. 

#### 1. 데이터 수집
모델을 훈련시키기 위해, 연구자들은 2,000개의 후보 이름을 수집했습니다. 이 후보 이름은 팟캐스트 설명, 에피소드 설명, 그리고 에피소드의 첫 350단어에서 추출된 인물 이름들입니다. 이 이름들은 두 단어로 구성된 이름만 포함되며, 단일 이름은 제외됩니다.

#### 2. 주석 작업
후보 이름에 대한 주석 작업이 진행되었습니다. 주석자는 각 이름이 호스트(HOST), 게스트(GUEST), 또는 그 외(NEITHER) 중 어떤 역할인지 판단합니다. 이 작업은 Prolific 플랫폼을 통해 진행되었으며, 각 이름에 대해 세 번의 주석이 수집되었습니다. 주석자들은 주어진 이름과 함께 팟캐스트 설명, 에피소드 설명, 그리고 해당 이름이 처음 등장하는 텍스트의 150단어를 참고하여 판단합니다.

#### 3. 모델 훈련
주석 데이터를 바탕으로 RoBERTa 모델을 훈련시킵니다. 이 모델은 각 이름의 주변 문맥을 고려하여 호스트, 게스트, 또는 그 외의 역할을 분류합니다. 훈련 과정에서 모델은 50단어의 문맥을 사용하여 각 이름을 분류합니다. 최종적으로 모델은 0.87의 교차 검증 정확도와 0.88의 테스트 정확도를 달성했습니다.

#### 4. 예시
- **입력 데이터 (트레이닝 데이터)**: 
  - 후보 이름: "John Doe"
  - 문맥: "안녕하세요, 저는 여러분의 호스트 John Doe입니다. 오늘은 특별한 게스트와 함께합니다."
  
- **출력 데이터**:
  - "John Doe" → HOST

- **입력 데이터 (테스트 데이터)**:
  - 후보 이름: "Jane Smith"
  - 문맥: "오늘의 에피소드에서는 Jane Smith가 게스트로 출연합니다."
  
- **출력 데이터**:
  - "Jane Smith" → GUEST

이러한 방식으로 모델은 팟캐스트 에피소드의 텍스트에서 호스트와 게스트의 역할을 자동으로 식별할 수 있습니다.

---




This paper describes the process of developing a model to automatically identify hosts and guests in podcasts. The model extracts names from the text of a given podcast episode and classifies their roles as either host or guest.

#### 1. Data Collection
To train the model, the researchers collected 2,000 candidate names. These candidate names were extracted from podcast descriptions, episode descriptions, and the first 350 words of the episode's transcript. Only two-word names were included, while single names were excluded.

#### 2. Annotation Task
An annotation task was conducted for the candidate names. Annotators judged whether each name was a host (HOST), guest (GUEST), or neither (NEITHER). This task was carried out using the Prolific platform, and three annotations were collected for each name. Annotators were provided with the name, the podcast description, the episode description, and 150 words of context surrounding the first appearance of the name in the transcript.

#### 3. Model Training
Based on the annotated data, a RoBERTa model was trained to classify each extracted name as HOST, GUEST, or NEITHER. The model considers the context of 50 words surrounding each name during classification. Ultimately, the model achieved a cross-validation accuracy of 0.87 and a test accuracy of 0.88.

#### 4. Example
- **Input Data (Training Data)**: 
  - Candidate Name: "John Doe"
  - Context: "Hello, I am your host John Doe. Today, I have a special guest with me."
  
- **Output Data**:
  - "John Doe" → HOST

- **Input Data (Test Data)**:
  - Candidate Name: "Jane Smith"
  - Context: "In today's episode, Jane Smith is featured as a guest."
  
- **Output Data**:
  - "Jane Smith" → GUEST

In this way, the model can automatically identify the roles of hosts and guests in the text of podcast episodes.

<br/>


이 논문에서는 2020년 5월과 6월에 수집된 110만 개의 팟캐스트 트랜스크립트를 기반으로 팟캐스트 생태계를 분석하였다. 연구 결과, 팟캐스트의 주제와 사회적 네트워크 구조가 서로 연결되어 있으며, 특히 COVID-19와 인종 정의와 같은 주제는 다양한 카테고리에서 폭넓게 논의되었다. 예를 들어, 조지 플로이드 사건에 대한 논의는 모든 카테고리에서 나타났으며, 이는 팟캐스트가 사회적 이슈에 대한 집단적 주의를 형성하는 데 중요한 역할을 한다는 것을 보여준다.




In this paper, the podcast ecosystem was analyzed based on 1.1 million podcast transcripts collected in May and June 2020. The results indicate that the topics and social network structures of podcasts are interconnected, with significant discussions on themes like COVID-19 and racial justice across various categories. For instance, discussions surrounding the George Floyd incident were prevalent across all categories, highlighting the role of podcasts in shaping collective attention on social issues.

<br/>
# 기타



1. **다이어그램 및 피규어**
   - **피규어 1**: 다양한 주제의 분포를 시각화하여, 특정 카테고리(예: 스포츠, 종교) 내에서의 주제의 일관성을 보여줍니다. 반면, "Black Lives Matter"와 같은 주제는 여러 카테고리에 걸쳐 분포되어 있습니다.
   - **피규어 2**: 팟캐스트-게스트 네트워크를 시각화하여, 비즈니스, 스포츠, 뉴스 카테고리가 서로 밀접하게 연결되어 있음을 보여줍니다. 반면, 종교와 사회 카테고리는 상대적으로 느슨한 연결을 보입니다.
   - **피규어 3**: 조지 플로이드 사건에 대한 팟캐스트의 반응을 시간에 따라 분석하여, 사건 발생 후 빠르게 주목도가 증가하고, 이후 서서히 감소하는 패턴을 보여줍니다.

2. **테이블**
   - **테이블 1**: 카테고리별 모듈성(modularity)을 측정하여, 비즈니스와 스포츠 카테고리가 서로 밀접하게 연결되어 있음을 나타냅니다. 이는 이들 카테고리 내에서 게스트 공유가 더 빈번하다는 것을 의미합니다.
   - **테이블 2**: 인종 정의와 관련된 주제의 주요 단어를 나열하여, 각 주제가 어떤 카테고리에서 주로 다루어지는지를 보여줍니다.
   - **테이블 3**: 각 주제의 평균 카테고리 비율을 정리하여, 특정 주제가 어떤 카테고리에서 더 많이 다루어지는지를 나타냅니다.

3. **어펜딕스**
   - **어펜딕스 A**: 전사(transcription) 품질을 평가하기 위한 방법론을 설명하며, Whisper 모델의 성능을 검증합니다. 평균 단어 오류율(Word Error Rate, WER)이 3%로 나타나, 전반적으로 높은 품질의 전사가 이루어졌음을 보여줍니다.
   - **어펜딕스 B**: 화자 구분(diarization) 성능을 평가하며, 평균 오류율이 2% 미만으로 나타나, 화자 구분이 효과적으로 이루어졌음을 나타냅니다.
   - **어펜딕스 C**: 호스트와 게스트의 역할을 자동으로 식별하는 방법론을 설명하며, 모델의 정확도가 88%에 달함을 보여줍니다.

### Insights from Figures, Tables, and Appendices

1. **Figures and Diagrams**
   - **Figure 1**: Visualizes the distribution of various topics, showing that certain categories (e.g., Sports, Religion) have coherent thematic sub-communities. In contrast, topics like "Black Lives Matter" span multiple categories.
   - **Figure 2**: Visualizes the podcast-guest network, indicating that Business, Sports, and News categories are closely interconnected, while Religion and Society categories show looser connections.
   - **Figure 3**: Analyzes the podcast ecosystem's response to the George Floyd incident over time, revealing a rapid increase in attention immediately following the event, followed by a gradual decline.

2. **Tables**
   - **Table 1**: Measures modularity by category, indicating that Business and Sports categories are more tightly connected, suggesting more frequent guest sharing within these categories.
   - **Table 2**: Lists key words associated with racial justice topics, showing which categories primarily discuss these themes.
   - **Table 3**: Summarizes the average category proportions for each topic, indicating which topics are more prevalent in specific categories.

3. **Appendices**
   - **Appendix A**: Describes the methodology for evaluating transcription quality, with an average word error rate (WER) of 3%, indicating high-quality transcriptions.
   - **Appendix B**: Evaluates the performance of speaker diarization, showing an average error rate of less than 2%, indicating effective speaker differentiation.
   - **Appendix C**: Explains the methodology for automatically identifying host and guest roles, with a model accuracy of 88%, demonstrating reliable role identification.

These insights collectively highlight the structure and dynamics of the podcast ecosystem, revealing how topics are interconnected across categories and how the community responds to significant events.

<br/>
# refer format:
### BibTeX 

```bibtex
@article{litterer2025mapping,
  title={Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus},
  author={Litterer, Benjamin and Jurgens, David and Card, Dallas},
  journal={arXiv preprint arXiv:2411.07892},
  year={2025},
  url={https://arxiv.org/abs/2411.07892}
}
```

### 시카고 스타일

Benjamin Litterer, David Jurgens, and Dallas Card. 2025. "Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus." arXiv preprint arXiv:2411.07892. https://arxiv.org/abs/2411.07892.
