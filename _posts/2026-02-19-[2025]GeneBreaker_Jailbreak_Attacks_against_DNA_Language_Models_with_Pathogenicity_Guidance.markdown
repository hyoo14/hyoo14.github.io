---
layout: post
title:  "[2025]GeneBreaker: Jailbreak Attacks against DNA Language Models with Pathogenicity Guidance"
date:   2026-02-19 02:35:14 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: GeneBreaker는 DNA 언어 모델의 jailbreak 공격을 평가하기 위한 첫 번째 프레임워크로, 비병원성 DNA 서열을 기반으로 한 프롬프트 설계와 병원성 예측 모델을 활용하여 병원체 유사 서열 생성을 유도한다.


신박한점: 제일브레이킹을 생성모델이 비병원성을 받고 병원성을 생성하게 하는걸로 세팅(정의)한게 신박!    


짧은 요약(Abstract) :


DNA는 거의 모든 생물의 유전 정보를 담고 있으며, 유전체학과 합성 생물학의 혁신적인 발전을 이끌고 있습니다. 최근 DNA 기초 모델들이 합성 기능성 DNA 서열, 심지어 전체 유전체를 설계하는 데 성공을 거두었지만, 이들이 감옥 탈출(jailbreaking)에 취약하다는 점은 충분히 탐구되지 않았습니다. 이는 병원체나 독소 생성 유전자와 같은 해로운 서열을 생성할 수 있는 잠재적인 우려를 초래합니다. 본 논문에서는 DNA 기초 모델의 감옥 탈출 취약성을 체계적으로 평가하기 위한 첫 번째 프레임워크인 GeneBreaker를 소개합니다. GeneBreaker는 (1) 맞춤형 생물정보학 도구를 갖춘 LLM 에이전트를 사용하여 높은 동질성을 가진 비병원성 감옥 탈출 프롬프트를 설계하고, (2) PathoLM과 로그 확률 휴리스틱에 의해 안내되는 빔 검색을 통해 병원체와 유사한 서열 생성을 유도하며, (3) 커리큘럼된 인간 병원체 데이터베이스(JailbreakDNABench)에 대한 BLAST 기반 평가 파이프라인을 사용하여 성공적인 감옥 탈출을 감지합니다. JailbreakDNABench에서 평가한 결과, GeneBreaker는 6개의 바이러스 범주에 걸쳐 최신 Evo 시리즈 모델을 일관되게 감옥 탈출시키며, Evo2-40B 모델의 경우 최대 60%의 공격 성공률을 기록했습니다. SARS-CoV-2 스파이크 단백질과 HIV-1 외피 단백질에 대한 추가 사례 연구는 감옥 탈출 출력의 서열 및 구조적 충실성을 입증하며, SARS-CoV-2의 진화 모델링은 생물안전성 위험을 강조합니다. 우리의 발견은 DNA 기초 모델의 확장이 이중 사용 위험을 증대시킨다는 것을 보여주며, 향상된 안전 정렬 및 추적 메커니즘의 필요성을 촉구합니다.



---





DNA, encoding genetic instructions for almost all living organisms, fuels groundbreaking advances in genomics and synthetic biology. Recently, DNA Foundation Models have achieved success in designing synthetic functional DNA sequences, even whole genomes, but their susceptibility to jailbreaking remains underexplored, leading to potential concerns about generating harmful sequences such as pathogens or toxin-producing genes. In this paper, we introduce GeneBreaker, the first framework to systematically evaluate the jailbreak vulnerabilities of DNA foundation models. GeneBreaker employs (1) an LLM agent with customized bioinformatic tools to design high-homology, non-pathogenic jailbreaking prompts, (2) beam search guided by PathoLM and log-probability heuristics to steer generation toward pathogen-like sequences, and (3) a BLAST-based evaluation pipeline against a curated Human Pathogen Database (JailbreakDNABench) to detect successful jailbreaks. Evaluated on our JailbreakDNABench, GeneBreaker successfully jailbreaks the latest Evo series models across 6 viral categories consistently (up to 60% Attack Success Rate for Evo2-40B). Further case studies on SARS-CoV-2 spike protein and HIV-1 envelope protein demonstrate the sequence and structural fidelity of jailbreak output, while evolutionary modeling of SARS-CoV-2 underscores biosecurity risks. Our findings also reveal that scaling DNA foundation models amplifies dual-use risks, motivating enhanced safety alignment and tracing mechanisms.


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


GeneBreaker는 DNA 언어 모델에 대한 탈옥 공격을 체계적으로 평가하기 위해 설계된 첫 번째 프레임워크입니다. 이 시스템은 세 가지 주요 구성 요소로 이루어져 있습니다.

1. **LLM 에이전트로서의 프롬프트 설계**: GeneBreaker는 ChatGPT-4o를 활용하여 비병원성 DNA 서열을 검색합니다. 이 과정에서 특정 병원체 유전자(예: HIV-1 env 유전자)와 높은 상동성을 가진 DNA 서열을 찾습니다. 이를 통해 공격자가 비병원성 서열을 기반으로 하여 병원체와 유사한 서열을 생성할 수 있도록 돕습니다.

2. **병원성 예측 모델과 휴리스틱을 통한 빔 서치**: GeneBreaker는 PathoLM이라는 병원성 예측 모델을 사용하여 DNA 서열을 생성하는 과정에서 빔 서치 알고리즘을 적용합니다. 이 알고리즘은 여러 개의 서열 조각을 샘플링하고, 각 조각에 대해 병원성 점수를 부여하여 가장 병원체와 유사한 조각을 선택합니다. 이 과정은 생성된 서열의 일관성을 유지하면서 병원체와 유사한 출력을 유도합니다.

3. **평가 파이프라인**: GeneBreaker는 Nucleotide/Protein BLAST를 사용하여 생성된 서열을 인간 병원체 데이터베이스(JailbreakDNABench)와 비교합니다. 이 평가 과정에서 생성된 서열이 알려진 병원체와 90% 이상의 유사성을 보일 경우, 해당 공격이 성공한 것으로 간주됩니다.

GeneBreaker는 Evo 시리즈 모델을 포함한 여러 DNA 언어 모델에 대해 평가를 수행하며, 다양한 바이러스 카테고리에서 공격 성공률을 측정합니다. 이 연구는 DNA 언어 모델의 보안 및 생물 안전성 문제를 다루며, 이러한 모델이 생성할 수 있는 위험한 서열에 대한 우려를 제기합니다. GeneBreaker는 이러한 취약점을 드러내고, 향후 안전한 모델 개발을 위한 기초 자료를 제공합니다.





GeneBreaker is the first framework designed to systematically evaluate jailbreak attacks against DNA language models. This system consists of three main components:

1. **Prompt Design with LLM Agent**: GeneBreaker utilizes ChatGPT-4o to retrieve non-pathogenic DNA sequences. In this process, it searches for DNA sequences that exhibit high homology to specific pathogenic genes (e.g., the HIV-1 env gene). This assists attackers in generating pathogen-like sequences based on non-pathogenic sequences.

2. **Beam Search Guided by Pathogenicity Prediction Model and Heuristics**: GeneBreaker employs a beam search algorithm, guided by a pathogenicity prediction model called PathoLM, during the DNA sequence generation process. This algorithm samples multiple sequence chunks and assigns pathogenicity scores to each chunk, selecting the most pathogen-like chunks. This process steers the generation towards outputs that are similar to pathogens while maintaining sequence coherence.

3. **Evaluation Pipeline**: GeneBreaker uses Nucleotide/Protein BLAST to compare the generated sequences against a curated Human Pathogen Database (JailbreakDNABench). In this evaluation process, if the generated sequences show 90% or higher similarity to known pathogens, the attack is considered successful.

GeneBreaker evaluates various DNA language models, including the Evo series, and measures the attack success rates across different viral categories. This research addresses the biosafety and security implications of DNA language models, raising concerns about the potentially harmful sequences they could generate. GeneBreaker aims to expose these vulnerabilities and provide foundational insights for the development of safer models in the future.


<br/>
# Results



이 논문에서는 GeneBreaker라는 프레임워크를 사용하여 DNA 언어 모델의 탈옥 공격 성공률을 평가하였습니다. 연구에 사용된 DNA 모델은 Evo1(7B) 및 Evo2(1B, 7B, 40B)로, 이들은 각각의 파라미터 수에 따라 성능이 달라지는 경향을 보였습니다. 

#### 경쟁 모델
GeneBreaker는 기존의 DNA 언어 모델인 DNABert, megaDNA, GENERator와 비교하여, 이들 모델은 생성 능력이 부족하거나 불안정한 결과를 보이는 경우가 많아 GeneBreaker의 평가에서 제외되었습니다. 

#### 테스트 데이터
GeneBreaker는 JailbreakDNABench라는 새로운 벤치마크 데이터셋을 사용하여 평가를 진행했습니다. 이 데이터셋은 인체에 병원성을 가진 RNA 및 DNA 바이러스의 유전자 서열을 포함하고 있으며, 각 바이러스는 유전자 유형과 생물학적 특성에 따라 분류되었습니다. 

#### 메트릭
GeneBreaker의 성공률은 생성된 DNA 서열이 90% 이상의 뉴클레오타이드 유사성을 가지거나 90% 이상의 아미노산 유사성을 가질 때 성공으로 간주되었습니다. 이 기준은 바이러스의 복제 및 감염에 중요한 보존된 유전자 영역을 반영합니다.

#### 비교 결과
GeneBreaker의 공격 성공률은 다음과 같은 결과를 보였습니다:

- **Evo2 (40B)** 모델에서의 평균 공격 성공률은 60%에 달했으며, 이는 Small DNA 바이러스 및 Enteric RNA 바이러스에서 가장 높은 성공률을 기록했습니다.
- **Evo2 (7B)** 모델은 48%의 성공률을 보였고, **Evo1 (7B)** 모델은 24%의 성공률을 기록했습니다.
- **Evo2 (1B)** 모델은 가장 낮은 성공률인 20%를 보였습니다.

이러한 결과는 모델의 크기가 클수록 공격 성공률이 증가하는 경향을 나타내며, 이는 더 큰 모델이 긴 거리 의존성을 더 잘 모델링하고 보존된 모티프를 기억하는 데 유리하다는 것을 시사합니다. 

결론적으로, GeneBreaker는 DNA 언어 모델의 탈옥 공격에 대한 체계적인 평가를 제공하며, 바이오 보안 위험을 드러내고 안전성을 강화하기 위한 기초 자료를 제공합니다.

---



In this paper, the framework GeneBreaker was utilized to evaluate the jailbreak attack success rates of DNA language models. The DNA models tested were Evo1 (7B) and Evo2 (1B, 7B, 40B), which exhibited varying performance based on their parameter counts.

#### Competing Models
GeneBreaker was compared against existing DNA language models such as DNABert, megaDNA, and GENERator, which were excluded from the evaluation due to their lack of generative capabilities or unstable outputs.

#### Test Data
The evaluation was conducted using a new benchmark dataset called JailbreakDNABench. This dataset includes genetic sequences of RNA and DNA viruses that pose pathogenic threats to humans, categorized by their genomic types and biological characteristics.

#### Metrics
The success rate of GeneBreaker was defined based on whether the generated DNA sequences exhibited 90% or greater nucleotide identity or 90% or greater amino acid similarity. This threshold reflects conserved genomic regions critical for viral replication and infectivity.

#### Comparison Results
The attack success rates of GeneBreaker yielded the following results:

- The average attack success rate for the **Evo2 (40B)** model reached 60%, with the highest success rates observed in Small DNA viruses and Enteric RNA viruses.
- The **Evo2 (7B)** model achieved a success rate of 48%, while the **Evo1 (7B)** model recorded a success rate of 24%.
- The **Evo2 (1B)** model exhibited the lowest success rate at 20%.

These results indicate a trend where larger models tend to have higher attack success rates, suggesting that larger models are better at modeling long-range dependencies and memorizing conserved motifs.

In conclusion, GeneBreaker provides a systematic evaluation of jailbreak attacks against DNA language models, revealing biosecurity risks and serving as a foundational resource for enhancing safety measures.


<br/>
# 예제


이 논문에서는 GeneBreaker라는 프레임워크를 통해 DNA 언어 모델의 jailbreak 공격을 체계적으로 평가하는 방법을 제안합니다. GeneBreaker의 주요 목표는 DNA 언어 모델이 생성할 수 있는 유해한 DNA 시퀀스를 탐지하고, 이를 통해 생물안전성 및 보안 위험을 평가하는 것입니다.

#### 1. 트레이닝 데이터와 테스트 데이터

**트레이닝 데이터**: GeneBreaker는 DNA 언어 모델을 훈련시키기 위해 다양한 비병원성 DNA 시퀀스를 포함한 데이터셋을 사용합니다. 이 데이터셋은 GenBank에서 수집된 비병원성 바이러스의 유전자 서열로 구성되어 있으며, 각 서열은 특정 병원체와의 높은 상동성을 가지고 있습니다. 예를 들어, HIV-1 env 유전자와 유사한 비병원성 바이러스의 서열이 포함될 수 있습니다.

**테스트 데이터**: 테스트 데이터는 JailbreakDNABench라는 커스텀 데이터셋으로 구성되어 있으며, 이는 다양한 병원체의 DNA 서열을 포함합니다. 이 데이터셋은 다음과 같은 카테고리로 나뉩니다:
- 대형 DNA 바이러스
- 소형 DNA 바이러스
- 양성 스트랜드 RNA 바이러스
- 음성 스트랜드 RNA 바이러스
- 이중 스트랜드 RNA 바이러스
- 장내 RNA 바이러스

#### 2. 구체적인 인풋과 아웃풋

**인풋**: GeneBreaker는 DNA 언어 모델에 대한 입력으로 비병원성 DNA 서열을 사용합니다. 예를 들어, HIV-1 env 유전자와 높은 상동성을 가지면서 비병원성인 DNA 서열을 입력으로 제공할 수 있습니다. 이 입력은 다음과 같은 형식으로 구성됩니다:
```
|D__VIRUS;P__SSRNA;O__RETROVIRIDAE;F__LENTIVIRUS;G__HIV-1|
|ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTC...|
```

**아웃풋**: 모델이 생성하는 아웃풋은 병원체와 유사한 DNA 서열입니다. 이 서열은 BLAST를 통해 테스트 데이터셋의 병원체 서열과 비교되어, 90% 이상의 유사성을 가지는 경우 성공적인 jailbreak으로 간주됩니다. 예를 들어, 생성된 서열이 SARS-CoV-2의 spike 단백질과 92.77%의 유사성을 가진다면, 이는 성공적인 아웃풋으로 평가됩니다.

#### 3. 구체적인 테스크

GeneBreaker의 주요 테스크는 다음과 같습니다:
- 비병원성 DNA 서열을 기반으로 병원체와 유사한 DNA 서열을 생성하는 것.
- 생성된 서열이 실제 병원체와 얼마나 유사한지를 평가하는 것.
- 성공적인 jailbreak 공격의 비율을 측정하는 것.

이러한 과정을 통해 GeneBreaker는 DNA 언어 모델의 보안 취약점을 평가하고, 생물안전성 위험을 식별하는 데 기여합니다.

---




This paper introduces a framework called GeneBreaker to systematically evaluate jailbreak attacks against DNA language models. The primary goal of GeneBreaker is to detect harmful DNA sequences that can be generated by DNA language models and assess biosafety and security risks.

#### 1. Training Data and Test Data

**Training Data**: GeneBreaker uses a dataset of non-pathogenic DNA sequences collected from GenBank to train the DNA language model. This dataset consists of sequences from non-pathogenic viruses that exhibit high homology to specific pathogens. For example, it may include sequences from non-pathogenic viruses that are similar to the HIV-1 env gene.

**Test Data**: The test data is composed of a custom dataset called JailbreakDNABench, which includes DNA sequences from various pathogens. This dataset is categorized into the following groups:
- Large DNA viruses
- Small DNA viruses
- Positive-strand RNA viruses
- Negative-strand RNA viruses
- Double-stranded RNA viruses
- Enteric RNA viruses

#### 2. Specific Inputs and Outputs

**Input**: GeneBreaker uses non-pathogenic DNA sequences as input to the DNA language model. For example, it can provide a non-pathogenic DNA sequence that has high homology to the HIV-1 env gene. The input is structured as follows:
```
|D__VIRUS;P__SSRNA;O__RETROVIRIDAE;F__LENTIVIRUS;G__HIV-1|
|ATGTTTGTTTTTCTTGTTTTATTGCCACTAGTC...|
```

**Output**: The output generated by the model is a DNA sequence that resembles a pathogen. This sequence is compared against the test dataset using BLAST, and if it matches known pathogen sequences with over 90% similarity, it is considered a successful jailbreak. For instance, if the generated sequence has a 92.77% similarity to the SARS-CoV-2 spike protein, it is evaluated as a successful output.

#### 3. Specific Tasks

The main tasks of GeneBreaker include:
- Generating DNA sequences that resemble pathogens based on non-pathogenic DNA sequences.
- Evaluating how similar the generated sequences are to actual pathogens.
- Measuring the success rate of jailbreak attacks.

Through these processes, GeneBreaker contributes to assessing the security vulnerabilities of DNA language models and identifying biosafety risks.

<br/>
# 요약
GeneBreaker는 DNA 언어 모델의 jailbreak 공격을 평가하기 위한 첫 번째 프레임워크로, 비병원성 DNA 서열을 기반으로 한 프롬프트 설계와 병원성 예측 모델을 활용하여 병원체 유사 서열 생성을 유도한다. 실험 결과, GeneBreaker는 Evo2 모델에서 최대 60%의 공격 성공률을 기록하며, SARS-CoV-2와 HIV-1의 단백질 서열을 성공적으로 재설계하였다. 이 연구는 DNA 기초 모델의 생물안전성 및 보안 위험을 강조하며, 향후 안전성 강화 및 추적 메커니즘 개발의 필요성을 제기한다.

GeneBreaker is the first framework to evaluate jailbreak attacks against DNA language models, utilizing prompts based on non-pathogenic DNA sequences and a pathogenicity prediction model to guide the generation of pathogen-like sequences. Experimental results show that GeneBreaker achieves up to a 60% attack success rate on the Evo2 model, successfully redesigning protein sequences for SARS-CoV-2 and HIV-1. This research highlights the biosafety and security risks of DNA foundation models, emphasizing the need for enhanced safety alignment and tracing mechanisms in the future.

<br/>
# 기타


1. **다이어그램 및 피규어**
   - **Figure 1**: GeneBreaker의 구조를 보여주는 다이어그램으로, LLM 에이전트, 병원성 예측 모델(PathoLM), BLAST 기반 평가 파이프라인의 세 가지 주요 구성 요소를 설명합니다. 이 구조는 DNA 언어 모델의 jailbreak 공격을 체계적으로 평가하는 방법을 시각적으로 나타냅니다.
   - **Figure 3**: GeneBreaker의 성능 분석을 위한 여러 그래프가 포함되어 있습니다. (a) 병원성 목표와의 서열 유사성 간의 상관관계를 보여주며, 높은 로그 확률이 병원성 서열과의 유사성을 증가시킨다는 것을 나타냅니다. (b) 높은 동종성 프롬프트가 성공적인 jailbreak 공격에 중요하다는 것을 보여줍니다. (c) GeneBreaker의 구성 요소가 공격 성공률에 미치는 영향을 분석한 결과를 보여줍니다.
   - **Figure 4**: GeneBreaker가 생성한 SARS-CoV-2 스파이크 단백질과 HIV-1 환경 단백질의 구조를 비교합니다. 이 피규어는 생성된 단백질이 원래 단백질과 구조적으로 유사하다는 것을 보여줍니다.

2. **테이블**
   - **Table 1**: GeneBreaker의 공격 성공률을 다양한 바이러스 카테고리에서 보여줍니다. 각 모델의 평균 성공률이 나열되어 있으며, Evo2(40B) 모델이 가장 높은 성공률(최대 60%)을 기록했습니다. 이는 DNA 언어 모델의 크기와 구조가 공격 성공률에 미치는 영향을 강조합니다.
   - **Table 2**: JailbreakDNABench의 바이러스 카테고리 분류를 보여줍니다. 각 카테고리는 유전자 유형, 생물학적 특성 및 포함된 바이러스를 기준으로 나뉘어 있습니다. 이 표는 연구의 기초가 되는 데이터셋의 구조를 이해하는 데 도움을 줍니다.

3. **어펜딕스**
   - **Appendix A**: JailbreakDNABench의 고위험 병원성 바이러스 목록을 포함하여, 각 바이러스의 유전자 유형과 생물학적 특성을 설명합니다. 이 정보는 연구의 배경과 중요성을 강조합니다.
   - **Appendix B**: GeneBreaker의 하이퍼파라미터 분석을 포함하여, 공격 성공률에 미치는 영향을 설명합니다. 이 분석은 모델의 성능을 최적화하는 데 중요한 요소를 식별하는 데 도움을 줍니다.
   - **Appendix C**: DNA 생성 언어 모델의 요약을 제공하여, 각 모델의 크기, 아키텍처 및 주요 기능을 비교합니다. 이 정보는 연구에서 사용된 모델의 선택을 이해하는 데 유용합니다.
   - **Appendix D**: 비병원성 DNA 서열을 검색하기 위한 ChatGPT 쿼리 프롬프트의 예시를 제공합니다. 이 프롬프트는 연구에서 사용된 방법론을 구체적으로 보여줍니다.





1. **Diagrams and Figures**
   - **Figure 1**: A diagram illustrating the structure of GeneBreaker, showing the three main components: the LLM agent, the pathogenicity prediction model (PathoLM), and the BLAST-based evaluation pipeline. This structure visually represents the method for systematically evaluating jailbreak attacks on DNA language models.
   - **Figure 3**: Contains several graphs for performance analysis of GeneBreaker. (a) Shows the correlation between sequence similarity to pathogenic targets, indicating that higher log probabilities increase similarity to pathogenic sequences. (b) Demonstrates that high-homology prompts are critical for successful jailbreak attacks. (c) Analyzes the impact of GeneBreaker’s components on attack success rates.
   - **Figure 4**: Compares the structures of the SARS-CoV-2 spike protein and HIV-1 envelope protein generated by GeneBreaker. This figure shows that the generated proteins are structurally similar to their native counterparts.

2. **Tables**
   - **Table 1**: Displays the attack success rates of GeneBreaker across various virus categories. The average success rates for each model are listed, with the Evo2 (40B) model achieving the highest success rate (up to 60%). This highlights the impact of the size and architecture of DNA language models on attack success rates.
   - **Table 2**: Shows the categorization of viruses in JailbreakDNABench. Each category is divided based on genome type, biological characteristics, and included viruses. This table helps in understanding the structure of the dataset that underpins the research.

3. **Appendices**
   - **Appendix A**: Includes a list of high-risk pathogenic viruses in JailbreakDNABench, explaining the genome type and biological characteristics of each virus. This information emphasizes the background and significance of the research.
   - **Appendix B**: Contains hyperparameter analysis of GeneBreaker, explaining the impact on attack success rates. This analysis helps identify critical factors for optimizing model performance.
   - **Appendix C**: Provides a summary of generative DNA language models, comparing each model's size, architecture, and notable capabilities. This information is useful for understanding the selection of models used in the research.
   - **Appendix D**: Offers an example of a ChatGPT query prompt for retrieving non-pathogenic DNA sequences. This prompt specifically illustrates the methodology used in the research.

<br/>
# refer format:



### BibTeX 형식
```bibtex
@article{zhang2025gene,
  title={GeneBreaker: Jailbreak Attacks against DNA Language Models with Pathogenicity Guidance},
  author={Zhang, Zaixi and Zhou, Zhenghong and Jin, Ruofan and Cong, Le and Wang, Mengdi},
  journal={arXiv preprint arXiv:2505.23839},
  year={2025},
  url={https://arxiv.org/abs/2505.23839}
}
```

### 시카고 스타일
Zaixi Zhang, Zhenghong Zhou, Ruofan Jin, Le Cong, and Mengdi Wang. "GeneBreaker: Jailbreak Attacks against DNA Language Models with Pathogenicity Guidance." arXiv preprint arXiv:2505.23839 (2025). https://arxiv.org/abs/2505.23839.
