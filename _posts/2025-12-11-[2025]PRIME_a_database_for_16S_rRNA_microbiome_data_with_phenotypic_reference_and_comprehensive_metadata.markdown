---
layout: post
title:  "[2025]PRIME: a database for 16S rRNA microbiome data with phenotypic reference and comprehensive metadata"
date:   2025-12-11 17:52:42 -0000
categories: study
---

{% highlight ruby %}

한줄 요약: PRIME 데이터베이스는 16S rRNA 미생물군집 데이터를 수집하고 표준화하여 53,449개의 샘플을 포함하며, 111개의 공개 연구에서 수집된 정보를 제공합니다.


짧은 요약(Abstract) :


PRIME(Phenotypic Reference for Integrated Microbiome Enrichment)는 인간의 마이크로바이옴 16S rRNA 증폭 서열 데이터의 표준화된 데이터베이스로, 교차 연구 분석, 재현성 및 표현형 기반 발견을 촉진하기 위해 설계되었습니다. PRIME은 111개의 공개 연구에서 53,449개의 샘플을 집계하며, 93개의 신체 부위와 101개의 표현형 범주를 포함하고 있습니다. 샘플 수준의 메타데이터는 질병 상태, 인구 통계, 신체 부위, 시퀀싱 프로토콜 및 실험 설계와 같은 세부 사항을 포함하여 철저하게 조화되었습니다. 각 샘플은 SILVA(138.2) 및 Greengenes2(2024.09) 참조 데이터베이스를 사용하여 생성된 세균의 풍부도 프로필을 포함하며, 결과는 관찰된 풍부도(읽기 수)와 상대 풍부도(비율)로 여러 세분화 수준에서 보고됩니다. PRIME의 주요 강점은 광범위한 수동 큐레이션으로, 연구 간의 표현형 및 맥락 메타데이터를 표준화하여 정밀한 쿼리 및 강력한 표현형 기반 비교를 가능하게 합니다. 사용자는 현대적인 웹 인터페이스를 통해 데이터베이스를 상호작용적으로 탐색하고, 메타데이터 필드별로 데이터를 필터링 및 시각화하며, 맞춤형 하위 집합을 다운로드할 수 있습니다. 프로그래밍 방식의 접근은 RESTful API와 R 패키지를 통해 지원됩니다. PRIME은 마이크로바이옴 데이터 통합을 촉진하고 있으며, 새로운 연구 및 기능을 통합하기 위해 지속적으로 업데이트됩니다. 데이터베이스는 https://primedb.sjtu.edu.cn에서 무료로 이용할 수 있습니다.



PRIME (Phenotypic Reference for Integrated Microbiome Enrichment) is a curated and standardized database of human microbiome 16S rRNA amplicon sequencing data, designed to facilitate cross-study analysis, reproducibility, and phenotype-driven discovery. PRIME aggregates 53,449 samples from 111 public studies, covering 93 body sites and 101 phenotypic categories, with detailed harmonization of sample-level metadata such as disease status, demographics, body sites, sequencing protocols, and experimental design. Each sample includes taxonomic abundance profiles generated via a consistent pipeline using both SILVA (138.2) and Greengenes2 (2024.09) reference databases, with results reported at multiple taxonomic levels as observed abundances (read counts) and relative abundances (proportions). A major strength of PRIME is its extensive manual curation, which standardizes phenotypic and contextual metadata across studies, enabling precise querying and robust phenotype-based comparisons. Users can interactively explore the database through a modern web interface, filter and visualize data by metadata fields, and download customized subsets. Programmatic access is supported via RESTful APIs and an R package. PRIME aims to advance microbiome data integration and is continuously updated to incorporate new studies and features. The database is freely available at https://primedb.sjtu.edu.cn.


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



PRIME 데이터베이스는 인간 미생물군의 16S rRNA 엠플리콘 시퀀싱 데이터를 수집하고 표준화하기 위해 여러 단계의 방법론을 사용합니다. 이 데이터베이스는 다음과 같은 주요 구성 요소로 이루어져 있습니다.

1. **데이터 수집**: PRIME은 NCBI BioProject 데이터베이스에서 "human microbiome" 및 "human metagenome" 키워드를 사용하여 공개적으로 이용 가능한 연구를 체계적으로 검색합니다. 이 과정에서 350개의 후보 연구를 검토하고, 135개의 연구가 포함 기준을 충족하는지 확인합니다. 최종적으로 Illumina 플랫폼에서 시퀀싱된 111개의 연구에서 53,449개의 샘플을 선택합니다.

2. **생물정보학 파이프라인**: 모든 원시 16S rRNA 엠플리콘 시퀀싱 데이터는 SRA Toolkit을 사용하여 다운로드되고, Cutadapt를 통해 프라이머 및 어댑터 서열이 제거됩니다. FastQC를 사용하여 읽기 품질을 평가하고, QIIME 2를 통해 처리된 읽기를 가져옵니다. DADA2 플러그인을 사용하여 노이즈 제거 및 정확한 아믈리콘 서열 변형(ASV)을 추론합니다. 최종 출력은 여러 분류 수준에서의 ASV 풍부도 테이블을 포함합니다.

3. **메타데이터 큐레이션 및 표준화**: 모든 포함된 연구의 원본 샘플 수준 메타데이터는 NCBI SRA Run Selector에서 검색됩니다. 연구의 관련 출판물을 수동으로 확인하고, SRA 메타데이터, 연구 초록 및 출판물에서 설명 필드를 체계적으로 파싱하여 해석 가능한 메타데이터 요소를 추출합니다. 메타데이터 필드는 통제된 어휘를 사용하여 표준화되며, 동의어 및 실험 조건을 통합하여 일관성을 유지합니다.

4. **세부 정보 페이지 및 사용자 인터페이스**: PRIME 웹 인터페이스는 사용자가 데이터베이스를 탐색하고 접근할 수 있는 인터랙티브한 플랫폼을 제공합니다. 사용자는 프로젝트, 샘플, 표현형, 신체 부위, 국가, 분류군 및 다운로드 페이지를 통해 메타데이터 및 풍부도 테이블을 검색하고 필터링할 수 있습니다. AI 기반의 채팅 도우미가 실시간으로 사용자 질문에 답변합니다.

5. **프로그램적 접근**: PRIME은 RESTful API와 R 패키지를 통해 프로그램적 접근을 지원합니다. API는 표준 REST 원칙을 따르며, 사용자가 프로젝트, 샘플, 메타데이터, 분류군, 표현형 및 풍부도 테이블을 쿼리할 수 있도록 JSON 형식으로 데이터를 제공합니다.

이러한 방법론을 통해 PRIME은 고품질의 메타데이터가 풍부한 16S rRNA 데이터셋을 제공하며, 연구자들이 미생물군 연구를 보다 효과적으로 수행할 수 있도록 지원합니다.

---




The PRIME database employs a multi-step methodology to collect and standardize 16S rRNA amplicon sequencing data from human microbiomes. The database consists of the following key components:

1. **Data Collection**: PRIME systematically searches the NCBI BioProject database using the keywords "human microbiome" and "human metagenome" to identify publicly available studies suitable for standardized processing. A total of 350 candidate studies are reviewed, and 135 studies are assessed for inclusion based on relevance and metadata availability. Ultimately, 111 studies sequenced on Illumina platforms are selected, comprising 53,449 samples.

2. **Bioinformatics Pipeline**: All raw 16S rRNA amplicon sequencing data are downloaded using the SRA Toolkit, and primer and adapter sequences are removed using Cutadapt. Read quality is assessed with FastQC, and processed reads are imported using QIIME 2. Denoising is performed using the DADA2 plugin, which infers exact amplicon sequence variants (ASVs). The final outputs include ASV abundance tables at multiple taxonomic levels.

3. **Metadata Curation and Standardization**: Original sample-level metadata for all included studies are retrieved from the NCBI SRA Run Selector. The associated publications are manually confirmed, and descriptive fields from SRA metadata, study abstracts, and publications are systematically parsed to extract interpretable metadata elements. Metadata fields are standardized using controlled vocabulary, ensuring consistency across studies.

4. **Detail Pages and User Interface**: The PRIME web interface provides an interactive platform for users to access and explore the database. Users can search and filter metadata and abundance tables through project, sample, phenotype, body site, country, taxa, and download pages. An AI-powered chat assistant is available to answer user questions in real-time.

5. **Programmatic Access**: PRIME supports programmatic access via a RESTful API and an accompanying R package. The API adheres to standard REST principles and delivers data in a structured JSON format, allowing users to query projects, samples, metadata, taxa, phenotypes, and abundance tables.

Through these methodologies, PRIME provides high-quality, metadata-rich 16S rRNA datasets, facilitating researchers in conducting microbiome research more effectively.


<br/>
# Results



PRIME 데이터베이스는 111개의 공개 연구에서 수집된 53,449개의 인간 관련 16S rRNA 시퀀싱 샘플로 구성되어 있습니다. 이 데이터베이스는 다양한 체계적 접근 방식을 통해 샘플 수준의 메타데이터를 수집하고 표준화하여, 연구자들이 교차 연구 분석을 수행할 수 있도록 지원합니다. PRIME은 다음과 같은 주요 기능을 제공합니다:

1. **경쟁 모델 비교**: PRIME은 기존의 여러 마이크로바이옴 데이터베이스와 비교하여, 샘플 수준의 메타데이터를 수동으로 큐레이션하고 표준화하여 제공하는 점에서 차별화됩니다. 예를 들어, GMrepo와 mBodyMap은 각각 인간 장내 미생물과 다양한 신체 부위의 미생물 분포를 다루지만, 샘플 수준의 풍부도 테이블을 내보내는 기능이 부족합니다. 반면, PRIME은 이러한 기능을 제공하여 연구자들이 보다 유용하게 데이터를 활용할 수 있도록 합니다.

2. **테스트 데이터**: PRIME은 16S rRNA 아밀리콘 시퀀싱 데이터를 기반으로 하며, SILVA와 Greengenes2 참조 데이터베이스를 사용하여 세균의 분류학적 풍부도 프로파일을 생성합니다. 이 데이터는 여러 세부 수준에서 제공되며, 관찰된 풍부도(읽기 수)와 상대 풍부도(비율)로 보고됩니다.

3. **메트릭**: PRIME은 데이터의 품질을 평가하기 위해 다섯 단계의 품질 점수 체계를 사용합니다. 이 점수는 원시 읽기 품질 프로파일과 DADA2 처리 후 유지된 입력 읽기 비율을 기반으로 합니다. 또한, 각 샘플에 대해 메타데이터 필드를 큐레이션하여, 샘플의 출처, 호스트 메타데이터(예: 질병 상태, 성별, 나이 등), 연구 설계(예: 시간 시리즈, 비교) 등을 포함합니다.

4. **비교**: PRIME은 기존의 데이터베이스와 비교하여, 수동 큐레이션된 메타데이터와 일관된 처리된 세균 풍부도 테이블을 제공함으로써, 연구자들이 이질적인 연구 간의 통합 분석을 수행할 수 있도록 지원합니다. PRIME은 RESTful API와 R 패키지를 통해 프로그램적 접근을 지원하여, 연구자들이 외부 분석 워크플로우에 쉽게 통합할 수 있도록 합니다.

결론적으로, PRIME은 고품질의 메타데이터가 풍부한 16S rRNA 데이터셋에 대한 접근을 용이하게 하여, 마이크로바이옴 연구의 발전에 기여할 것으로 기대됩니다.

---




The PRIME database comprises 53,449 human-associated 16S rRNA sequencing samples collected from 111 publicly available studies. This database supports researchers in conducting cross-study analyses by systematically collecting and standardizing sample-level metadata. PRIME offers the following key features:

1. **Comparison with Competing Models**: PRIME differentiates itself from several existing microbiome databases by providing manually curated and standardized sample-level metadata. For instance, GMrepo and mBodyMap focus on the human gut microbiome and microbial distributions across various body sites, respectively, but lack the functionality to export sample-level abundance tables. In contrast, PRIME offers this capability, enabling researchers to utilize the data more effectively.

2. **Test Data**: PRIME is based on 16S rRNA amplicon sequencing data and generates taxonomic abundance profiles using the SILVA and Greengenes2 reference databases. This data is provided at multiple taxonomic levels, reported as observed abundances (read counts) and relative abundances (proportions).

3. **Metrics**: To assess data quality, PRIME employs a five-level scoring system based on raw read quality profiles and the proportion of input reads retained after DADA2 processing. Additionally, metadata fields are curated for each sample, including information on the source, host metadata (e.g., disease status, sex, age), and study design (e.g., time series, comparison).

4. **Comparison**: PRIME supports integrative analyses across heterogeneous studies by providing manually curated metadata and consistently processed taxonomic abundance tables, which is a significant advantage over existing databases. PRIME also facilitates programmatic access through a RESTful API and an R package, allowing researchers to seamlessly integrate it into external analysis workflows.

In conclusion, PRIME is expected to contribute to the advancement of microbiome research by providing easy access to high-quality, metadata-rich 16S rRNA datasets at scale.


<br/>
# 예제



PRIME 데이터베이스는 16S rRNA 마이크로바이옴 데이터를 수집하고 표준화하여 제공하는 플랫폼입니다. 이 데이터베이스는 53,449개의 샘플을 포함하고 있으며, 각 샘플은 다양한 신체 부위와 표현형(phenotype) 카테고리에 대한 메타데이터를 포함하고 있습니다. PRIME의 주요 목표는 연구자들이 마이크로바이옴 데이터를 통합하고 비교할 수 있도록 돕는 것입니다.

#### 트레이닝 데이터와 테스트 데이터의 예시

1. **트레이닝 데이터**
   - **입력(Input)**: 
     - 샘플 메타데이터: 나이, 성별, 질병 상태, 샘플 수집 위치(예: 대장, 구강 등)
     - 시퀀싱 데이터: 16S rRNA 유전자 서열
   - **출력(Output)**: 
     - 각 샘플의 미생물 군집 구성: 특정 미생물의 상대적 풍부도(예: Firmicutes 30%, Bacteroidetes 20% 등)
     - 표현형 예측: 특정 질병(예: 비만, 당뇨병 등)과의 연관성

2. **테스트 데이터**
   - **입력(Input)**: 
     - 새로운 샘플 메타데이터: 나이, 성별, 질병 상태, 샘플 수집 위치
     - 새로운 시퀀싱 데이터: 16S rRNA 유전자 서열
   - **출력(Output)**: 
     - 새로운 샘플의 미생물 군집 구성 예측
     - 해당 샘플이 특정 질병과 관련이 있는지 여부 예측

#### 구체적인 테스크
- **미생물 군집 분석**: 샘플의 미생물 군집을 분석하여 특정 질병과의 연관성을 평가합니다.
- **표현형 예측**: 주어진 메타데이터를 기반으로 특정 질병의 발생 가능성을 예측합니다.
- **데이터 통합**: 다양한 연구에서 수집된 데이터를 통합하여 대규모 분석을 수행합니다.

---




The PRIME database is a platform that collects and standardizes 16S rRNA microbiome data. It contains 53,449 samples, each with metadata related to various body sites and phenotypic categories. The primary goal of PRIME is to assist researchers in integrating and comparing microbiome data.

#### Example of Training Data and Test Data

1. **Training Data**
   - **Input**: 
     - Sample metadata: age, gender, disease status, sample collection site (e.g., colon, oral cavity)
     - Sequencing data: 16S rRNA gene sequences
   - **Output**: 
     - Microbial community composition for each sample: relative abundance of specific microbes (e.g., Firmicutes 30%, Bacteroidetes 20%)
     - Phenotype prediction: association with specific diseases (e.g., obesity, diabetes)

2. **Test Data**
   - **Input**: 
     - New sample metadata: age, gender, disease status, sample collection site
     - New sequencing data: 16S rRNA gene sequences
   - **Output**: 
     - Prediction of microbial community composition for the new sample
     - Prediction of whether the sample is associated with a specific disease

#### Specific Tasks
- **Microbial Community Analysis**: Analyze the microbial community of samples to assess associations with specific diseases.
- **Phenotype Prediction**: Predict the likelihood of developing specific diseases based on given metadata.
- **Data Integration**: Integrate data collected from various studies to perform large-scale analyses.

<br/>
# 요약


PRIME 데이터베이스는 16S rRNA 미생물군집 데이터를 수집하고 표준화하여 53,449개의 샘플을 포함하며, 111개의 공개 연구에서 수집된 정보를 제공합니다. 데이터는 일관된 생물정보학 파이프라인을 통해 처리되며, 샘플 수준의 메타데이터는 수동으로 큐레이션되어 연구 간 비교를 용이하게 합니다. 사용자는 웹 인터페이스, RESTful API 및 R 패키지를 통해 데이터에 접근하고, 맞춤형 데이터 다운로드 기능을 활용할 수 있습니다.

---

The PRIME database aggregates and standardizes 16S rRNA microbiome data, encompassing 53,449 samples from 111 public studies. The data is processed through a consistent bioinformatic pipeline, and sample-level metadata is manually curated to facilitate comparisons across studies. Users can access the data through a web interface, RESTful API, and R package, with options for customized data downloads.

<br/>
# 기타



#### 다이어그램
PRIME 데이터베이스 아키텍처 다이어그램은 데이터 처리 및 접근 방식을 시각적으로 설명합니다. 이 다이어그램은 세 가지 핵심 데이터 유형(수동으로 큐레이션된 샘플 수준 메타데이터, 통합된 생물정보학 파이프라인을 통해 생성된 분류학적 풍부도 테이블, NCBI 분류학 식별자에 정확하게 매핑된 외부 분류학 주석)을 통합하는 방법을 보여줍니다. 또한, 웹 인터페이스, RESTful API, R 패키지와 같은 세 가지 사용자 접근 방식을 통해 데이터에 접근할 수 있는 방법을 설명합니다.

#### 피규어
1. **PRIME 웹 인터페이스 기능**: 이 피규어는 PRIME 웹 인터페이스의 주요 기능을 보여줍니다. 사용자는 프로젝트 및 샘플 검색, 필터링, 다운로드 기능을 통해 데이터에 접근할 수 있습니다. AI 어시스턴트 기능도 포함되어 있어 사용자가 실시간으로 도움을 받을 수 있습니다.
   
2. **상세 정보 페이지**: 각 데이터 카테고리에 대한 상세 정보 페이지는 관련 메타데이터와 상호작용 가능한 시각화를 제공합니다. 예를 들어, 특정 분류군의 상세 페이지는 관련 샘플과 표현형을 보여주며, 그 분류군의 상대적 풍부도를 시각화합니다.

#### 테이블
1. **신체 시스템에 따른 샘플 분류**: 이 테이블은 PRIME 데이터베이스에서 정의된 신체 시스템에 따라 샘플을 분류한 내용을 보여줍니다. 각 시스템에 대한 대표적인 신체 부위가 나열되어 있으며, 이는 연구자들이 특정 신체 부위와 관련된 미생물 데이터를 쉽게 찾을 수 있도록 돕습니다.

2. **메타데이터 필드**: PRIME 데이터베이스에서 큐레이션된 메타데이터 필드는 샘플링 사이트, 호스트 메타데이터, 연구 설계, 지리적 세부사항, 시퀀싱 매개변수 등 다양한 정보를 포함합니다. 이는 연구자들이 샘플을 보다 정확하게 이해하고 분석할 수 있도록 지원합니다.

#### 어펜딕스
어펜딕스에는 사용된 소프트웨어, 데이터 처리 방법, 메타데이터 표준화 과정 등이 포함되어 있습니다. 이 정보는 연구자들이 PRIME 데이터베이스를 활용할 때 필요한 기술적 세부사항을 제공합니다.

---

### Summary of Results and Insights

#### Diagrams
The diagram of the PRIME database architecture visually explains the data processing and access methods. It illustrates how three core data types (manually curated sample-level metadata, taxonomic abundance tables generated through a unified bioinformatics pipeline, and external taxonomic annotations accurately mapped to NCBI taxonomy identifiers) are integrated. Additionally, it describes the three user access modes: a web interface, RESTful API, and R package.

#### Figures
1. **PRIME Web Interface Features**: This figure showcases the main functionalities of the PRIME web interface. Users can access data through project and sample searches, filtering, and download options. An AI assistant feature is also included, allowing users to receive real-time assistance.

2. **Detail Pages**: The detail pages for various data categories present relevant metadata and interactive visualizations. For instance, a taxon detail page displays related samples and phenotypes, along with visualizations of its relative abundance.

#### Tables
1. **Sample Classification by Body Systems**: This table shows the classification of samples according to body systems defined in the PRIME database. Representative body sites for each system are listed, aiding researchers in easily locating microbiome data related to specific body areas.

2. **Metadata Fields**: The curated metadata fields in the PRIME database include various information such as sampling sites, host metadata, study designs, geographical details, and sequencing parameters. This supports researchers in accurately understanding and analyzing samples.

#### Appendix
The appendix includes details on the software used, data processing methods, and metadata standardization processes. This information provides the technical specifics necessary for researchers to effectively utilize the PRIME database.

<br/>
# refer format:
### BibTeX 



```bibtex
@article{zhang2025prime,
  title={PRIME: a database for 16S rRNA microbiome data with phenotypic reference and comprehensive metadata},
  author={Zhang, Zhizhuo and Zhao, Hongyu and Wang, Tao},
  journal={Nucleic Acids Research},
  volume={53},
  number={D1},
  pages={D1--D8},
  year={2025},
  publisher={Oxford University Press},
  doi={10.1093/nar/gkaf1057},
  url={https://doi.org/10.1093/nar/gkaf1057}
}
```

### 시카고 스타일

Zhizhuo Zhang, Hongyu Zhao, and Tao Wang. "PRIME: A Database for 16S rRNA Microbiome Data with Phenotypic Reference and Comprehensive Metadata." *Nucleic Acids Research* 53, no. D1 (2025): D1–D8. https://doi.org/10.1093/nar/gkaf1057.
