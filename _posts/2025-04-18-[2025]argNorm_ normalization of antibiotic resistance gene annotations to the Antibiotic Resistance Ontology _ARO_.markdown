---
layout: post
title:  "[2025]argNorm: normalization of antibiotic resistance gene annotations to the Antibiotic Resistance Ontology (ARO)"  
date:   2025-04-18 11:43:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 


매핑 테이블 만들어서 ARO 기반 도구 밑 데이터베이스  클래스 매핑   


짧은 요약(Abstract) :    

현재 사용되는 여러 ARG(항생제 내성 유전자) 주석 도구들은 유전자 이름과 약물 분류 체계가 제각각이라 결과 비교가 어렵습니다. 이를 해결하기 위해 연구진은 argNorm이라는 커맨드라인·파이썬 라이브러리를 개발해, 6개 주석 도구(8개 데이터베이스)의 모든 유전자를 **ARO(Antibiotic Resistance Ontology)**의 표준 ID와 약물 · 약물군 정보로 일괄 매핑합니다. argNorm은 도구별 출력에 동일한 ARO 기반 카테고리를 덧붙여 서로 다른 결과를 바로 비교‑분석할 수 있게 해 주며, hAMRonization과도 연동됩니다. 코드는 GitHub·PyPI·Bioconda·nf‑core에서 자유롭게 사용할 수 있습니다. ​  



Outputs from commonly used antimicrobial‑resistance‑gene (ARG) annotation tools employ inconsistent gene names and drug classifications, hindering direct comparison. argNorm—a command‑line tool and Python library—normalises detected genes from six annotation tools (eight databases) to the standard identifiers and hierarchy of the Antibiotic Resistance Ontology (ARO). By appending unified ARO IDs plus drug and drug‑class information, argNorm renders ARG results cross‑tool comparable and can be combined with hAMRonization for fully harmonised pipelines. The software is open‑source and available via GitHub, PyPI, Bioconda and nf‑core. ​  





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


1. **ARG → ARO 매핑 구축**   
   - 지원하는 8개 ARG 데이터베이스의 모든 서열을 RGI v6.0.3(CARD v4.0.0)로 분석해 각 유전자를 ARO ID에 연결했습니다.  
   - 대부분의 서열은 **Perfect/Strict/Loose** 등급으로 자동 매칭되지만, <1 %는 항생제 분류 불일치·미검출 등의 이유로 연구자가 직접 수동 교정했습니다.  
   - ARO에 없는 유전자는 같은 유전자 패밀리의 상위 ARO 항목으로 대신 매핑했습니다.   

2. **약물 · 약물군 분류**  
   - ARO의 계층 관계를 따라 **confers_resistance_to_antibiotic / drug_class** 링크를 추적해, 각 유전자가 저항성을 부여하는 개별 항생제와 그 항생제의 **직계 상위 노드(antibiotic molecule의 자식)** 를 ‘약물군’으로 정의했습니다.  
   - 혼합제(예: 트리메토프림–설파메톡사졸)는 **has_part** 관계를 활용해 구성 성분별로 여러 군에 연결합니다.   

3. **argNorm 구현**  
   - **Python**(pandas·BioPython·pronto) 기반으로 작성되었으며, ABRicate·AMRFinderPlus·ARGs‑OAP·DeepARG·GROOT·ResFinder 6개 툴 출력(또는 hAMRonization으로 전처리된 파일)을 입력으로 받습니다.  
   - 사전 구축한 매핑 테이블을 활용해 유전자 이름을 ARO ID로 변환한 뒤, 동일 규칙으로 약물·약물군 정보를 덧붙여 CSV/TSV로 출력합니다.  
   - 자동 매핑과 수동 교정을 분리 저장하여 업데이트 시 기존 수동 데이터가 덮어쓰이지 않도록 설계했습니다.   

---

 
1. **Building ARG‑to‑ARO mappings**  
   - Sequences from eight ARG databases were processed with **RGI v6.0.3** (CARD v4.0.0). Each gene was linked to an ARO accession, with hits labelled Perfect, Strict or Loose.  
   - Fewer than 1 % of genes required **manual curation** owing to drug‑class mismatches or absent hits; genes not present in ARO were mapped to the appropriate **family‑level** ARO term. 

2. **Drug and drug‑class assignment**  
   - argNorm traverses ARO relations—**confers_resistance_to_antibiotic / drug_class**—from each gene (and its ancestors) to list specific antibiotics.  
   - It then reports the **immediate children of the “antibiotic molecule” node** as drug classes; mixtures are resolved via the **has_part** relation so that all constituent classes are retained.   

3. **Software implementation**  
   - Written in **Python** using pandas, BioPython and pronto, argNorm ingests raw outputs from ABRicate, AMRFinderPlus, ARGs‑OAP, DeepARG, GROOT and ResFinder (or hAMRonization‑formatted files).  
   - The tool replaces gene names with ARO IDs via the pre‑built mapping table and appends harmonised drug/drug‑class metadata, exporting the result as CSV/TSV.  
   - Automated RGI mappings are stored separately from manual edits, enabling straightforward updates without overwriting curated records. 

   
 
<br/>
# Results  



 
argNorm은 **“유전자 → ARO ID → 항생제 → 항생제 군(Drug Class)”**라는 네 단계로 레이블을 단일화합니다.  

| 단계 | 통일 방법 | 최종 레이블 수¹ | 예시 |
|---|---|---|---|
|① 유전자|각 도구가 반환한 유전자명을 **사전 구축한 ARO 매핑표**로 변환|≈ 6,000 개 ARO ID|`ANT(2”)-Ia → ARO:3000230`|
|② 항생제|ARO의 `confers_resistance_to_antibiotic` 관계를 따라 **구체적인 약물명**을 부여|약 900 여 항생제 용어|`tobramycin`, `kanamycin A` …|
|③ 항생제 군|각 항생제를 ARO 계층에서 **‘antibiotic molecule’의 직계 자식**으로 승격|고정 **62 개 drug class**|`aminoglycoside antibiotic`, `β‑lactam antibiotic` …|
|④ 도구·DB 식별|원본 결과에 남아 있는 툴/DB 메타데이터|6 툴·8 DB|`AMRFinderPlus (NCBI RGD)` 등|

¹ 8 개 데이터베이스 전체를 합친 값(중복 제거 기준); 논문 Table 1에 나타난 각 DB별 고유 ARO ID(최대 5,921 개)를 통합하면 약 6 천 개의 유전자‑레이블이 지원됩니다. drug class는 ARO가 제공하는 62 개로 고정되어, 예컨대 CARD/ARO보다 적은 19 개(ARG‑ANNOT)나 32 개(DeepARG)만 제공하던 기존 DB보다 훨씬 세분화된 레이블을 일관되게 얻을 수 있습니다.   

**핵심 결과**  
* 6 개 주석 도구 출력 59,550 개 유전자 중 > 99 %를 자동 매핑(Perfect 50.8 %, Strict 28 %, Loose 21.2 %); 나머지 < 1 %는 수동으로 보정.  
* 약물이 혼합제인 경우(**has_part** 관계) 모든 구성 항생제와 해당 항생제군을 병기해 복합 저항성을 정확히 표기.  
* 정규화된 CSV/TSV 산출물에는 ▶ `ARO_ID` ▶ `standard_gene_symbol` ▶ `antibiotic(s)` ▶ `drug_class(62종)` ▶ 원본 스코어(coverage, identity 등)가 포함되어, 도구 간 결과를 그대로 비교하거나 요약 통계(예: 특정 drug class 빈도) 계산이 가능.   

---

  
argNorm unifies labels through a four‑level pipeline: **Gene → ARO ID → Antibiotic → Drug Class**.

| Stage | Normalisation rule | Final label count¹ | Example |
|---|---|---|---|
|① Gene | Map reported gene names to ARO accessions via a pre‑built table | ≈ 6,000 unique ARO IDs | `ANT(2“)-Ia → ARO:3000230` |
|② Antibiotic | Follow `confers_resistance_to_antibiotic` edges to list concrete drugs | ~ 900 antibiotic terms | `tobramycin`, `kanamycin A` |
|③ Drug Class | Ascend to the immediate children of **“antibiotic molecule”** | Fixed **62 drug classes** | `aminoglycoside antibiotic`, `β‑lactam antibiotic` |
|④ Tool/DB tag | Retain original tool / database metadata | 6 tools, 8 DBs | `AMRFinderPlus (NCBI RGD)` |

¹ After merging all eight databases (duplicate AROs collapsed); Table 1 shows up to 5,921 unique AROs per single DB, giving ~6 k overall. The 62 ARO drug classes offer finer granularity than the 19–37 classes available in individual databases.   

**Key findings**  
* Of 59,550 genes across the six tools, **>99 % were mapped automatically** (Perfect 50.8 %, Strict 28 %, Loose 21.2 %); <1 % required manual curation.  
* For antibiotic mixtures, argNorm exploits the **has_part** relation to list every constituent drug and class, accurately reflecting composite resistance.  
* The harmonised output (CSV/TSV) reports `ARO_ID`, `standard_gene_symbol`, `antibiotic(s)`, `drug_class` (62 options) and original scoring metrics, enabling direct cross‑tool comparison or aggregated statistics such as class‑level prevalence. 
   


<br/>
# 예제  



 
아래는 argNorm이 **AMRFinderPlus** 출력 파일을 정규화했을 때 생성되는 행(가상 데이터) 세 개를 보여 줍니다.  

| tool | original gene name | ARO_ID | standard symbol | antibiotics | drug class |
|---|---|---|---|---|---|
| AMRFinderPlus | **ANT(2”)‑Ia** | ARO:3000230 | ant(2“)‑Ia | kanamycin A, tobramycin, gentamicin | **aminoglycoside antibiotic** |
| AMRFinderPlus | **blaCTX‑M‑15** | ARO:3000040 | blaCTX‑M‑15 | cefotaxime, ceftazidime | **β‑lactam antibiotic** |
| AMRFinderPlus | **tetM** | ARO:3000072 | tetM | doxycycline, tetracycline | **tetracycline antibiotic** |

**해설**  
* `original_gene_name` 열은 도구가 출력한 원본 표기, `standard_symbol`은 ARO가 권장하는 표준 기호입니다.  
* `antibiotics` 열에는 ARO가 연결한 구체적 약물이 쉼표로 나열되고, `drug_class`는 62개 클래스 중 하나로 통일됩니다.  
* 같은 분석을 DeepARG·ResFinder 등 다른 툴의 결과에 적용해도 **ARO_ID와 drug class가 같으면 같은 유전자로 인식**되므로, 여러 데이터셋을 그대로 합쳐 빈도 분석·시각화를 할 수 있습니다.   

---


Below is a small **mock** snippet illustrating how argNorm normalises an **AMRFinderPlus** output file:

| tool | original gene name | ARO_ID | standard symbol | antibiotics | drug class |
|---|---|---|---|---|---|
| AMRFinderPlus | **ANT(2”)‑Ia** | ARO:3000230 | ant(2“)‑Ia | kanamycin A, tobramycin, gentamicin | **aminoglycoside antibiotic** |
| AMRFinderPlus | **blaCTX‑M‑15** | ARO:3000040 | blaCTX‑M‑15 | cefotaxime, ceftazidime | **β‑lactam antibiotic** |
| AMRFinderPlus | **tetM** | ARO:3000072 | tetM | doxycycline, tetracycline | **tetracycline antibiotic** |

**Notes**  
* `original_gene_name` is the raw label from the annotation tool, whereas `standard_symbol` is the canonical ARO symbol.  
* `antibiotics` lists specific drugs retrieved via the `confers_resistance_to_antibiotic` relation; `drug_class` collapses them to one of ARO’s 62 immediate classes.  
* When you process outputs from DeepARG, ResFinder, etc., identical `ARO_ID` and `drug_class` values let you merge and compare datasets without manual relabelling. 


      







<br/>  
# 요약   


argNorm은 RGI로 구축한 ARG→ARO 매핑과 ARO 그래프 탐색으로 6개 주석 도구(8 DB) 결과를 표준화한다.  
전체 5.9 만 유전자 중 99 % 이상을 ARO ID와 62개 약물군으로 자동 통합해 도구 간 비교·메타분석을 가능케 했다.  
예컨대 AMRFinderPlus의 ‘ANT(2”)‑Ia’는 ARO:3000230으로 매핑되어 kanamycin A·tobramycin 등과 함께 ‘aminoglycoside antibiotic’ 클래스로 분류된다.   

argNorm standardises outputs from six annotation tools (eight databases) by applying an RGI‑derived ARG→ARO mapping table and traversing the ARO graph.  
It harmonised >99 % of 59 k genes to unique ARO IDs and 62 unified drug classes, enabling seamless cross‑tool comparison and meta‑analysis.  
For example, AMRFinderPlus’s “ANT(2“)‑Ia” is mapped to ARO:3000230 and linked to kanamycin A, tobramycin, etc., under the “aminoglycoside antibiotic” class. 




<br/>  
# 기타  



  
| 구분 | 내용 | 핵심 메시지 |
|---|---|---|
| **Figure 1** | **(a) Overview:** 여섯 ARG 주석 도구(또는 hAMRonization 전처리 출력)를 입력받아, argNorm이 “유전자 → ARO ID → 항생제 → 약물군” 정규화를 수행하는 과정을 한눈에 보여 줌.<br>**(b) Workflow:** 불일치 유전자명을 ARO accession으로 매핑한 뒤, `confers_resistance_to_*`·`is_a`·`has_part` 관계를 따라 항생제·약물군을 결정하는 그래프 탐색 단계(파란 화살표) 시각화. | 파이프라인 전모와 핵심 알고리즘 흐름을 직관적으로 제시 |
| **Table 1** | 8개 데이터베이스별 **유전자 수·매핑된 고유 ARO 수·수동 교정 건수·약물군 종류** 요약 (예: DeepARG 12,279 유전자 → 2,413 ARO; ARG‑ANNOT 2,063 ARO 등). | argNorm이 지원하는 자원 범위와 데이터 커버리지를 정량화 |
| **Supplementary / GitHub 자료** | *argNorm_benchmark* 저장소: 매핑 정확도 raw 결과, 각 DB별 loose·strict·perfect 매칭 비율, 실행 스크립트 포함.<br>*argNorm* 패키지 데이터 폴더: RGI 자동 매핑 테이블(`aro_mappings.tsv`), 수동 교정 목록(`manual_curations.tsv`). | 재현성 확보: 사용자가 같은 버전으로 다시 실행·검증 가능 |

> 본문에 포함된 표·그림만으로도 파이프라인 이해와 데이터 커버리지 파악이 가능하며, 상세 매핑 통계·스크립트는 GitHub 부록으로 제공돼 투명성을 높입니다.   

---

  
| Item | Content | Take‑away |
|---|---|---|
| **Figure 1** | **(a) Overview:** Schematic showing six annotation tools (or hAMRonization outputs) entering argNorm, which converts “gene → ARO ID → antibiotic → drug class.”<br>**(b) Workflow:** Graph visualisation of how inconsistent gene names are mapped to ARO accessions and then traced via `confers_resistance_to_*`, `is_a` and `has_part` edges to yield antibiotic and class labels (blue arrows). | Offers an at‑a‑glance view of the pipeline and its core graph‑traversal logic. |
| **Table 1** | Summarises, for each of eight databases, the **number of genes, unique AROs obtained, manually curated entries, and distinct drug classes** (e.g. DeepARG: 12,279 genes → 2,413 AROs; ARG‑ANNOT: 2,063 AROs). | Quantifies the resource coverage argNorm currently supports. |
| **Supplementary / GitHub assets** | *argNorm_benchmark* repo: raw mapping‑accuracy outputs, loose/strict/perfect hit counts, and run scripts.<br>*argNorm* package data: RGI‑derived mapping table (`aro_mappings.tsv`) and manual curation list (`manual_curations.tsv`). | Ensures full reproducibility—users can rerun or audit the mapping process. |

> Together, the main‑text figure and table convey the pipeline and coverage, while the online appendices provide granular statistics and code for transparent validation. 



<br/>
# refer format:     



@article{Perovic2025argNorm,
  author  = {Ugarcina Perovic, Svetlana and Ramji, Vedanth and Chong, Hui and Duan, Yiqian and Maguire, Finlay and Coelho, Luis Pedro},
  title   = {argNorm: normalization of antibiotic resistance gene annotations to the Antibiotic Resistance Ontology (ARO)},
  journal = {Bioinformatics},
  year    = {2025},
  doi     = {10.1093/bioinformatics/btaf173},
  url     = {https://github.com/BigDataBiology/argNorm}
}



Perovic, Svetlana Ugarcina, Vedanth Ramji, Hui Chong, Yiqian Duan, Finlay Maguire, and Luis Pedro Coelho. “ArgNorm: Normalization of Antibiotic Resistance Gene Annotations to the Antibiotic Resistance Ontology (ARO).” Bioinformatics (2025). https://doi.org/10.1093/bioinformatics/btaf173. ​  



