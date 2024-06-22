---
layout: post
title:  "[2024]RAFT Adapting Language Model to Domain Specific RAG"  
date:   2024-06-21 06:47:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    

대규모 언어 모델(LLMs)을 대규모 텍스트 데이터로 사전 학습시키는 것은 이제 표준 패러다임이 되었습니다. 이러한 LLMs를 많은 후속 응용 프로그램에 사용할 때, RAG 기반 프롬프트 또는 미세 조정을 통해 사전 학습된 모델에 새로운 정보를 추가하는 것이 일반적입니다. 그러나 정보를 통합하는 가장 좋은 방법은 여전히 ​​열린 질문으로 남아 있습니다. 본 논문에서는 "오픈 북" 도메인 내 설정에서 질문에 답변하는 모델의 능력을 향상시키는 학습 방법인 Retrieval Augmented Fine Tuning(RAFT)을 제시합니다. RAFT를 학습할 때 질문과 검색된 문서 세트를 제공하면 질문에 답변하는 데 도움이 되지 않는 문서(방해 문서)를 무시하도록 모델을 학습시킵니다. RAFT는 관련 문서에서 올바른 시퀀스를 그대로 인용하여 질문에 답하는 데 도움을 줌으로써 이를 달성합니다. RAFT의 chain-of-thought 스타일 응답과 결합되어 모델의 추론 능력을 향상시키는 데 도움이 됩니다. 도메인별 RAG에서 RAFT는 PubMed, HotpotQA 및 Gorilla 데이터 세트 전반에서 모델 성능을 지속적으로 개선하여 사전 학습된 LLM을 도메인별 RAG에 맞게 향상시키는 후속 학습 방법을 제공합니다.    


Pretraining Large Language Models (LLMs) on large corpora of textual data is now a standard paradigm. When using these LLMs for many downstream applications, it is common to additionally incorporate new information into the pretrained model either through RAG-based-prompting, or finetuning. However, the best methodology to incorporate information remains an open question. In this paper, we present Retrieval Augmented Fine Tuning (RAFT), a training recipe which improves the model’s ability to answer questions in "open-book" in-domain settings. In training RAFT, given a question, and a set of retrieved documents, we train the model to ignore those documents that don’t help in answering the question, which we call, distractor documents. RAFT accomplishes this by citing verbatim the right sequence from the relevant document to help answer the question. This coupled with RAFT’s chain-of-thought-style response helps improve the model’s ability to reason. In domain specific RAG, RAFT consistently improves the model’s performance across PubMed, HotpotQA, and Gorilla datasets, presenting a post-training recipe to improve pre-trained LLMs to in-domain RAG.   


* Useful sentences :  
** RAFT를 학습할 때 질문과 검색된 문서 세트를 제공하면 질문에 답변하는 데 도움이 되지 않는 문서(방해 문서)를 무시하도록 모델을 학습시킵니다. RAFT는 관련 문서에서 올바른 시퀀스를 그대로 인용하여 질문에 답하는 데 도움을 줌으로써 이를 달성합니다.  
** In training RAFT, given a question, and a set of retrieved documents, we train the model to ignore those documents that don’t help in answering the question, which we call, distractor documents. RAFT accomplishes this by citing verbatim the right sequence from the relevant document to help answer the question. This coupled with RAFT’s chain-of-thought-style response helps improve the model’s ability to reason.  


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1ClJHBY0smXDvMaO6dG9_Ozw-GlhiBJyV?usp=sharing)  
[~~Lecture link~~]()   

<br/>

# 단어정리  
*  
 
<br/>
# Methodology    


이 논문에서는 Retrieval-Augmented Generation (RAG)과 함께 지시 미세 조정 (IFT)을 결합하는 방법을 연구합니다. 우리는 Retrieval-Augmented Fine Tuning (RAFT)이라는 새로운 적응 전략을 제안합니다. RAFT는 도메인 지식을 통합하는 동시에 도메인 내 RAG 성능을 향상시키는 문제를 해결합니다. RAFT의 목표는 모델이 미세 조정을 통해 도메인별 지식을 학습할 수 있도록 하는 것뿐만 아니라 검색된 방해 정보에 대한 내성을 보장하는 것입니다. 이를 위해 RAFT는 질문(프롬프트), 검색된 도메인별 문서, 올바른 답변 간의 역학을 이해하도록 모델을 훈련합니다. 오픈북 시험에 대한 비유를 다시 들자면, 우리의 접근 방식은 관련성과 무관한 검색된 문서를 인식하여 오픈북 시험에 대비하는 것과 유사합니다.

RAFT에서는 문서(D*)에서 질문(Q)에 답하여 답변(A*)을 생성하도록 모델을 훈련합니다. 여기서 A*는 체인 오브 사고(chain-of-thought) 추론을 포함하며 방해 문서(Dk)가 존재합니다. 우리는 섹션 3에서 방법론을 설명하고 섹션 5에서 훈련 및 테스트 시간에 방해 문서의 수(k)에 대한 민감도를 분석합니다. RAFT는 PubMed, HotPotQA 및 Gorilla 데이터 세트에서 RAG가 있거나 없는 상태에서 기존의 감독 하 미세 조정을 일관되게 능가하여 사전 훈련된 LLM을 도메인별 RAG에 맞게 향상시키는 새로운 학습 방법을 제공합니다. 우리의 코드는 https://github.com/ShishirPatil/gorilla 에서 제공됩니다.  



In this paper, we study how to combine instruction fine-tuning (IFT) with retrieval-augmented generation (RAG). We propose a novel adaptation strategy – Retrieval-Augmented Fine Tuning (RAFT). RAFT specifically addresses the challenge of fine-tuning LLMs to both incorporate domain knowledge while also improving in-domain RAG performance. RAFT aims to not only enable models to learn domain-specific knowledge through fine-tuning, but also to ensure robustness against distracting retrieved information. This is achieved by training the models to understand the dynamics between the question (prompt), the domain-specific documents retrieved, and the right answer. Going back to our analogy to the open book exam, our approach is analogous to studying for an open-book exam by recognizing relevant, and irrelevant retrieved documents.

In RAFT, we train the model to answer the question (Q) from Document(s) (D*) to generate answer (A*), where A* includes chain-of-thought reasoning Wei et al. (2022); Anthropic (2023), and in the presence of distractor documents (Dk). We explain the methodology in Section 3 and analyze the sensitivity to the number of distractor documents (k) at train- and test- time in Section 5. RAFT consistently outperforms Supervised-finetuning both with- and without- RAG across PubMed Dernoncourt & Lee (2017), HotPot QA Yang et al. (2018), and HuggingFace Hub, Torch Hub, and Tensorflow Hub Gorilla datasets Patil et al. (2023), presenting a novel, yet simple technique to improve pre-trained LLMs for in-domain RAG. Our code is available at https://github.com/ShishirPatil/gorilla【4†source】.    



<br/>
# Results  


본 논문에서는 RAFT 모델의 성능을 평가하기 위해 다양한 데이터셋과 기준선을 사용했습니다. Tab. 1에서 우리는 RAFT 모델이 기준선을 일관되게 그리고 크게 능가한다는 것을 확인했습니다. 기본 Llama-2 지시 조정 모델과 비교할 때, RAFT는 RAG와 함께 정보 추출 및 방해 요소에 대한 내성 측면에서 훨씬 더 뛰어납니다. 이러한 향상은 Hotpot QA에서 35.25%, Torch Hub 평가에서 76.35%에 달할 수 있습니다. 특정 데이터셋에서 DSF와 비교할 때, 우리의 모델은 제공된 컨텍스트를 활용하여 문제를 해결하는 데 더 뛰어납니다. RAFT는 Hotpot과 HuggingFace 데이터셋에서 더 나은 성능을 보여주며(Hotpot에서 30.87%, HuggingFace에서 31.41%), PubMed QA에서는 이진 예/아니오 질문이므로 DSF + RAG와 비교했을 때 큰 향상을 관찰하지 못했습니다. 심지어 더 크고 더 나은 모델인 GPT-3.5와 비교했을 때도 RAFT는 상당한 이점을 보여줍니다.

전체적으로, LLaMA-7B 모델은 RAG가 있거나 없는 상태에서 그라운드 트루스와 맞지 않는 답변 스타일로 인해 성능이 좋지 않습니다. 도메인별 튜닝을 적용함으로써 우리는 그 성능을 크게 향상시킬 수 있었습니다. 이 과정은 모델이 적절한 답변 스타일을 학습하고 채택할 수 있도록 합니다. 그러나 도메인별로 미세 조정된(DSF) 모델에 RAG를 도입한다고 해서 항상 더 나은 결과를 얻을 수 있는 것은 아닙니다. 이는 모델이 컨텍스트 처리 및 유용한 정보 추출에 대한 훈련이 부족함을 나타낼 수 있습니다. 우리의 방법인 RAFT를 도입함으로써, 우리는 모델이 요구되는 답변 스타일과 일치할 뿐만 아니라 문서 처리 능력을 향상시키도록 훈련합니다. 결과적으로, 우리의 접근 방식은 모든 다른 접근 방식보다 뛰어납니다.



Using the above datasets and baselines, we evaluate our model RAFT and demonstrate the effectiveness of RAFT in Tab. 1. We see that RAFT consistently and significantly outperforms the baselines. Compared with the base Llama-2 instruction-tuned model, RAFT with RAG does much better in terms of extracting information as well as being robust towards distractors. The gain can be as big as 35.25% on Hotpot QA and 76.35% on Torch Hub evaluation. Compared with DSF on the specific dataset, our model does better at relying on the provided context to solve the problem. RAFT does much better on the tasks like Hotpot and HuggingFace datasets (30.87% on Hotpot and 31.41% on HuggingFace). Note that for PubMed QA, since it is a binary yes/no question, we don’t observe significant gains when we compare our model with DSF + RAG. Even compared with a much larger and better model GPT-3.5, RAFT demonstrates significant advantages.

Overall, the LLaMA-7B model, both with and without the RAG, performs poorly due to its answering style not aligning with the ground truth. By applying domain-specific tuning, we significantly enhance its performance. This process enables the model to learn and adopt the appropriate style of answering. However, introducing RAG to a domain-specifically fine-tuned (DSF) model doesn’t invariably lead to better outcomes. This might indicate that the model lacks training in context processing and extracting useful information from it. By incorporating our method, RAFT, we train the model not only to match its answering style with that required but also to improve its document processing capabilities. Consequently, our approach outperforms all others.



<br/>
# 예시  


이 섹션에서는 RAFT와 도메인별 미세 조정(DSF) 접근법 간의 잠재적 이점을 예시를 통해 설명합니다. 그림 4에서 우리는 비교 예시를 제시합니다. 이 예시는 질문에 대한 문맥을 제공할 때 DSF 모델이 혼란스러워하는 시나리오를 보여줍니다. 예를 들어, "Evolution"이라는 영화의 각본가를 묻는 질문에 대해, DSF 모델은 올바른 이름 대신 영화 제목 중 하나를 잘못 인용합니다. 반면에 RAFT 모델은 정확하게 질문에 답합니다. 이 차이는 질문-답변 쌍만으로 모델을 훈련하는 것이 제공된 문서에서 관련 컨텍스트를 추론하는 능력을 저하시킬 수 있음을 시사합니다. 이 비교는 표준 지시 조정 및 문맥 이해를 훈련 데이터 세트에 포함시키는 중요성을 강조합니다.

**HotPot QA 예시**  
질문: "Evolution"이라는 영화의 각본가는 누구인가? 이 영화에는 니콜라스 케이지와 티아 레오니가 출연한다.  
문서: "David Weissman은 각본가이자 감독입니다. 그의 영화 크레딧에는 'The Family Man'(2000), 'Evolution'(2001), 'When in Rome'(2010)이 포함됩니다."

**RAFT 응답:**  
##Reason: "Evolution"이라는 영화의 각본가는 David Weissman입니다. 참조 문서에는 David Weissman이 'The Family Man'(2000), 'Evolution'(2001), 'When in Rome'(2010)의 각본가로 언급되어 있습니다. 따라서 "Evolution"의 각본가는 David Weissman입니다.  
##Answer: David Weissman ✓

**DSF 응답:**  
The Family Man ✗  



In this section, we present a comparative example to illustrate the potential advantages of RAFT over the domain-specifically fine-tuned (DSF) approach. In Figure 4, we present an example where the DSF model becomes confused by a question asking for the identity of a screenwriter. Instead of providing the correct name, it mistakenly cites one of the films written by the screenwriter. In contrast, the RAFT model accurately answers the question. This discrepancy suggests that training a model solely with question-answer pairs may impair its ability to derive relevant context from provided documents. The comparison underscores the importance of incorporating both standard instructional tuning and context comprehension into the training dataset.

**HotPot QA Example**  
Question: "Who is the screenwriter with credits for 'Evolution,' a film starring Nicolas Cage and Téa Leoni?"  
Documents: "David Weissman is a screenwriter and director. His film credits include 'The Family Man' (2000), 'Evolution' (2001), and 'When in Rome' (2010)."

**RAFT Response:**  
##Reason: The screenwriter with credits for the film "Evolution," starring Nicolas Cage and Téa Leoni, is David Weissman. This information is provided in the reference documents where it mentions David Weissman as a screenwriter with film credits including "The Family Man" (2000), "Evolution" (2001), and "When in Rome" (2010). Therefore, the screenwriter for "Evolution" is David Weissman.  
##Answer: David Weissman ✓

**DSF Response:**  
The Family Man ✗  


<br/>  
# 요약 


RAFT는 대규모 언어 모델(LLMs)의 도메인 특화 RAG 성능을 향상시키기 위해 개발된 학습 전략입니다. RAFT는 질문에 답하는 데 도움이 되지 않는 방해 문서를 무시하도록 모델을 훈련시킵니다. 체인 오브 사고(chain-of-thought) 스타일의 응답을 통해 모델의 추론 능력을 향상시킵니다. 실험 결과, RAFT는 PubMed, HotpotQA 및 Gorilla 데이터 세트에서 기존 방법보다 일관되게 더 나은 성능을 보였습니다. RAFT는 특히 도메인 특화 문서에서 효과적입니다.


RAFT is a training strategy developed to improve the domain-specific RAG performance of large language models (LLMs). RAFT trains models to ignore distractor documents that do not aid in answering questions. It enhances the model's reasoning ability through chain-of-thought style responses. Experimental results show that RAFT consistently outperforms existing methods across PubMed, HotpotQA, and Gorilla datasets. RAFT is particularly effective in domain-specific document settings.

# 기타  

RAFT의 핵심 요지는 도메인 특화 RAG 성능을 향상시키기 위해 대규모 언어 모델(LLMs)을 도메인에 맞게 인스트럭션 파인튜닝했다는 것입니다. 이를 통해 모델이 방해 문서를 무시하고 체인 오브 사고 스타일의 응답을 통해 질문에 대한 정확한 답변을 제공할 수 있도록 했습니다.   


The key point of RAFT is that it fine-tunes large language models (LLMs) with domain-specific instruction to improve domain-specific RAG performance. This allows the model to ignore distractor documents and provide accurate answers to questions through chain-of-thought style responses.   


<br/>
# refer format:     
Tianjun Zhang, Shishir G. Patil, Naman Jain, Sheng Shen, Matei Zaharia, Ion Stoica, and Joseph E. Gonzalez. "RAFT: Adapting Language Model to Domain Specific RAG." arXiv preprint arXiv:2403.10131 (2024). doi:10.48550/arXiv.2403.10131.
  
  
@article{zhang2024raft,
  title={RAFT: Adapting Language Model to Domain Specific RAG},
  author={Tianjun Zhang and Shishir G. Patil and Naman Jain and Sheng Shen and Matei Zaharia and Ion Stoica and Joseph E. Gonzalez},
  journal={arXiv preprint arXiv:2403.10131},
  year={2024},
  doi={10.48550/arXiv.2403.10131}
}
