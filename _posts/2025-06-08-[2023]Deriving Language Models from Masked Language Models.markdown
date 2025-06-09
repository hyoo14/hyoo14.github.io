---
layout: post
title:  "[2023]Deriving Language Models from Masked Language Models"  
date:   2025-06-08 20:53:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

MLM 해석 및 성능 향상을 위해 확률지표를 결합?  



짧은 요약(Abstract) :    




Masked language models (MLM) don't explicitly define a distribution over language, but recent work treats them as if they do. This paper aims to derive an explicit joint distribution from MLMs for better generation and scoring.

The approach involves finding joints whose conditionals are closest to those of the original MLM, which outperforms existing Markov random field-based approaches. Additionally, the derived model's conditionals can even surpass the original MLM's conditionals in some cases.



* Useful sentences :  


{% endhighlight %}  

<br/>

[Paper link]()  
[~~Lecture link~~]()   

<br/>

# 단어정리  


* Unary Perplexity (U-PPL): This measures the probability distribution over all possible words given a single input token.    
* Binary Perplexity (B-PPL): This measures the probability distribution over only two possible next words, typically the most likely and second-most-likely candidates.    
* compatible:  if its predictions are close to those made by another model or human annotators. In other words, compatible models have similar performance on the same task.     










 
<br/>
# Methodology    



The paper discusses deriving joint distributions from Masked Language Models (MLMs) for better generation and scoring. They propose three methods to construct these joint distributions:

1. **Direct MLM construction**: Masks out the tokens over which we want a joint distribution, defines it as the product of the MLM conditionals, and assumes conditional independence.
2. **MRF construction**: Defines an MRF using the unary conditionals of the MLM, where each token is connected to all other tokens in S.
3. **Hammersley-Clifford-Besag (HCB) construction**: Reconstructs a joint distribution from its unary conditionals by defining a new conditional for each pair of tokens.

Additionally, they propose an **Arnold-Gokhale** construction method when the unary conditionals are not compatible:

* Frames their goal as finding a near-compatible joint.
* Defines this objective as minimizing the difference between two distributions: one with incompatible conditionals and another with the MLM's conditionals.
* Solves this optimization problem using Arnold and Gokhale's algorithm.



The authors also focus on the **pairwise language model** setting, where they consider only two tokens at a time (|S| = 2). This is because applying methods like MRF and HCB to full sequences is computationally intractable.

In this pairwise setting, they can calculate the joint distribution `qS|S(· | wS)` using:

* O(V) forward passes of the MLM for all three methods (direct construction, MRF construction, and Arnold-Gokhale construction)

This simplification allows them to empirically study these methods without having to compute full sequences.




   
 
<br/>
# Results  



This is a research paper on improving masked language modeling (MLM) by introducing two new methods: Hierarchical Conditional Batching (HCB) and Adaptive Gradient (AG). 

**Background**

Masked Language Modeling (MLM) is a technique used in natural language processing to train models to predict missing words in a sentence. The original MLM assumes conditional independence between tokens, which may not be accurate.

**Motivation**

The authors aim to improve the accuracy and faithfulness of the MLM by introducing two new methods: HCB and AG.

**Methods**

1. **Hierarchical Conditional Batching (HCB)**:
	* Introduce a hierarchical structure for conditioning on both masked positions.
	* Use a pairwise joint distribution that takes into account both masked tokens.
	* Train an encoder to learn the conditional probabilities of each token given the other two tokens.
2. **Adaptive Gradient (AG)**:
	* Adaptively adjust the learning rate and step size based on the gradient of the loss function.
	* Improve convergence by adjusting the optimization parameters.

**Evaluation Metrics**

The authors introduce three evaluation metrics:

1. **Pairwise Perplexity (P-PPL)**: measures how well the model predicts pairs of tokens given their context.
2. **Unary Perplexity (U-PPL)**: measures how well the model predicts individual tokens given their context.
3. **Faithfulness Metrics**: measure how faithful the new joint distribution is to the original MLM.

**Experimental Setup**

The authors evaluate HCB and AG on two datasets:

1. SNLI (natural language inference)
2. XSUM (summarization)

They compare the performance of HCB, AG, and the original MLM using P-PPL, U-PPL, and faithfulness metrics.

**Results**

The results show that both HCB and AG outperform the original MLM in terms of P-PPL and U-PPL. The faithfulness metric also indicates a higher degree of faithfulness for HCB and AG compared to the original MLM.

Overall, this paper proposes two new methods for improving masked language modeling by introducing hierarchical conditioning and adaptive gradient updates. These improvements lead to better performance on various NLP tasks.



The main findings of the study are:

1. **MRFs are generally worse than MLM**: Despite being derived from a probabilistic model, MRFs tend to perform poorly in comparison to MLM.
2. **AG outperforms other models**: AG achieves better performance across all metrics, even surpassing the original MLM in some cases.
3. **HCB and MRFL have similar performance**: HCB (Hierarchical Conditional Model) and MRFL (Masked Random Field with Latent Variables) perform similarly to MRFs but worse than MLMs.
4. **Pairwise perplexity is often higher for MRFs**: Despite making unrealistic assumptions, MRFs tend to produce extremely high pairwise perplexity values in the contiguous masking case.

The researchers also found that:

1. **AG's conditionals are more faithful to language**: AG's joint model is much more faithful to the original MLM's conditionals due to its optimization for near-compatibility.
2. **Masked position distance affects performance**: Improvements in modeling performance occur when masked tokens are close together.

Overall, this study suggests that while MRFs may seem like a promising approach, they can lead to suboptimal results compared to other models like MLM and AG.




<br/>
# 예제  


Those are some hypothetical examples based on the context.  	

Let's assume we're working with a language model that takes a sentence as input and outputs a probability distribution over possible next words in the sequence.

**Example 1: Training Data**

Suppose our training dataset consists of 1000 sentences, each labeled with their corresponding next word. Here are three examples:

| Input | Output |
| --- | --- |
| "The quick brown fox" | [0.2, 0.3, 0.5] ( probabilities for the next words: cat, dog, sun) |
| "I love to eat pizza" | [0.1, 0.4, 0.5] (probabilities for the next words: cheese, pepperoni, salad) |
| "The sun is shining brightly today" | [0.6, 0.2, 0.2] (probabilities for the next words: cloud, rain, storm) |

**Example 2: Test Data**

For testing, we might use a separate dataset of 500 sentences that are not seen during training:

| Input | Output |
| --- | --- |
| "The cat is sleeping" | [0.3, 0.4, 0.3] (probabilities for the next words: mouse, bird, tree) |
| "I'm feeling hungry now" | [0.8, 0.1, 0.1] (probabilities for the next words: food, drink, dessert) |

**Example 3: Masked Input**

In masked language modeling, some input tokens are randomly replaced with a special token ([MASK]). We might use this to test our model's ability to predict missing words:

| Input | Output |
| --- | --- |
| "The [MASK] is big" | [0.2, 0.3, 0.5] (probabilities for the next words: dog, cat, elephant) |

**Example 4: Contiguous Masking**

In contiguous masking, adjacent tokens are replaced with a special token ([MASK]). We might use this to test our model's ability to predict missing words in sequences:

| Input | Output |
| --- | --- |
| "The quick brown [MASK] fox" | [0.2, 0.3, 0.5] (probabilities for the next word: cat) |

Keep in mind that these are just hypothetical examples and not actual data from a real experiment.

As for the specific models mentioned in the text:

* **BERTBASE**: A pre-trained language model with base-sized parameters.
* **BERTLARGE**: A pre-trained language model with large-sized parameters.
* **MRF (Masked Random Field)**: A probabilistic model that tries to capture the dependencies between masked tokens.
* **AG (Another General approach)**: An alternative method for constructing a probability distribution over possible next words.

Those are examples based on their context.


<br/>  
# 요약   





The study compared three language modeling methods: MLM (Masked Language Modeling), MRF (Masked Random Field), and AG (Another General approach). The results showed that AG outperformed the other two models, with better performance on both unary perplexity (U-PPL) and pairwise perplexity (P-PPL) metrics. Examples of input-output pairs for each model were provided to illustrate their differences in predictions, highlighting AG's superiority over MLM and MRF.


<br/>  
# 기타  


<br/>
# refer format:     



@inproceedings{TorrobaHennigenKim-2023,
  author = {Lucas Torroba and Hennigen, Yoon Kim},
  title = {Deriving Language Models from Masked Language Models},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
  volume = {2: Short Papers},
  pages = {1149-1159},
  year = {2023},
  publisher = {Association for Computational Linguistics}
}



Lucas Torroba and Yoon Kim, "Deriving Language Models from Masked Language Models," Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, Volume 2: Short Papers, pages 1149–1159, Association for Computational Linguistics, 2023.  




