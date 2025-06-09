---
layout: post
title:  "[2023]Unsupervised Discontinuous Constituency Parsing with Mildly Context-Sensitive Grammars"  
date:   2025-06-08 19:09:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

unsupervised 가 메인인데..  파싱을 잘 하는 경제적(컴퓨테이션이)인 전략 정도    
너무 엄격한 context-free grammar 보다는 mildly context sensitive grammar approach  



짧은 요약(Abstract) :    




The authors propose an approach for unsupervised parsing of discontinuous structures using mildly context-sensitive grammars. They use the Probabilistic Linear Context-Free Rewriting System (LCFRS) formalism, which allows them to fix the rule structure in advance and focus on parameter learning with maximum likelihood.

To reduce computational complexity:

* They restrict the grammar to binary LCFRS with fan-out 2.
* They discard rules that require O(ℓ6) time to parse, reducing inference to O(ℓ5).

They use tensor decomposition-based rank-space dynamic programming with embedding-based parameterization of rule probabilities to scale up the number of nonterminals.

The approach is tested on German and Dutch languages, showing that it can induce linguistically meaningful trees with continuous and discontinuous structures.




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


# 2.1 Background: Scaling PCFGs with low-rank neural parameterizations

The paper discusses how to scale Probabilistic Context-Free Grammars (PCFGs) with a large number of nonterminals. 

Traditional PCFG inference is cubic in the number of nonterminals, making it difficult to scale up.

Two approaches are presented:

1. **Canonical Polyadic Decomposition (CPD)**: Cohen et al. (2013) decompose the 3D probability tensor into a low-rank factorization using CPD. This reduces the complexity from O(ℓ^3m) to O(ℓ^3r + ℓ^2mr), where r is the rank of the decomposition.
2. **Low-Rank Neural Parameterization**: Yang et al. (2021b) propose a neural network-based approach that learns shared symbol embeddings for U, V, and W. This reduces the complexity to O(ℓ^3r + ℓ^2r^2), which is smaller than CPD when r ≪ m.

The paper adapts this low-rank neural parameterization to the LCFRS case (a variant of PCFGs) to scale up to a large number of nonterminals.


# 2.2 Restricted LCFRS


This section appears to be discussing the Restricted LCFRS (LCFRS-2) model, which is a restricted form of the Linear Constituent Grammar Representation System. Here's a summary:

**What is LCFRS?**

LCFRS is a way to represent grammars using strings and rules that define how these strings can be combined.

**Restricted LCFRS-2 (LCFRS-2)**

LCFRS-2 is a restricted form of LCFRS, where each nonterminal node has at most two adjacent contiguous strings in its yield. This means that the fan-out of an LCFRS-2 rule is limited to 2.

**Why restrict LCFRS?**

The original LCFRS model can be computationally expensive for parsing, which makes it difficult to use for unsupervised learning from natural language data. By restricting the fan-out to 2, we reduce the complexity of parsing and make it more efficient.

**How does LCFRS-2 work?**

In an LCFRS-2 rule:

* A nonterminal node (A) has a yield that is either:
	+ A contiguous string with one adjacent terminal symbol
	+ Two adjacent contiguous strings, where the first span belongs to N1 and the second span belongs to N2

**Parsing complexity**

The parsing complexity of an LCFRS-2 rule depends on its fan-out. For rules with a fan-out of 2 (LCFRS-2), the parsing complexity is O(ℓ^5|G|).

**Illustrative example**

An illustrative example shows how an LCFRS-2 model can represent discontinuous spans, where A has two adjacent contiguous strings in its yield.

The section also mentions that Stanojević and Steedman (2020) report that LCFRS-2 can cover up to 87% of the gold discontinuous constituents in a treebank.




# 2.3 Tensor decomposition-based neural parameterization

This text appears to be a technical description of an algorithm for parsing and generating sentences in the context of natural language processing (NLP). Here's a breakdown of the key components:

**Rank-space dynamic programming**: The authors propose using rank-space dynamic programming, which is a type of optimization technique used to find the most likely parse tree given a set of observations. This approach allows them to efficiently compute sentence likelihoods.

**Tensor decomposition-based LCFRS-2 grammar**: The text describes a specific grammar called LCFRS-2 (Linear Context-Free Rewriting System with two levels), which is a type of context-free grammar that can be used for NLP tasks like parsing and generation. The authors use tensor decomposition to represent the grammar, allowing them to efficiently compute sentence likelihoods.

**Parameterization**: They parameterize the component matrices using neural networks over shared embeddings, which allows the model to learn representations of words in a high-dimensional space.

**Symbol embeddings**: The text describes four sets of symbol embeddings:

1. E1: nonterminal embeddings
2. E2: fan-out-2 nonterminal embedding matrix
3. r: start symbol embedding
4. R1-R4: rank embeddings

These embeddings are used to compute the entries of the U, V, W matrices.

**Decoding**: The authors propose using Minimal Bayes Risk (MBR) decoding, which involves first computing posterior probabilities for each span being a constituent and then using these probabilities as input into CKY-style argmax decoding. This approach allows them to efficiently decode sentences without having to instantiate the full probability tensors.

Overall, this text describes an algorithm that uses rank-space dynamic programming and tensor decomposition-based LCFRS-2 grammar to generate sentences in a high-dimensional space, with parameterization using neural networks over shared embeddings. The authors also propose using MBR decoding as a way to efficiently decode sentences without having to instantiate the full probability tensors.



   
 
<br/>
# Results  



The authors are evaluating various models for unsupervised parsing tasks, which involves predicting the syntactic structure of sentences without any labeled training data.


**Models evaluated:**

1. Neural/Compound LCFRSs (Low-Complexity Factorized Regularized Sentence Models)
2. PCFG (Probabilistic Context-Free Grammars) and its extensions
3. TN-LCFRS (Tensor-based Low-rank Factorization for Sentences)

**Experimental setup:**

* Four test datasets from German, Dutch, and English languages
* Different sizes of grammars (|P| = 45, 450, 4500)
* Various hyperparameters and training settings

**Results:**

1. The authors find that TN-LCFRSs outperform other models in terms of F1 score for both continuous and discontinuous spans.
2. As the grammar size increases, TN-LCFRSs show significant improvements in performance.
3. Supervised training with TN-LCFRS4500 achieves better results than unsupervised parsing.

**Key findings:**

* Parameter sharing through low-rank factorizations leads to improved performance.
* Discontinuous F1 scores are non-trivial for unsupervised parsers, but there is still a large gap between supervised and oracle-based models.





<br/>
# 예제  



Here is an example of a test or train dataset for the unsupervised parsing task:

**Dataset:**

* Language: German
* Treebank: NEGRA (a publicly available treebank)
* Sentence: "Der Hund läuft schnell um den Baum herum."
("The dog runs quickly around the tree.")

**Gold Standard Parsing:**

* Constituent labels:
	+ S (Sentence)
	+ NP (Noun Phrase)
	+ N (Proper Noun)
	+ Det (Determiner)
	+ Art (Article)
	+ Adj (Adjective)
	+ Adv (Adverb)
	+ P (Preposition)
	+ VP (Verb Phrase)
* Constituent structure:
```
S
  NP [Det, Art] Hund [N]
    N Läuft [VP]
      Det schnell
        Adv
    um den Baum herum [P]
      Det den
        Det
          N Baum
            P herum
              VP
                Adj schnell
                  Adv
```
**Task:**

* Predict the constituent labels and structure of the sentence using an unsupervised parsing model.

Note that this is a simple example.  





<br/>  
# 요약   




**Method:** The authors evaluate unsupervised parsing models using the German NEGRA and Dutch LASSY treebanks.

**Result:** TN-LCFRSs outperform other models in terms of F1 score for both continuous and discontinuous spans, with significant improvements as grammar size increases.

**Example:** For a test sentence "Der Hund läuft schnell um den Baum herum.", the gold standard parsing yields [S NP Det Art Hund N VP Det Adv Läuft P Den P BM Herum], while an unsupervised model predicts [S NP Det Art Hund N VP Det Adv Läuft P Um P Den P BM].


<br/>  
# 기타  


<br/>
# refer format:     



@inproceedings{yang2023,
  author = {Songlin Yang and Roger P. Levy and Yoon Kim},
  title = {Unsupervised Discontinuous Constituency Parsing with Mildly Context-Sensitive Grammars},
  booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: Volume 1: Long Papers},
  pages = {5747--5766},
  year = {2023},
  publisher = {{Association for Computational Linguistics}},
  address = {July 9-14, 2023}
}   





Songlin Yang, Roger P. Levy, and Yoon Kim, "Unsupervised Discontinuous Constituency Parsing with Mildly Context-Sensitive Grammars," Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: Volume 1: Long Papers (2023), pp. 5747-5766.
   




