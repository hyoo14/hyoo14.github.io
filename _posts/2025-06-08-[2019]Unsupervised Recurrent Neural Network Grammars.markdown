---
layout: post
title:  "[2019]Unsupervised Recurrent Neural Network Grammars"  
date:   2025-06-08 17:13:40 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

RNN기반 grammar와 문장 생성을 joint하는 생성형 언어모델(비지도학습)과 이의 성능 확인  



짧은 요약(Abstract) :    



Recurrent neural network grammars (RNNG) are generative models of language which jointly model syntax and surface structure by incrementally generating a syntax tree and sentence in a top-down, left-to-right order.
Supervised RNNGs achieve strong language modeling and parsing performance, but require an annotated corpus of parse trees. In this work, we experiment with unsupervised learning of RNNGs. Since directly marginalizing over the space of latent trees is tractable, we instead apply amortized variational inference.
To maximize the evidence lower bound, we develop an inference network parameterized as a neural CRF constituency parser. On language modeling, unsupervised RNNGs perform as well their supervised counterparts on benchmarks in English and Chinese. On constituency grammar induction, they are competitive with recent neural language models that induce tree structures from words through attention mechanisms.




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




parse tree와 문장 생성의 joint probability로 학습을 함


# 2.1  Generative Model  


The text describes an unsupervised recurrent neural network (RNN) grammar, which is used to model binary parse trees over sequences of length T. The RNN consists of two main components: 

1. A generative model that predicts both the sequence and the corresponding binary tree.
2. An inference network that produces a distribution over valid binary trees.

The generative model uses a stack LSTM (Stack-based LSTM) to generate terminals (words) and a tree LSTM to obtain constituent representations upon REDUCE actions. The training process involves sampling a binary tree from the inference network, converting it into a sequence of shift-reduce actions, and optimizing the log joint likelihood.

**Key concepts:**

1. **Shift**: Generates a terminal symbol via an affine transformation and softmax.
2. **Reduce**: Pops the last two elements off the stack, combines them using a tree LSTM, and updates the stack with the new representation.
3. **Stack LSTM**: A type of RNN that processes sequences from top to bottom.
4. **Tree LSTM**: A variant of LSTMs used for processing binary trees.

**Training:**

1. The joint log likelihood decomposes into two terms:
	* Terminal/action log likelihood (logp(xz))
	* Action log likelihood (log p(z x<z))

The text also mentions that the standard approach to maximize the marginal likelihood is intractable due to the dependence of zt on all previous actions.

**Amortized variational inference:**

To address these issues, the authors use amortized variational inference, which allows for efficient optimization without explicit independence assumptions.



# 2.2 Amortized Variational Encoder  

This text is a technical paper on using a neural CRF parser to learn a distribution over binary trees, which can be used as an inference network in a generative model. Here's a high-level summary of the main points:

**Motivation**

The authors want to use an inference network that injects inductive bias into the generative model, similar to posterior regularization.

**CRF Parser**

They propose using a neural CRF parser (Durrett and Klein, 2015) as the inference network. The CRF parser defines a distribution over binary trees via the Gibbs distribution:

q(Bx) = 1 / ZT(x) exp ∑_{i,j} Bijsij

where B is the binary matrix representation of the tree, and sij is computed using an MLP.

**Inside Algorithm**

To calculate the partition function ZT(x), they use an "inside algorithm" (Baker, 1979):

1. Compute scores ij for all i and j.
2. For each span [i,i+length-1], compute the score sij by taking a weighted sum of the scores over all possible positions k.

**Gradient-based Optimization**

The computation is itself differentiable and can be optimized using gradient-based methods.

**Bijective Mapping**

They define a bijective mapping f: BT → ZT between binary tree matrices and sequences of SHIFT/REDUCE actions, which allows them to express the distribution q(z x) in terms of the true posterior p(z x).

The paper discusses various aspects of this approach, including:

* The use of a curved exponential family, which is a subset of the marginal polytope of the full exponential family.
* The limitations and potential pitfalls of using a CRF parser as an inference network.

Overall, the authors propose a novel approach to learning distributions over binary trees using a neural CRF parser, with the goal of injecting inductive bias into generative models.


# 2.3 Optimization  

This section discusses the optimization of a variant of the ELBO (Expectation- Maximization) for learning distributions over binary trees. The authors propose an estimator that combines two parts: the entropy term and the log likelihood term.

**Entropy Term**

The entropy term is calculated exactly in O(T^3), using intermediate values from the inside algorithm. Since each step of this dynamic program is differentiable, the gradient can be obtained using automatic differentiation.

**Log Likelihood Term**

An estimator for the log likelihood term with respect to Eq(z x)[logp(xz)] is obtained via the score function gradient estimator (Glynn, 1987; Williams, 1992). This involves calculating a weighted sum of logarithmic terms:

Eq(z x)[logp(xz)]
= Eq(z x)[logp(xz) logq(z x)]

1
K
K
k=1
logp(xz(k)) logq(z(k) x)

**Algorithm 2: Top-Down Sampling**

The algorithm for top-down sampling from q(z x) is as follows:

1. Initialize a binary matrix representation of the tree B.
2. Create an empty queue Q of constituents.
3. While Q is not empty, pop (i j) and calculate i = j - 1.
4. For k := i to j-1 do:
   * Get distribution over splits wk = [i k] [k+1j].
   * Sample a split point k'.
5. Update B by setting Bik = 1 if left child has width > 1, and similarly for right child.

**Gradient Estimation**

The gradient of the ELBO with respect to Eq(z x)[logp(xz)] involves two parts: the entropy term H[q (z x)] and the log likelihood term. The authors use a control variate derived from an average of other samples' joint likelihoods to reduce variance in the estimator.

**Control Variate**

The control variate is calculated as:

1
K
K
k=1
(logp(xz(k)) r(k))

where r(k) = 1/K ∑[j=k logp(xz(j))]. This works better than alternatives such as estimates of baselines from an auxiliary network (Mnih and Gregor, 2014; Deng et al., 2018).

Overall, the authors propose a novel approach to learning distributions over binary trees using a neural CRF parser, with the goal of injecting inductive bias into generative models.
   
 


<br/>
# Results  




**Grammar Induction**

* The authors compare two models: RNNG (Recurrent Neural Grammar) and URNNG (URNNG), which is an extension of RNNG.
* They evaluate their results using F1 scores, reconstruction perplexity, Kullback-Leibler divergence, prior entropy, post-entropy, and uniform entropy metrics.

**Results**

* The authors find that both models perform well on English but PRPN (a model used as a baseline) performs better on Chinese.
* URNNG outperforms RNNG in terms of reconstruction perplexity and Kullback-Leibler divergence.
* However, the difference between the two models is not significant.

**Comparison with Baselines**

* The authors compare their results to those reported by Drozdov et al. (2019) using a different evaluation setup.
* They find that their results are similar but slightly better than those of Drozdov et al.

**Syntactic Evaluation**

* The authors perform syntactic evaluation based on the setup from Marvin and Linzen (2018).
* Each model is given two minimally different sentences, one grammatical and one ungrammatical.
* The task is to identify the grammatical sentence by assigning it a higher probability.
* **Table 6: Accuracy Results**
	+ Overall, supervised RNNG significantly outperforms other models, indicating opportunities for further work in unsupervised modeling.
	+ URNNG does slightly outperform an RNNLM (Recurrent Neural Language Model).
	+ The distribution of errors made by both models is similar, and thus it's not clear whether the outperformance is simply due to better perplexity or learning different structural biases.

Overall, this report presents a technical evaluation of two NLP models (RNNG and URNNG) for grammar induction. The results show that both models perform well but with some differences in performance metrics. Additionally, the syntactic evaluation section highlights the potential benefits of supervised training on improving model performance, particularly when compared to unsupervised approaches like RNNLMs.



<br/>
# 예제  



hypothetical examples of train and test datasets for grammar induction tasks.

**Assumptions:**

* The study uses a dataset with English sentences.
* The task is to predict whether a sentence is grammatical (e.g., "The cat chased the mouse.") or ungrammatical (e.g., "The dog chase the ball.").
* The models are trained on this data and evaluated using metrics like F1 score, reconstruction perplexity, Kullback-Leibler divergence, prior entropy, post-entropy, and uniform entropy.

**Hypothetical Train or Test Dataset:**

| Sentence | Label (Grammatical/Ungrammatical) |
| --- | --- |
| The cat chased the mouse. | Grammatical |
| The dog chase the ball. | Ungrammatical |
| The sun is shining brightly today. | Grammatical |
| The elephant eats a lot of food. | Grammatical |
| The book on the table is mine. | Grammatical |


**Task:**

* Train a model (RNNG or URNNG) on this dataset to predict whether each sentence in the test set is grammatical or ungrammatical.
* Evaluate the performance of the trained models using metrics like F1 score, reconstruction perplexity, Kullback-Leibler divergence, prior entropy, post-entropy, and uniform entropy.

Please note that these are hypothetical examples and not actual data from the original study. If you're interested in learning more about grammar induction or NLP tasks, I'd be happy to provide additional resources!


<br/>  
# 요약   


RNNG and URNNG models are trained on a dataset of English sentences with labels indicating whether each sentence is grammatical or ungrammatical.   
Both models perform well, but RNNG outperforms other models like RNNLM in terms of reconstruction perplexity and Kullback-Leibler divergence. The distribution of errors between the two models is similar, making it unclear if RNNG's performance is due to better perplexity or learning different structural biases.  
Train or test Dataset seems like this: "The cat chased the mouse." (Grammatical).  






<br/>  
# 기타  


<br/>
# refer format:     


@inproceedings{Kim_2019,
    author = {Yoon Kim and Alexander M. Rush and Lei Yu and Adhiguna Kuncoro and Chris Dyer and Gabor Meli},
    title = {Unsupervised Recurrent Neural Network Grammars},
    booktitle = {Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Main Conference)},
    journaltitle = {NAACL-HLT 2019},
    pages = {1105--1117},
    location = {Minneapolis, Minnesota},
    date = {June 2-6, 2019}
}





Yoon Kim, Alexander M. Rush, Lei Yu, Adhiguna Kuncoro, Chris Dyer, and Gabor Meli. "Unsupervised Recurrent Neural Network Grammars." Proceedings of NAACL-HLT 2019 pp. 1105–1117, Association for Computational Linguistics, Minneapolis, MN, June 2-6, 2019.



