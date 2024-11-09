---
layout: post
title:  "[2024]A Topology-aware Graph Coarsening Framework for Continual Graph Learning"  
date:   2024-11-09 11:00:40 -0500
categories: study
---

{% highlight ruby %}


한줄 요약: 



짧은 요약(Abstract) :    


본 연구는 점진적 학습이 필요한 그래프 신경망(GNN)을 위해, 기존 데이터의 손실을 최소화하면서 새롭게 유입되는 그래프 데이터에 맞춰 적응하는 문제를 해결하고자 합니다. 이를 위해 저자들은 TACO라는 토폴로지 인식 그래프 축소 및 지속적 학습 프레임워크를 제안합니다. TACO는 과거 데이터를 축소된 그래프로 저장하고, 새로운 그래프와의 결합 및 축소를 통해 기억을 유지하면서 모델을 업데이트합니다. 이 프레임워크는 노드의 근접성을 기반으로 한 그래프 축소 알고리즘을 통해 효율적으로 토폴로지 정보를 보존하며, 원본 그래프 학습과 유사한 성능을 유지하도록 설계되었습니다. 실험 결과, TACO는 다양한 실제 데이터 세트에서 기존 방법들에 비해 높은 성능을 보였습니다.


This study addresses the challenge of training a Graph Neural Network (GNN) that needs to continually learn while minimizing the loss of knowledge from previous data as new graph data arrives. To tackle this issue, the authors propose a topology-aware graph coarsening and continual learning framework named TACO. TACO stores past data as a reduced graph, which it combines with new graphs over time while maintaining knowledge stability through a coarsening algorithm based on node proximity. This approach effectively preserves topological information and achieves performance close to that of learning on the original graph. Experimental results demonstrate TACO’s superior performance over existing methods on various real-world datasets.


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





TACO(TOrage-aware Coarsening for continual learning on graphs) 프레임워크는 지속적 학습(Continual Learning) 환경에서 그래프 데이터가 점진적으로 추가될 때, 기존의 정보를 효율적으로 보존하면서 새로운 데이터와의 결합을 통해 그래프 신경망(GNN) 모델을 학습시키는 방법을 제공합니다. 이 방법론의 주요 단계는 결합(Combine), 축소(Reduce), 생성(Generate)으로 나누어지며, 각각이 지속적 학습을 위한 핵심 역할을 합니다.

	1.	노드 임베딩 기반 유사도 계산
	•	TACO에서 노드는 일반적으로 특정 개체나 객체를 나타내며, 소셜 네트워크의 사용자, 논문 인용 네트워크의 논문, 제품 추천 네트워크의 제품 등 다양한 데이터를 표현할 수 있습니다.
	•	각 노드에는 해당 개체의 특성이나 속성이 벡터 형식의 임베딩으로 표현되며, 그래프 신경망(GNN)을 통해 학습된 임베딩 벡터를 사용하여 노드 간의 유사도를 계산합니다.
	•	이 유사도는 노드 특성(feature), 인접 노드(neighbor)의 특성, 그리고 노드 간의 거리와 같은 요소를 기반으로 측정되며, TACO는 이러한 노드 간의 코사인 유사도를 사용해 두 노드가 얼마나 비슷한지를 정량화합니다.
	2.	결합 단계 (Combine)
	•	목표: 현재의 그래프(Gt)와 이전에 축소된 그래프(Grt-1)를 결합하여 새로운 결합 그래프(Gct)를 생성합니다.
	•	과정:
	1.	Gt는 새롭게 추가된 노드와 과거 작업에서 나타났던 노드들을 포함합니다.
	2.	Gt와 Grt-1 간의 공통 노드를 기준으로 그래프를 결합하며, 노드 간 상호 관계를 표현하여 과거와 새로운 작업의 관계를 반영합니다.
	3.	결합된 그래프는 새로운 노드와 중요한 이전 노드를 포함하여 그래프의 구조적 변화를 저장합니다.
	•	결과: 이 결합된 그래프를 통해 모델이 새로운 데이터와 기존의 중요한 정보를 포함한 노드 분류 작업을 수행할 수 있습니다.
	3.	축소 단계 (Reduce)
	•	목표: 결합된 그래프(Gct)의 크기를 줄이면서 중요한 구조적 정보와 노드의 특성을 보존하는 것입니다.
	•	과정:
	1.	RePro라는 그래프 축소 알고리즘을 사용하여 노드를 군집화하여 슈퍼노드로 통합합니다. 이 과정에서 노드의 특성, 이웃의 유사성, 기하학적 근접성을 고려하여 유사한 노드들을 병합합니다.
	2.	코사인 유사도 값에 따라 유사도가 높은 노드들이 먼저 병합되며, 목표 크기에 도달할 때까지 반복됩니다.
	3.	이를 통해 생성된 슈퍼노드는 원본 그래프의 주요 구조적 정보를 보존하여 그래프가 축소되더라도 성능을 유지할 수 있도록 합니다.
	•	특징: TACO는 스펙트럼 기반 방법과 달리 노드 속성과 그래프 구조적 유사성을 모두 반영하여, 더 효율적이며 시간 복잡도가 낮은 그래프 축소를 구현합니다.
	4.	생성 단계 (Generate)
	•	목표: 축소된 그래프(Grt)에서 새롭게 생성된 슈퍼노드의 특성과 인접 관계를 재구성하여, 축소된 그래프에서 학습이 원활히 이루어지도록 합니다.
	•	과정:
	1.	각 슈퍼노드의 중요도를 평가하기 위해 노드의 연결된 인접 노드 수, 인접 노드의 중요도 합산 등의 척도를 사용하여, 중요한 노드의 속성을 더 많이 보존하도록 합니다.
	2.	결합 과정에서 형성된 군집 정보(Qt)를 통해 노드의 특성 및 연결 관계를 계산하여, 슈퍼노드가 본래 그래프의 중요한 정보를 유지하도록 합니다.
	5.	노드 충실도 보존 전략 (Node Fidelity Preservation)
	•	문제점: 그래프가 반복적으로 축소되면서, 소수 클래스나 덜 중요한 노드가 점차 소실될 위험이 있습니다.
	•	해결책: TACO는 특정 노드를 병합하지 않도록 패널티를 부여하는 Node Fidelity Preservation 전략을 통해 중요한 노드를 선택하여 소수 클래스의 정보를 보존합니다.

전체적인 학습 과정

TACO는 위의 결합, 축소, 생성을 반복하여, 새로운 작업이 추가될 때마다 과거의 정보를 재학습하지 않고도 보존합니다. 따라서 메모리 사용량을 줄이면서도 모델 성능을 유지할 수 있으며, 다양한 실제 데이터 세트에서 TACO의 탁월한 성능이 입증되었습니다.



The TACO (Topology-aware Coarsening for continual learning on graphs) framework provides a method to train a Graph Neural Network (GNN) in a continual learning setting where new graph data arrives incrementally. TACO enables the model to combine this incoming data with previously stored information, thus minimizing information loss while updating the model. The methodology is divided into three main steps: Combine, Reduce, and Generate, each serving as a core function in continual learning.

	1.	Node Embedding-based Similarity Calculation
	•	In TACO, nodes generally represent entities or objects, such as users in a social network, papers in a citation network, or products in a recommendation network.
	•	Each node’s properties are represented as embedding vectors, which are learned through a GNN. The embedding vectors of each node capture both the node’s features and the information of its neighbors.
	•	Node similarity is calculated based on the cosine similarity of their embedding vectors, considering aspects like node features, neighbor similarity, and geometric closeness.
	2.	Combine Step
	•	Objective: To combine the current graph (Gt) with the previously reduced graph (Grt-1) to create a new combined graph (Gct).
	•	Process:
	1.	The current graph (Gt) includes both newly added nodes and nodes from previous tasks.
	2.	Nodes shared between Gt and Grt-1 are aligned, thus merging the graphs and reflecting inter-task relationships.
	3.	The combined graph now stores the structural evolution by including both new and important past nodes.
	•	Outcome: This combined graph allows the model to perform node classification tasks that incorporate both new and essential past information.
	3.	Reduce Step
	•	Objective: To reduce the size of the combined graph (Gct) while preserving essential structural information and node attributes.
	•	Process:
	1.	The graph coarsening algorithm, RePro, groups nodes into clusters to form super-nodes based on feature similarity, neighbor similarity, and geometric closeness.
	2.	Nodes with high cosine similarity are merged iteratively until the target size is reached, ensuring that the reduced graph retains the main structure of the original.
	3.	This step results in super-nodes that retain the original graph’s structural information, allowing TACO to maintain model performance even on a smaller graph.
	•	Characteristics: TACO’s approach incorporates both node attributes and structural similarity, achieving efficient graph reduction with lower time complexity compared to spectral-based methods.
	4.	Generate Step
	•	Objective: To reconstruct the attributes and adjacency matrix of the newly generated super-nodes within the reduced graph (Grt).
	•	Process:
	1.	Each super-node’s importance is assessed using measures like degree and neighbor degree sum, allowing TACO to preserve critical information about important nodes.
	2.	The clustering information (Qt) derived from the Combine step is used to calculate node features and adjacency, ensuring that super-nodes retain significant information from the original graph.
	5.	Node Fidelity Preservation Strategy
	•	Problem: Repeated reduction can lead to the loss of minority class nodes or less significant information.
	•	Solution: TACO includes a Node Fidelity Preservation strategy, which assigns penalties to similarity scores, preventing certain nodes from merging and thus preserving minority class information.

Overall Learning Process

TACO iteratively performs the combine, reduce, and generate steps, maintaining past knowledge without the need to retrain on older data. This allows the framework to reduce memory usage while preserving high model performance, as demonstrated by TACO’s superior results on various real-world datasets.
 
   
  
 
<br/>
# Results  




TACO 프레임워크의 성능은 노드 분류 작업에서 Kindle, DBLP, ACM 세 개의 시계열 그래프 데이터셋을 통해 평가되었습니다. TACO는 여러 최신 지속적 그래프 학습(CGL) 모델과 비교되었으며, 주요 평가 지표로 F1-AP(평균 성능)와 F1-AF(평균 망각률)가 사용되었습니다.

	1.	데이터셋 및 평가 지표:
	•	데이터셋: Kindle (아마존 킨들 스토어 데이터), DBLP (학술 인용 네트워크), ACM (학술 인용 네트워크).
	•	평가 지표: 10번의 실험 평균을 사용해 균형 잡힌 정확성을 측정하는 매크로 F1-AP 및 F1-AF 점수로 평가되었습니다.
	2.	비교 모델:
	•	기준 모델: TACO는 EWC, GEM, ERGNN (rs, rb, mf), SSM, SSRM 등과 비교되었습니다.
	•	성능 개선: TACO는 다른 모델보다 높은 F1-AP와 낮은 F1-AF 값을 기록하여 지식 보존 성능에서 우수성을 보였습니다.
	3.	주요 성능:
	•	Kindle 데이터셋: TACO는 F1-AP 82.97%, F1-AF 4.91%를 기록하며, SSM (F1-AP 78.99%, F1-AF 8.19%)을 비롯한 다른 모델보다 뛰어났습니다.
	•	DBLP 데이터셋: TACO는 F1-AP 84.60%, F1-AF 2.51%로, 강력한 메모리 보존 성능을 나타냈습니다.
	•	ACM 데이터셋: TACO는 F1-AP 70.96%, F1-AF 8.02%를 기록하여 다른 접근 방식들을 능가했습니다.

이 결과는 TACO의 지속적 학습 과제에서의 탁월한 성능을 입증하며, 특히 정확도를 유지하고 망각률을 최소화하는 데서 기존 모델들보다 뛰어난 성과를 보였습니다.



The TACO framework’s performance was evaluated on node classification tasks using three time-stamped graph datasets: Kindle, DBLP, and ACM. TACO was compared against several state-of-the-art continual graph learning (CGL) models, with F1-AP (average performance) and F1-AF (average forgetting) as the primary metrics.

	1.	Datasets and Metrics:
	•	Datasets: Kindle (Amazon Kindle store data), DBLP (academic citation network), ACM (academic citation network).
	•	Metrics: Macro F1-AP and F1-AF scores were used to measure balanced accuracy, averaged over 10 trials for statistical significance.
	2.	Comparison with Baseline Models:
	•	Baseline Models: TACO was compared against models such as EWC, GEM, ERGNN (rs, rb, mf), SSM, and SSRM.
	•	Significant Improvement: TACO demonstrated superior performance, achieving higher F1-AP and lower F1-AF values than other models, showing effective knowledge retention across tasks.
	3.	Performance Highlights:
	•	Kindle Dataset: TACO achieved an F1-AP of 82.97% with an F1-AF of 4.91%, outperforming models like SSM (78.99% F1-AP, 8.19% F1-AF).
	•	DBLP Dataset: TACO recorded an F1-AP of 84.60% and F1-AF of 2.51%, indicating strong memory retention capabilities.
	•	ACM Dataset: TACO maintained high accuracy with an F1-AP of 70.96% and F1-AF of 8.02%, surpassing other approaches.

These results underscore TACO’s effectiveness in continual learning tasks, highlighting its ability to maintain accuracy while minimizing forgetting, especially compared to conventional models.


<br/>
# 예제  



	•	Kindle 데이터셋: 아마존 킨들 스토어에서 가져온 데이터로, 각 노드는 킨들 아이템을 나타내며, 노드의 타임스탬프는 출시 날짜를 나타냅니다. 노드 간의 엣지는 “자주 함께 구매된” 관계를 나타냅니다. 이 데이터셋은 종교/영성, 어린이 전자책, 건강/피트니스, 공상과학/판타지, 비즈니스/머니, 로맨스 등의 카테고리로 구성된 6개의 클래스가 있습니다.
	•	DBLP 데이터셋: 학술 논문 인용 네트워크로, 각 노드는 논문을 나타내며, 타임스탬프는 출판 날짜를 의미합니다. 노드 간의 엣지는 인용 관계를 나타내며, 34개 학술지에 출판된 논문을 데이터베이스, 데이터 마이닝, 인공지능(AI), 컴퓨터 비전의 4개 클래스로 분류합니다.
	•	ACM 데이터셋: 이 또한 학술 논문 인용 네트워크이며, DBLP와 유사하게 노드는 논문을 나타내고 타임스탬프는 출판 날짜를 나타냅니다. 노드 간의 엣지는 인용 관계를 나타내며, 66개 학술지에서 정보 시스템, 신호 처리, 응용 수학, AI의 4개 클래스로 분류합니다.

각 데이터셋은 특정 시간 간격으로 나뉘며, 과거에 존재하지 않은 새로운 클래스가 등장할 수도 있습니다. 모델 학습과 테스트 시에는 선택된 클래스의 라벨이 임의로 마스킹되어 다양한 지속적 학습 환경을 시뮬레이션합니다 ￼.



	•	Kindle Dataset: This dataset consists of items from the Amazon Kindle store, where each node represents a Kindle item, with a timestamp indicating its release date. Edges represent a “frequently co-purchased” relationship. This dataset includes six categories: Religion & Spirituality, Children’s eBooks, Health, Fitness & Dieting, Sci-Fi & Fantasy, Business & Money, and Romance.
	•	DBLP Dataset: This academic citation network dataset has each node representing a paper with a publication date timestamp. Edges indicate citation relationships, and the dataset includes papers from 34 venues divided into four classes: Database, Data Mining, AI, and Computer Vision.
	•	ACM Dataset: Similar to DBLP, this dataset is an academic citation network where each node represents a paper with a publication date timestamp. Edges denote citation relationships, and papers are categorized into four classes: Information Systems, Signal Processing, Applied Mathematics, and AI, selected from 66 venues.

Each dataset is split by specific time intervals, and classes may vary in availability across tasks to simulate a dynamic continual learning environment where certain classes may reappear or new classes emerge ￼.


<br/>  
# 요약   




TACO 프레임워크는 지속적 학습 환경에서 그래프 데이터의 구조적 정보를 보존하며, 점진적으로 추가되는 데이터를 효율적으로 학습하는 모델입니다. 노드 임베딩을 활용해 유사도를 계산하고, 유사한 노드를 군집화하여 그래프를 축소하는 방식으로 메모리 사용을 최적화합니다. 테스트 결과, TACO는 Kindle, DBLP, ACM 데이터셋에서 기존의 여러 지속적 학습(CGL) 모델보다 높은 성능(F1-AP)과 낮은 망각률(F1-AF)을 기록했습니다. 예를 들어, Kindle 데이터셋에서 F1-AP 82.97%와 F1-AF 4.91%로 SSM 등 다른 모델을 능가하는 성과를 보였습니다. 이러한 결과는 TACO가 지속적 학습에서 기존 모델보다 효율적으로 정확도를 유지하며, 망각을 최소화할 수 있음을 입증합니다.



The TACO framework is designed to preserve structural information in graph data, enabling efficient learning of incrementally added data in a continual learning setting. By utilizing node embeddings to calculate similarity, TACO clusters similar nodes to reduce the graph size, optimizing memory usage. In tests, TACO outperformed several state-of-the-art continual learning (CGL) models, achieving higher F1-AP and lower F1-AF on datasets like Kindle, DBLP, and ACM. For instance, on the Kindle dataset, TACO achieved an F1-AP of 82.97% and F1-AF of 4.91%, surpassing models like SSM. These results demonstrate TACO’s superior ability to retain accuracy while minimizing forgetting compared to existing models in continual learning.

 

<br/>  
# 기타  



TACO 프레임워크는 지속적 학습 환경에서 노드 분류 작업을 수행하여 성능을 평가합니다. 노드 분류는 그래프의 각 노드가 특정 클래스로 분류되는 것을 목표로 합니다. 예를 들어, Kindle 데이터셋에서는 아마존 킨들 제품을 종교, 어린이 전자책, 건강/피트니스 등으로 분류하고, DBLP와 ACM 데이터셋에서는 학술 논문을 데이터베이스, 인공지능(AI), 컴퓨터 비전 등과 같은 학문 분야로 분류합니다. TACO는 이 작업에서 F1-AP와 F1-AF를 주요 평가 지표로 사용합니다. **F1-AP (Average Performance)**는 각 작업의 F1 점수를 평균하여 전체적인 성능을 나타내며, **F1-AF (Average Forgetting)**는 새로운 작업을 학습하면서 이전 작업에 대한 성능 저하를 측정해 모델의 망각률을 평가합니다. 높은 F1-AP와 낮은 F1-AF는 TACO가 새로운 데이터를 학습하면서도 기존 정보를 잘 보존한다는 것을 의미합니다.



The TACO framework is evaluated using node classification tasks within a continual learning setting. Node classification aims to assign each node in the graph to a specific class. For example, in the Kindle dataset, Amazon Kindle products are categorized into classes such as Religion, Children’s eBooks, and Health & Fitness, while in the DBLP and ACM datasets, academic papers are classified into fields like Database, Artificial Intelligence (AI), and Computer Vision. TACO uses F1-AP and F1-AF as its primary evaluation metrics. F1-AP (Average Performance) represents the overall performance by averaging F1 scores across tasks, while F1-AF (Average Forgetting) measures the decrease in performance on previous tasks when learning new tasks, indicating the model’s forgetting rate. High F1-AP and low F1-AF indicate that TACO retains prior knowledge effectively while learning new data.



망각률

망각률은 지속적 학습에서 모델이 새로운 작업을 학습할 때 이전 작업에 대한 성능이 얼마나 감소하는지를 나타내는 지표입니다. TACO 프레임워크에서는 **F1-AF (Average Forgetting)**를 사용하여 망각률을 측정합니다. F1-AF는 각 작업에서 얻은 F1 점수를 기준으로, 새로운 작업을 학습한 이후 이전 작업의 성능 저하를 평균하여 계산합니다. 낮은 F1-AF 값은 모델이 새로운 데이터에 적응하면서도 이전에 학습한 내용을 잘 유지하고 있다는 의미입니다. 반대로 F1-AF 값이 높다면 모델이 새로운 작업을 학습하는 동안 이전 정보를 많이 잊어버리고 있다는 것을 나타냅니다.

Forgetting Rate

The Forgetting Rate indicates how much a model’s performance on previous tasks declines as it learns new tasks in a continual learning setting. In the TACO framework, F1-AF (Average Forgetting) is used to measure this rate. F1-AF is calculated by averaging the drop in F1 scores for each task after learning new tasks. A low F1-AF value means the model effectively retains knowledge from previous tasks while adapting to new data. Conversely, a high F1-AF value suggests that the model has forgotten a significant amount of information from prior tasks as it learns new ones.

<br/>
# refer format:     

@article{Han2024,
  author    = {Xiaoxue Han and Zhuo Feng and Yue Ning},
  title     = {A Topology-aware Graph Coarsening Framework for Continual Graph Learning},
  journal   = {arXiv preprint arXiv:2401.03077},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.03077},
  note      = {Accessed: 2024-11-09}
}




Han, Xiaoxue, Zhuo Feng, and Yue Ning. “A Topology-aware Graph Coarsening Framework for Continual Graph Learning.” arXiv preprint arXiv:2401.03077 (2024). Accessed November 9, 2024. https://arxiv.org/abs/2401.03077.   


 

