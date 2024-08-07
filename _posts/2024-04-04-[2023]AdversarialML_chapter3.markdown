---
layout: post
title:  "[2023]Adversarial Machine Learning: Attack Surfaces, Defence Mechanisms, Learning Theories in Artificial Intelligence: Chapter3: Adversarial Attack Surfaces"  
date:   2024-04-04 09:23:29 -0400
categories: study
---

{% highlight ruby %}


한줄 요약: 

짧은 요약(Abstract) :    
이 장에서 저자들은 적대적 공격 표면을 탐색한다

저자들은 이를 통해 기계 학습의 취약점을 어떻게 이용할 수 있는지, 학습 시스템의 보안과 개인 정보 보호에 대한 공격으로부터 학습 알고리즘을 어떻게 견고하게 만들 수 있는지 검토한다

취약점을 탐색하기 위해 저자들은 지도 및 비지도 설정에서 다양한 공격 시나리오 하에서 여러 모델 학습 과정을 시뮬레이션할 수 있다

각 공격 전략은 기능 조작 또는 라벨 조작, 또는 둘 다를 수행할 수 있는 지능적인 적으로 가정된다

적들의 최적 공격 정책은 적대적 데이터를 출력하는 최적화 문제의 해결책에 의해 결정된다

저자들은 배운 지식을 적용하여 공격에 대한 방어를 더 잘하기 위해 학습 절차를 개선하고 강화할 수 있다

이 장에서 요약된 민감도 분석은 적대적 학습 알고리즘의 무작위화, 차별화, 신뢰성 및 학습 가능성에 대한 최적화 목표와 통계적 추론을 개발하기 위해 사용될 수 있다

이는 기계 학습 모델의 견고성, 공정성, 설명 가능성 및 투명성으로 연구 경로를 만든다
Useful sentences :  
*   


{% endhighlight %}  

<br/>

[Paper link](https://drive.google.com/drive/folders/1p1MrsOhGYsfzcphgMb3DdRDPq2tQpOSm?usp=sharing)  
[~~Lecture link~~]()  

<br/>

# 단어정리  
* 

<br/>
# 3.1 Security and Privacy in Adversarial Learning
## Evasion Attacks
Biggio 등은 배포된 분류기 시스템의 테스트 시간에 대한 적대적 보안을 논의한다

보안 평가는 그 다음 맬웨어 탐지에서 비선형 분류기 성능의 다양한 위험 수준에서 수행된다

경사 하강법 접근을 사용하여 차별화 가능한 판별 함수에 대한 안전한 분류기가 제안된다

적의 목표는 의사 결정 경계를 넘는 긍정적인 적대적 샘플로 분류기의 손실 함수를 최소화하는 것으로 정의된다

이 모델은 또한 적대적 공격 시나리오의 정의에 애플리케이션별 적대적 지식을 포함할 수 있다

이러한 적대적 지식에는 훈련 데이터, 특징 표현의 유형, 학습 알고리즘 및 그 결정 함수, 분류 가중치, 분류기로부터의 피드백에 대한 사전 지식이 포함된다

## Poisoning Attacks
보안에 민감한 환경에서는 기계 학습 알고리즘이 훈련 데이터가 자연적이고 잘 행동하는 분포에서 온다고 가정할 수 없다

Biggio 등은 SVM에 대한 독성 공격을 조사한다

기울기 상승 절차는 SVM의 비볼록 오류 표면의 국부 최대값으로서 적대적 예제를 계산하는 데 사용된다

기울기 상승 반복 후 각 공격 예제 업데이트에 이어 최적의 의사 결정 경계가 증분 SVM의 솔루션에서 계산된다

검색 절차는 많은 작은 기울기 단계를 수행한다

공격 예제가 훈련 데이터에서 너무 많이 벗어날 때 검색이 중지된다

악의적인 입력으로 인해 SVM의 의사 결정 경계가 변경되는 것은 스팸, 웜, 침입, 사기 탐지와 같은 애플리케이션 도메인에서 중요하다 

## Inference Attacks
Shokri 등은 인터넷 응용 프로그램에서 훈련 데이터에 대한 정보를 유출하여 상업적 분류 모델의 프라이버시 침해 문제를 조사한다

적대자는 주어진 입력에 대한 모델의 출력을 검색하기 위해 대상 모델을 블랙박스로 쿼리한다

이러한 입력은 대상 모델의 동작을 모방하는 그림자 모델을 훈련시켜 생성된다

블랙박스 대상 모델과 달리, 그림자 모델은 추론된 레코드에 대한 지상 진실 라벨을 알고 있다

블랙박스 모델은 아마존 ML과 구글 예측 API에서 이미지 데이터 세트로 훈련된 신경망 모델이다

블랙박스 모델의 세부 정보는 데이터 소유자로부터 숨겨져 있다

데이터 세트는 소매 구매, 위치 추적 및 병원 입원 체류에서 얻은 것이다

여기서 프라이버시 침해는 적대자가 모델의 출력을 사용하여 모델 입력에서 민감한 속성의 값을 추론할 수 있을 때 발생한다고 한다

속성 추론은 Shokri 등에 의해 주어진 데이터 레코드의 모델의 훈련 데이터 세트 내 클래스 멤버십 존재에 대한 클래스 멤버십 추론 측면에서 정의된다

제안된 클래스 멤버십 추론의 성공은 대상 모델의 공격 정밀도와 공격 리콜 측면에서 측정된다

그림자 모델은 힐 클라이밍 알고리즘으로 생성된 후보 레코드로 훈련된다

힐 클라이밍의 각 반복에서 대상 모델에 의해 높은 확신으로 분류된 후보 레코드가 제안된다

<br/>
# 3.1.1 Linear Classifier Attacks
Dalvi 등은 분류자의 성능을 분류자가 적대자에 적응하면서 적대자가 분류자로 하여금 거짓 부정적 결과를 생성하도록 만드는 게임으로 보며 분석한다

여기서 비용을 고려한 적대자는 비용을 고려한 분류자와 대비된다

이는 적대적 분류에서 데이터 생성 과정이 시간에 따라 변할 수 있을 뿐만 아니라 이 변화가 분류자 매개변수의 함수로 허용됨을 의미한다

적대적 분류는 따라서 적대자와 분류자라는 두 플레이어 간의 게임으로 정의되며, 여기서 분류자는 분류자의 예상 페이오프를 적대자의 비용 매개변수에 대해 최대화하는 페이오프 함수로 자신의 페이오프를 최대화한다

Dalvi 등은 적대자의 목표가 적대자의 예상 페이오프를 극대화하는 분류 기능 변경 전략을 찾는 것이라고 제안한다

적대적 예제는 나이브 베이즈 분류자의 페이오프 함수를 평가 함수로 사용하여 표준 특징 선택 알고리즘으로 생성된다

적대적 분류에서 계산적으로 처리 가능한 나쉬 균형 전략의 이론은 양자 비영합 게임을 분석함으로써 여전히 열린 질문으로 남아있다

Lowd 등은 공격을 받는 선형 분류자를 위한 적대적 학습 알고리즘을 소개한다

적대적 학습의 목표는 도메인 특정 기능 표현을 구성하거나(i) 훈련 데이터 분포에 대한 확률적 과정을 가정하지 않고(ii) 분류자의 의사 결정 경계의 일부를 학습하고 공격하는 것이다

Lowd 등은 적대자가 악의적 예제와 비악의적 예제를 구별하기 위해 분류자에게 멤버십 쿼리를 보낼 수 있다고 가정한다

가능한 멤버십 쿼리의 계산 복잡도는 각 기능 차원을 따라 선 검색의 다항식 수에 의해 제한된다

그런 다음 적대적 학습 알고리즘은 적대자가 배우게 될 비악의적 인스턴스 공간을 통해 선형 적대적 비용 함수를 최소화한다

최적의 적대적 비용 함수는 적대자에게 접근 가능한 기본 적대적 예제와 가장 유사한 비악의적 예제를 생성한다

Lowd 등은 또한 나이브 베이즈 모델, 선형 커널을 가진 지지 벡터 머신, 스팸 필터링을 위해 부울 특징을 학습하는 최대 엔트로피 모델과 같은 선형 분류자에 대한 적대적 훈련 실험을 보여준다

제안된 학습 프레임워크(ACRE라고 함)는 공격자 또는 적대자와 방어자 또는 분류자 모두를 연구하는 데 유용하다

적대적 비용 함수를 최소화함으로써 적대자가 분류자를 효율적으로 충분히 배

울 수 있는지 여부를 결정하는 데 사용할 수 있다

<br/>
#  3.2 Feature Weighting Attacks
전통적인 기계 학습 알고리즘은 통제되고 고품질의 데이터에서 알고리즘 훈련을 수행할 수 있다고 가정한다

실제 세계에서 기계 학습은 노이즈가 많고 불확실한 데이터에서 수행된다

여기에서 강건한 분류기는 훈련 중에 노이즈가 있는 특징을 가정할 때만 테스트 중에 노이즈가 있는 특징을 예측할 수 있다

또한 강건한 분류기는 가능한 한 많은 유익하거나 중요한 특징에 대해 훈련되어야 하는 밀집된 분류기여야 한다

적대적 환경에서 특징 가중치 기법에 대한 이러한 고려 사항이 중점이다

여기에서 적대적 데이터는 지능적인 적대자에 의해 생성된 데이터로 자연 세계에서 발견되는 무작위 노이즈와 다르다

<br/>
# 3.3 Poisoning Support Vector Machines
적대적 보안 메커니즘에 따르면, 독성 공격은 특별히 제작된 공격 지점이 훈련 데이터에 주입되는 인과적 공격이다

독성 공격에서는 적대자가 훈련 데이터베이스에 접근할 수 없지만 새로운 훈련 데이터를 제공할 수 있다

독성 공격은 대규모 학습 시스템의 보안을 손상시킨다

이 시스템은 복잡한 대규모 데이터 세트에서 숨겨진 패턴을 추론하여 행동 통계를 지원함으로써 의사 결정을 지원한다

이전의 독성 공격은 이상 탐지 방법으로 연구되었다

<br/>
#  3.4 Robust Classifier Ensembles
Biggio 등은 선형 분류기 앙상블이 감독 학습의 정확성뿐만 아니라 견고성을 향상시킬 수 있다고 제안한다

그 이유는 앙상블의 전체를 손상시키기 위해 하나 이상의 분류기를 회피하거나 오염시켜야 하기 때문이다

훈련 전략은 데이터의 구별 가능한 특징과 비구별 가능한 특징 사이에 특징 가중치를 균등하게 분배한다

분류기의 구별 가능한 가중치를 손상시키면 분류기의 정확도도 손상될 수 있다

그러므로 견고한 분류기 앙상블의 목표는 견고성과 정확성 사이의 올바른 균형을 찾는 것이다

여기에서 적대자는 분류기를 조작하기 위해 많은 특징 값을 수정해야 한다

Biggio 등은 적대적 알고리즘에서 가중치를 분배하기 위해 부스팅과 랜덤 서브스페이스 방법(RSM)을 설계한다

적대적 행위는 분류기의 완전한 지식을 가진 최악의 시나리오와 분류기에 대한 대략적인 지식만 있는 평균적인 시나리오의 두 가지 시나리오 측면에서 모델링된다

그런 다음 앙상블 구별 함수는 원래 특징 집합에서 무작위로 선택된 다른 하위 집합에 대해 훈련된 다양한 선형 분류기의 평균을 통해 얻어진다

<br/>
#  3.5 Robust Clustering Models  
적대적 군집화 문제는 데이터 세트의 확률적 노이즈가 아닌 표적 적대적 조작을 다루는 군집 안정성 기준으로 해결될 수 없다

Biggio 등은 단일 연결 계층적 군집화를 위한 독성 및 혼란 공격을 고안한다

독성 공격에서 적의 목표는 군집화 품질 측정에 적대적 예시를 주입하는 것이다

혼란 공격에서 적의 목표는 특징 값을 조작하여 기존 군집에 데이터 샘플을 숨기는 것이다

이러한 공격에서 군집은 분할 군집화 알고리즘의 하드 파티션(및 소프트 파티션)뿐만 아니라 연결 유형 군집화 알고리즘에서의 지배적 집합(및 매개변수화된 하위 집합 계층)으로 정의된다

Biggio 등은 조작되지 않은 훈련 데이터와 조작된 적대적 데이터 사이의 거리 측정 기준을 포함한 공격 시나리오에 추가 제약을 둔다

적의 지식 정도는 공격 샘플의 확률 분포의 엔트로피에 의해 인코딩된다

확률 분포는 적의 지식 공간 위에 정의되어 데이터 세트 및 군집화 알고리즘의 매개변수화에 대한 정보를 제공한다

적의 지식을 가정한 상태에서 적의 목표는 (i) 군집화 간의 실제 값 거리 측정 기준으로 평가된 독성 공격 샘플 및 (ii) 공격 샘플과 목표 샘플 간의 비음수 실제 스칼라 발산 측정 기준으로 표현된 목적 함수이다

<br/>
#  3.6 Robust Feature Selection Models  
전통적 분류기는 적대적 환경의 변화에 지속적으로 적응할 수 없기 때문에 Kolcz 등은 테스트 데이터의 분포가 원본 훈련 데이터의 분포와 다를 때 우아하게 열화되는 분류기를 설계하려고 한다

이는 분류를 위해 덜 중요한 특징에 대한 재가중치를 통해 특징 선택 과정을 통해 이루어진다

특징 가중치는 데이터에서 개념 이동에 강건함을 제공하면서 모델 성능을 향상시키지만 모델의 추가 계산 비용을 초래한다

이 접근 방식의 직관은 학습 알고리즘에 대한 특징의 가중치 분포가 비지도 및 지도 학습을 위한 특징의 중요성을 반영한다는 것이다


Kolcz 등은 두 단계 접근 방식을 통해 강건한 분류기 훈련을 구상한다

여기서 분류기는 첫 번째 단계에서 특징에 가중치를 할당한 후 이를 특징 가중치를 통해 변형하여 두 번째 단계에서 최종 모델을 유도한다

최종 모델은 최적화될 목적 함수를 만족한다

지도 및 비지도 학습 모두에 대한 단일 최고의 재가중치 체계가 문헌에서 사용할 수 없기 때문에 Kolcz 등은 특징 가중치에 대한 여러 선택을 실험한다

특징 가중치의 목적 함수는 Kolcz 등에 의해 정규화 위험 최소화의 특별한 경우로서 이차 형태 정규화 및 볼록 손실 함수로 공식적으로 분석된다


Kolcz 등의 실험에 포함된 특징 가중치 방법에는 특징 배깅, 분할 로지스틱 회귀, 신뢰 가중 학습, 특징 노이즈 주입 및 샘플 선택 편향 수정이 포함된다

<br/>
# 3.7 Robust Anomaly Detection Models
Kloft 등은 (온라인 센트로이드) 이상 탐지 알고리즘에 대한 적대적 예시를 탐구한다

이상 탐지는 컴퓨터 보안 응용 프로그램에서 유한 슬라이딩 윈도우를 통해 비정상적인 사건을 찾는다

예를 들어 자동 서명 생성 및 침입 탐지 시스템에서 사용된다

독성 공격은 훈련 데이터에서 적대적 예시를 생성하기 위해 가정된다

이때 적대자는 훈련 데이터의 일정 비율을 통제한다

이상 데이터 포인트는 훈련 데이터의 경험적 평균으로부터의 유클리드 거리에 따라 측정된다

경험적 평균은 비정상 데이터에 대한 유한 슬라이딩 윈도우 온라인 알고리즘에 의해 훈련 데이터에서 계산된다

적대자는 경험적 평균 지점을 적대적 예시 쪽으로 밀어내어 이상 탐지 알고리즘으로 하여금 이상 데이터 포인트를 정상 훈련 데이터로 받아들이게 한다

<br/>
# 3.8 Robust Task Relationship Models
Zhao 등은 다중 작업 관계 학습(MTRL)에서 작업 관련성에 대한 데이터 독성 공격을 제안한다

MTRL에서 최적의 공격은 임의의 대상 작업과 공격 작업에 적응할 수 있는 이중 최적화 문제를 해결한다

이러한 공격은 확률적 기울기 상승 절차에 의해 발견된다

MTRL의 취약성은 특징 학습 접근, 저랭크 접근, 작업 군집 접근, 그리고 작업 관계 접근으로 분류되며, 여기서 학습 목표는 예측 함수를 공동으로 학습하는 것이다

그 다음 임의의 볼록 손실 함수와 양의 준정부 공분산 행렬을 가진 선형 예측 함수의 MTRL이 연구된다

적의 목표는 공격 작업 집합에 독성 데이터를 주입하여 대상 작업 집합의 성능을 저하시키는 것으로 정의된다

적의 페이오프 함수는 대상 MTRL 모델의 완전한 지식을 가진 대상 작업에서 훈련 데이터의 경험적 손실로 정의된다

기울기 상승 절차에서 독성 데이터는 적대적 페이오프 함수를 최대화하는 방향으로 반복적으로 업데이트된다

예측 함수는 회귀 작업에 대한 최소 제곱 손실 함수이고 분류 작업에 대한 제곱 힌지 손실 함수이다

예측 성능은 분류 작업의 곡선 아래 영역을 최대화하고 회귀 작업의 정규화된 평균 제곱 오류를 최소화하여 평가된다

<br/>
# 3.9 Robust Regression Models
높은 차원의 회귀 문제에서의 적대적 감독 학습을 Liu 등이 연구한다

적대적 조작의 대상은 훈련 데이터이며 이러한 시나리오는 독성 공격으로 불린다

견고한 감독 학습 모델과 달리 제안된 적대적 감독 학습은 입력 분포와 특징 행렬의 이후 특성, 특징 독립성 및 신호-노이즈 비율에 대한 강력한 통계적 가정을 완화한다

결과적인 성능 보장은 견고한 주성분 회귀를 기준 모델로 삼아 비교된다

이러한 기계 학습 모델은 스팸 필터링, 트래픽 분석 및 사기 탐지에 응용되어 강력한 적대자에 대한 보안을 강화한다

알고리즘 설계에서 다루어야 할 학습 과제는 차원 축소가 신뢰할 수 있는 저차원 공간 패턴을 회복할 수 있고, 이 공간에서 수행된 회귀가 정확한 예측을 회복할 수 있음이다

또한 이러한 설계 목표는 훈련 데이터 세트에 적대적으로 독성이 있는 샘플에도 불구하고 달성되어야 한다

이를 위해 저자들은 가능한 한 공간을 정확하게 회복하는 견고한 행렬 분해 알고리즘을 개발하고, 회복된 기저와 잘린 최적화를 사용하여 선형 모델 매개변수를 추정하는 트림된 주성분 회귀를 사용한다

견고한 회귀의 솔루션은 노이즈 잔여물이다

이는 적대적 데이터가 회귀 모델 설계와 추정치를 현저하게 왜곡하는 능력에 미치는 영향을 연구하는 데 사용된다

이로 인해 적대적 학습을 위한 유계 손실 함수의 설계가 이루어진다

그런 다음 적대자는 차원 축소 알고리즘과 회귀 모델에서 최악의 성능을 유발하는 독성 전략을 생성하는 것으로 가정할 수 있다

이러한 공격 중 가장 효과적인 것은 학습된 추정치를 최대한 수정하는 방향으로 데이터 샘플을 이동시킨다

실험 결과는 적대적 데이터 독성에 견고한 선형 회귀 모델과 비교된다

이러한 적대적 학습 모델은 공격 시나리오를 설정하여 오분류 오류 비용을 검증하는 것보다 적대자에 대한 방어에 더 초점을 맞추어 분포적 견고함을 가진 적대적 학습 알고리즘을 생산하는 경향이 있다

<br/>
#  3.10 Adversarial Machine Learning in Cybersecurity
적대적 공격은 컴퓨터 비전, 자연어 처리, 사이버 공간 보안, 그리고 물리적 세계에서 다양한 응용 분야를 가진다

컴퓨터 비전에서는 이미지 분류 및 객체 감지를 위한 적대적 공격이 생성된다

자연어 처리에서는 텍스트 분류 및 기계 번역을 위한 적대적 공격이 생성된다

사이버 공간 보안에서는 클라우드 서비스, 맬웨어 탐지 및 침입 탐지를 위한 적대적 공격이 생성된다


물리적 세계에서는 대규모 모델 및 데이터 세트에 대한 적대적 훈련을 확장하기 위해 적대적 공격이 생성된다

Kurakin 등은 카메라 및 기타 센서를 입력으로 하는 물리적 세계 시나리오를 논의한다

Eykholt 등은 도로 표지 분류의 실제 사례를 위해 다양한 물리적 조건에서 견고한 시각적 적대적 변조를 생성한다

여기서 컴퓨터 비전 작업은 물리적 시스템에서 제어 파이프라인으로 작용하며, 견고한 물리적 변조를 생성하는 주요 도전 과제는 환경 변동성이다

Melis 등은 로봇 비전 시스템에 대한 물리적 세계 공격을 생성한다

Sharif 등은 감시 및 접근 제어를 위한 얼굴 인식 모델을 사용하는 얼굴 생체 인식 시스템에 대한 물리적 세계 공격을 생성한다

Xiao 등은 Lp 거리가 패널티 적대적 변조의 지각 품질 측정 기준으로 작용하는 적대적 변조의 공간 변환에 대해 논의한다

Akhtar 등은 컴퓨터 비전에서 딥 러닝에 대한 적대적 공격을 조사하는 리뷰를 수행한다

게임 이론적 공격과 비교하여 물리적 세계 공격은 훈련된 감지를 속이기 위해 객체의 외관을 물리적으로 변경한다

그들은 게임 이론적 설정에서 한 번만 공격 플레이에 제한되며, 특정 위협에 적용된다

그들은 조합 게임 이론에서 몬테 카를로 트리 검색과 같은 검색 휴리스틱을 사용하여 우리의 게임에서 확률적 검색 정책을 생성하는 데 적용될 수 있다


딥 러닝 방법은 감지, 모델링, 모니터링, 분석 및 각종 위협으로부터 민감한 데이터와 보안 시스템을 방어하는 등의 사이버 보안 목표를 발전시키는 데 사용될 수 있다

Rossler 등은 DeepFakes, Face2Face, FaceSwap, NeuralTextures와 같은 얼굴 조작의 주요 대표자로서 데이터 기반 위조 탐지기에 기반한 합성 이미지 생성 및 조작 벤치마크에 대해 논의한다

Matern 등은 DeepFakes와 Face2Face와 같은 얼굴 편집 알고리즘에서 얼굴 추적 및 편집으로부터의 아티팩트를 전시

하여 조작을 드러낸다


사이버 보안에서 적대적 기계 학습의 추가 응용 프로그램에는 맬웨어 탐지, 맬웨어 분류, 스팸 탐지, 피싱 탐지, 봇넷 탐지, 침입 탐지 및 침입 방지 및 이상 탐지가 포함된다

Tong 등은 PDF 맬웨어 탐지에서 회피 공격을 논의한다

Melis 등은 안드로이드 맬웨어 탐지에 설명 가능한 기계 학습 모델을 적용한다

Marino 등은 데이터 기반 침입 탐지 시스템에서 잘못된 분류를 설명한다

Corona 등은 침입 탐지 시스템(IDS) 및 컴퓨팅 인프라에서 적대적 공격의 분류 체계를 제공한다

Demetrio 등은 맬웨어 바이너리 분류에 의미 있는 설명을 제공하기 위해 특징 귀속을 제안한다

Fleshman 등은 맬웨어 탐지 모델을 사용하는 기계 학습 기반 안티 바이러스 제품의 시스템 견고성을 정량화한다 

<br/>
#  3.10.1 Sensitivity Analysis of Adversarial Deep Learning
딥러닝 모델의 기계 학습 모델을 이해하는 것은 그 정확성을 검증하고 알고리즘 편향과 정보 유출을 탐지하며 데이터에서 새로운 패턴을 학습하는 데 유용하다

복잡한 기계 학습 모델, 특히 최첨단 성능을 자랑하는 모델은 해석 가능성을 희생하면서 얻어진다

학습 가능성과 견고성 사이의 균형을 맞추어야 한다

"학습 가능성"이란 분류기가 노이즈에 관계없이 정확한 라벨을 예측하는 능력을 의미하며, "견고성"이란 노이즈가 있든 없든 예측이 동일하다는 것을 의미한다

우리가 관찰하는 교환 관계는 더 많은 학습 가능성이 더 적은 견고성을 가져오고 그 반대의 경우도 마찬가지다

민감도 분석은 독립 변수의 변화에 대해 종속 변수에 미치는 영향을 연구하는 것이다

이는 적대적 학습에서 블랙박스 공격 시나리오를 연구하는 데 유용하며, 여기서 학습 모델과 과정의 출력은 여러 입력의 불투명한 함수다

즉, 기계 학습에 대한 입력과 출력 간의 정확한 관계는 분석적으로 잘 이해되지 않는다

기계 학습에서 민감도 분석은 인공 지능의 복잡한 시스템 분석에서 중요한 역할을 한다

이는 연구 대상 시스템의 모델을 결정하는 데 사용될 수 있다

데이터 분석 출력 변동성 요인에 기여하는 모델 매개 변수를 식별할 수 있다

분석 요소와 그 상호 작용에 대한 보정 연구에서 관심 영역의 최적 검색 영역을 식별할 수 있다

마지막으로 분석 방법에 대한 상관관계/분류/회귀, 베이지안 추론 및 기계 학습 내에서 평가된 영향력 있는 응답의 출력 분포를 생성하는 분석 모델을 평가할 수 있다 


<br/>  
# 요약 또는 기타 정보  
* "Black box attacking"은 적대적 기계 학습에서 일반적으로 사용되는 기법 중 하나로, 공격자가 타깃이 되는 시스템의 내부 작동 방식에 대한 직접적인 지식이나 접근 권한이 없는 상황에서 이루어집니다. 공격자는 시스템의 입력과 출력만을 관찰하여, 시스템의 의사 결정 프로세스를 이해하고 이를 조작하는 방법을 찾아내려고 합니다.  

* Black Box Attack의 주요 접근 방식:  
모델 출력의 탐색:
	공격자는 시스템에 다양한 입력을 제공하고, 그에 따른 출력을 관찰합니다. 이를 통해 시스템이 어떤 입력에 대해 어떤 반응을 보이는지 학습하게 됩니다.
그림자 모델 학습:
	공격자는 수집한 데이터를 사용하여 원본 모델과 유사하게 동작하는 그림자 모델을 훈련시킵니다. 이 모델을 통해 공격자는 보다 정교한 공격을 계획할 수 있으며, 원본 모델의 행동을 더 잘 예측할 수 있습니다.
공격 샘플 생성:
	그림자 모델을 이용하여, 원본 모델을 속일 수 있는 적대적 예제를 생성합니다. 이러한 예제들은 원본 모델이 잘못된 분류나 예측을 하도록 유도할 수 있습니다.
적대적 예제의 반복적 개선:
	공격자는 생성된 적대적 예제를 실제 모델에 적용해보고 결과를 분석합니다. 이를 통해 예제들을 점점 더 개선해나가면서 원본 모델을 더 효과적으로 공격할 수 있는 방법을 찾아냅니다.

* 질문거리:
보안 대책:
	적대적 공격에 대비하여, 모델은 어떻게 학습과정에서 보다 강건하게 만들 수 있을까요?
	모델의 강건성을 평가하기 위해 어떤 테스트 방법이 사용될 수 있나요?
그림자 모델의 한계:
	그림자 모델이 원본 모델을 완벽하게 모방할 수 있는지, 그 한계는 어디인가요?
	그림자 모델 학습을 위해 필요한 데이터는 얼마나 많이 필요하며, 데이터의 품질은 어느 정도 되어야 하나요?
적대적 예제의 탐지:
	시스템은 어떻게 적대적 예제를 탐지하고 대응할 수 있나요?
	적대적 예제를 식별하는 데 있어 가장 효과적인 기술은 무엇인가요?
윤리적 고려:
	적대적 공격 기술의 발전이 가져올 수 있는 윤리적 문제에는 어떤 것들이 있나요?
	이러한 기술의 남용을 방지하기 위해 어떤 법적 또는 규제적 조치가 필요할까요?

이러한 질문들은 적대적 기계 학습을 연구하거나 이를 응용하는 분야에서 깊이 있는 논의를 촉진할 수 있습니다. 이를 통해 보다 안전하고 신뢰할 수 있는 기계 학습 시스템의 발전에 기여할 수 있습니다.