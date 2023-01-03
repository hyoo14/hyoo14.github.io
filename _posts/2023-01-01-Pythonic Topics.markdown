---
layout: post
title:  "Pythonic Topics"
date:   2023-01-01 20:30:30 +0900
categories: study
---







{% highlight ruby %}


프레임워크(Django), Test, Security, GCP 등등.  



{% endhighlight %}  

# test  
# 1. 디버깅 출력에는 repr 문자열을 사용하라     


타입이 안 보이는 문제가 있으니  
** repr 문자열 형식 사용하면 편리  
** repr 문자열 : repr(), str.%r, F-string에서 !r 접미사 뭍이지 않고 텍스트 치환식 사용하는 경우       


# 2. TestCase 하위 클래스를 사용해 프로그램에서 연관된 행동 방식을 검증하라       

  
** TestCase 상속 받아서 test code 짜면 됨  
** 앞선 테스트 코드 참조  
** assertEqual, assertIn, assertTrue (값이 같은지, 포함하는지, 참인지)와 같은 전용 메서드 사용하면 됨   


** 준비코드를 줄이려면 subTest 도우미 메서드를 사용해 데이터 기반 테스트를 정의해야함. 
** 예외 발생 검증 위해 with문 안에 assertRaises 메서드 사용(try/except문처럼)  
** 미묘한 경우(edge case)가 많다면 각 함수를 서로 다른 TestCase 하위 클래스에 정의할 때도 있음. 
** subTest 도우미 메서드로 한 테스트 메서드 안에 여러 테스트 정의 가능. 


# 3. setUp, tearDown, setUpModule, tearDownModule을 사용해 각각의 테스트를 격리하라       


테스트 환경 구축(테스트트 하네스test harness), 격리리 중요  
** setUp : 유닛테스트에서 테스트 메서드 실행 전(준비 단계)  
** tearDown : 유닛테스트에서 테스트 메서드 실행 이후(정리 단계)    
** setUpModule : 통합테스트에서 테스트 메서드 실행 전  
** tearDownModule : 통합테스트에서 테스트 메서드 실행 전  
** 결과적으로 단위테스트, 통합테스트 격리 편리     



# 4. 목을 사용해 의존 관계가 복잡한 코드를 테스트하라  
** DB에 데이터 채우고 트랜젝션 주고 받는 것은 cost 큼, 목 사용 괜찬
** 목(mock, mocking)은 DB와 row데이터들이 있는 것 처럼 흉내내서 응답  
** 관심 없는 파라미터는 unittest.mock.ANY 상수 사용해 퉁침   

** assert_called_with 메서드는 가장 최근 목 호출 때 전달된 인자 알려줌   
** unittest.mock.patch 관련 함수들은 목 주입을 더 쉽게 만들어줌.? 




 
# 5. 의존 관계를 캡슐화해 모킹과 테스트를 쉽게 만들라.   


** 반복적인 준비 코드 많다면 를 많이 사용해야 한다면, 클래스로 캡슐화 권장   
** unittest.mock 내장 모듈의 Mock 클래스로 시뮬레이션할 수 있음 
** end-to-end 테스트를 위해서는 도우미 함수를 더 많이. 포함하도록 코드를 리팩터해야함 



# 6. pdb를 사용해 대화형으로 디버깅해라   
*유용한 명령어  
**where: 현재 실행 중인 프로그램의 호출 스택 출력. 
**up: 현재 관찰 중인 함수를 호출한 쪽(위)으로 호출 스택 영역을 한 단계 이동해서 해당 함수의 지역변수 관찰  
**down: 실행 호출 스택에서 한 수준 아래로 이동  
*실행 제어 명령어  
**step: 디버거 프롬프트를 표시  
**next: 다음 제어를 디버거로 돌려서 디버거 프롬프트를 표시  
**return: 반환될 때까지 프로그램 실행  
**continue: 다음 중단점에 도달할 때까지 프로그램 계속 실행   
**quit: 디버거 나가면서 프로그램 중단  


** 독립 실행한 파이썬 프로그램에서 예외가 발생한 경우, pdb 모듈을 사용(python -m pub -c continue 프로그램 경로)하거나 대화형 파이썬 인터프리터(import pdb; pdb.pm())를 사용해 디버깅할 수 있음  




# 7. 프로그램이 메모리를 사용하는 방식과 메모리 누수를 이해하기 위해 tracemalloc을 사용하라     
** 파이썬 프로그램이 메모리를 사용하고 누수하는 양상을 이해하기는 어려움   
** 메모리 사용을 디버깅하는 방법  
*** 1. gc 내장 모듈을 사용해 쓰레기 수집기가 알고 잇는 모든 객체 나열  
*** 둔탁하지만 메모리 사용 어디인지, 감 잡기 쉬움  
*** 이러한 gc.get_objects의 문제점은 객체가 어떻게 할당됐는지 알려주지 않아서 메모리 누수 객체 알아내기 어렵
*** 2. tracemalloc 내장 모듈 사용(권장)    
*** 객체를 자신이 할당된 장소와 연결  
*** 메모리 사용 이전과 이후 스냅샷을 만들어 비교하면 변경 부분 파악 가능  





# Django에서 FBV, CBV  
fbv  
-다 펑션  




cbv  
-뭘 많이 제공  
-미리 만들어 놓아서 제공을 장고에서 함  
-더 쉽게 만듬  
-코드가 짧아지고  
-공통함수 재사용, DRY: Don't Repeat Yourself. 원칙 따르도록 함  
-메소드명을 HTTP 메소드명과 동일하게 사용(원칙, 이렇게 해야 별도 처리(데코레이터?) 없이 인식)  
-FBV 패턴에서는 각 URL마다 호출할 메소드를 직접 명시했다. (views.snippet_list())  
CBV 패턴에서는 상속했던 APIView 클래스의 클래스 메소드인 as_view()를 호출  


• FBV vs CBV  
	• FBV가 먼저 생김, 이후 CBV  
	• FBV 심플, 단순(편하게 구현, 읽기 편한 로직, 데코레이터 사용이 명료) vs CBV 상속, 믹스인 기능 통해 코드 재사용성 높임(중복 줄임, 확장/재사용 용이, 다중상속  Mixin 가능, HTTP   METHOD가 클래스 안에서 나누어 처리, 강력한  Generic Class View)  
	• 단점: FBV 확장/재사용 어려움 vs CBV: 읽기 어렵고, 상속/Mixin으로 인해 코드 이해 위해 곳곳 찾아봐야함  
	• 상속/재사용 많은 곳은 CBV, 아닌 곳 FBV, 적절히 섞어서 사용하는 것도 가능  
	• 참고문헌  
		○ https://velog.io/@yejin20/DjangoFBV-%EC%99%80-CBV  
		○ https://leffept.tistory.com/318  
		○ https://dingrr.com/blog/post/djangofbvs-vs-cbvs-%ED%95%A8%EC%88%98%ED%98%95-%EB%B7%B0-vs-%ED%81%B4%EB%9E%98%EC%8A%A4%ED%98%95-%EB%B7%B0  



# Django Security 관련  
*DDOS(Distributed Denial of Service) 대비??  
https://security.stackexchange.com/questions/114/what-techniques-do-advanced-firewalls-use-to-protect-againt-dos-ddos  
**bad client 구분할 정답은 없음  
**최선의 노력을 하는 것  
***로드벨런싱, 패일오버시스템 구축  
***IP 블랙리스트  
***관리자에게 알람 기능, 모니터링  
***CloudFlare에서 제공하는 Content Delivery Network?(내 서버 분산시켜 놓는..)    


mtv 구조(model, template, view)   

client: front.com -> url.py -> views.py -> models.py -> DB  

DB -> models.py -> views.py -> template: index.html -> front.com  



tdd (리팩토링을 위해서도 필요)  
  
setup assertEqual, assertIn 쓰이네 실제 테스트에서도  







# 기타: Backend Framework 관련  

플라스크란?
Python 기반 micro 프레임워크

What does “micro” mean?
The “micro” in microframework means Flask aims to keep the core simple but extensible.
Everything else is up to you, so that Flask can be everything you need and nothing you don’t.

심플하지만 확장가능하게 유지한것을 의미한다.
즉, 어떻게 사용하냐에 따라 좋은 프레임워크가 될 수 있고 그렇지 않을 수도 있다.



(MSA에 적합, 단일 기능만 적용할 때!)
(fast api가 무섭게 오는 중)


Flask와 Django의 차이점
구분	Flask	Django
생성년도	2010	2005
프레임워크 성향	MSA	모놀리식
어드민페이지	X	O
ORM (=Object Relational Mapping)	X	O
지원기능	상대적으로 적음	상대적으로 많음
러닝커브	상대적으로 낮음	상대적으로 높음
코드크기	상대적으로 작음	상대적으로 큼
유연성	좋음	제한됨
개발자의책임	상대적으로 큼	상대적으로 작음




어느 프레임워크가 더 좋은가요?
어느 쪽이 더 좋다고 할 순 없습니다. 프로젝트에 맞는 프레임워크를 선택해야 하면 됩니다.
일반적으로 MSA형태의 소규모 프로젝트에 단일 기능을 구현하는 웹에 Flask가 보다 더 적합합니다.

간혹 Django는 소규모 프로젝트에 한해서 오버스펙이 되기도 합니다.

Flask의 경우, 지원기능이 적은만큼 필요한 기능을 구현해야 할 때마다, 별도의 라이브러리를 설치하고 Flask 어플리케이션과 바인딩 해줘야합니다.

즉, 살이 붙으면 붙을수록 개발 cost도 높아집니다.

주의점 : 너무 커스텀하지말자























# GCP 관련  
-(04)Cloud IAM(
-(05)Compute Engine
-(08)GCS(Google Cloud Storage)
-(10)BigQuery
-(11)Cloud Composer
-(23)CloudFunctions



-(04)Cloud IAM(Identity and Access Management)-누가 어떤 리소스에 어떤 엑세스 권한 갖는지 제어
-(05)Compute Engine-가상 머신 서비스(AWS의 EC2), 네트워크에 연결된 가상 서버 제공
-(08)GCS(Google Cloud Storage)-(AWS의 S3) 객체 Repository
-(10)BigQuery-데이터 웨어하우스(탄력확장, 전용query)
-(11)Cloud Composer-워크플로우 관리 서비스(파이프라인 작성, 예약 및 모니터링)
-(23)CloudFunctions-(AWS의 람다) 클라우드 인프라 및 서비스의 이벤트 연결 함수 작성 가능



-(04)Cloud IAM(Identity and Access Management)-누가 어떤 리소스에 어떤 엑세스 권한 갖는지 제어
—G Suite - 조직에 생성된 모든 구글 계정으로 구성된 가상 그룹
—IAM 정책 - 사용자에게 부여되는 액세스 권한 유형 가진 리스트
—각 계정은 서비스 키 쌍과 연결되며 서비스 계정 키는 크게 GCP 관리용(서비스간 인증용)과 사용자 관리용(사용자계정) 2가지로 나뉨
—gcloud 명령어 사용 

-(05)Compute Engine-가상 머신 서비스(AWS의 EC2), 네트워크에 연결된 가상 서버 제공
—쿠버네티스 엔진 통해 도커 컨테이너를 실행 및 관리 조정할 수 있음
—하나의 프로젝트는 여러 개의 인스턴스를 가질 수 있음
—실시간 이전(라이브 마이그레이션) 가능(껐다 켤 필요 없이)
—관리형인스턴스그룹은 자동으로 오토 스케일링 지원, 자동복구정책 설정가능, 트래픽 분산가능, 크게 영역/리전 관리 2개로 나뉨
—비관리형은 위가 제공되지 않음
—스냅샷은 인스턴스의 특정 시점에 백업 가능

-(08)GCS(Google Cloud Storage)-(AWS의 S3) 객체 Repository
—프로젝트 > 버킷(전체 gcs중 교유명, repository 등급 나뉨) > 객체(버킷에 저장되는 파일들, 객체/메타로 나뉨, 변경불가, 중복저장(안정성))
—repository 등급은 멀티리저널(자주 어세스), 리저널(분산용), 니어라인(자주x), 콜드라인(가장저렴, 백업용) 4가지

-(10)BigQuery-데이터 웨어하우스(탄력확장, 전용query)
—쿼리로 ml, gis(지리정보시스템) 가능

-(11)Cloud Composer-워크플로우 관리 서비스(파이프라인 작성, 예약 및 모니터링)
—아파치 에어플로 기반 통합서비스, python기반으로 dag/task 코드 작성 가능
—쿠버네티스(배포환경)+클라우드 sql(메타데이터 저장)+에어플로(웹 서버 호스팅)+Stackdriver(로그관리)
—airflow - python기반 워크플로우 통합도구
—DAG - 방향비순환그래프로 하나의 워크플로우라고 생각하면 됨
—operator - DAG 안의 작업함수로 이를 이용하여 태스크 만듬(bash/python 정도가 있음, 각각 bash/python 실행용)
—저장은 자동으로 gcs서 버킷 만들고 스토리지 FUSE를 사용하여 에어플로 인스턴스와 gcs버킷 매핑함

-(23)CloudFunctions-(AWS의 람다) 클라우드 인프라 및 서비스의 이벤트 연결 함수 작성 가능
—코드가 완전 관리형 환경에서 실행, 프로비저닝/서버관리 필요x
—클라우드 연결/확장에 용이 - gcs파일 업로드,pub/sub 메세지확인 등 gcp 서비스 이용 가능
—cloud이벤트 - 클라우드 환경에서 발생하는 모든 상황
—서버리스 - 서버관리, 소프트웨어구성, 프레임워크 업데이트, 운영체제 패치 등 신경 쓸 필요 x, 사용자코드만 추가o 
—배포시 자동으로 프로비저닝함(ETL(파일 생성 및 변경 삭제 데이터처리), Webhook(github, slack 요청/응답), 간단api/BE, IoT에 활용


-(04)Cloud IAM(
-(05)Compute Engine
-(08)GCS(Google Cloud Storage)
-(10)BigQuery
-(11)Cloud Composer
-(23)CloudFunctions



-(04)Cloud IAM(Identity and Access Management)-누가 어떤 리소스에 어떤 엑세스 권한 갖는지 제어
-(05)Compute Engine-가상 머신 서비스(AWS의 EC2), 네트워크에 연결된 가상 서버 제공
-(08)GCS(Google Cloud Storage)-(AWS의 S3) 객체 Repository
-(10)BigQuery-데이터 웨어하우스(탄력확장, 전용query)
-(11)Cloud Composer-워크플로우 관리 서비스(파이프라인 작성, 예약 및 모니터링)
-(23)CloudFunctions-(AWS의 람다) 클라우드 인프라 및 서비스의 이벤트 연결 함수 작성 가능



-(04)Cloud IAM(Identity and Access Management)-누가 어떤 리소스에 어떤 엑세스 권한 갖는지 제어
—G Suite - 조직에 생성된 모든 구글 계정으로 구성된 가상 그룹
—IAM 정책 - 사용자에게 부여되는 액세스 권한 유형 가진 리스트
—각 계정은 서비스 키 쌍과 연결되며 서비스 계정 키는 크게 GCP 관리용(서비스간 인증용)과 사용자 관리용(사용자계정) 2가지로 나뉨
—gcloud 명령어 사용
--policy( [member(accounts)+role(permissions)]s )
--IAM -> API메서드에서 리소스 사용할 수 있는지 권한role 확인
Pub/sub에서도구독자에role부여정책생성가능
--OAUTH 2.0
https://blog.naver.com/mds_datasecurity/222182943542


-(05)Compute Engine-가상 머신 서비스(AWS의 EC2), 네트워크에 연결된 가상 서버 제공
—쿠버네티스 엔진 통해 도커 컨테이너를 실행 및 관리 조정할 수 있음
—하나의 프로젝트는 여러 개의 인스턴스를 가질 수 있음
—실시간 이전(라이브 마이그레이션) 가능(껐다 켤 필요 없이)
—관리형인스턴스그룹은 자동으로 오토 스케일링 지원, 자동복구정책 설정가능, 트래픽 분산가능, 크게 영역/리전 관리 2개로 나뉨
—비관리형은 위가 제공되지 않음
—스냅샷은 인스턴스의 특정 시점에 백업 가능

-(08)GCS(Google Cloud Storage)-(AWS의 S3) 객체 Repository
—프로젝트 > 버킷(전체 gcs중 교유명, repository 등급 나뉨) > 객체(버킷에 저장되는 파일들, 객체/메타로 나뉨, 변경불가, 중복저장(안정성))
—repository 등급은 멀티리저널(자주 어세스), 리저널(분산용), 니어라인(자주x), 콜드라인(가장저렴, 백업용) 4가지

-(10)BigQuery-데이터 웨어하우스(탄력확장, 전용query)
—쿼리로 ml, gis(지리정보시스템) 가능
--컴퓨팅과 스토리지가 분리되어 있어 독립적으로 확장
--쿼리 실행 시 여러 작업자가 작업을 동시 분산하여 결과 수집
--완전 관리형으로 프로비저닝, 스토리지 단위 예약이 불필요
--리전에 데이터 복제, 내구성/가용성 좋음
--암호화 제공
--데이터는 열 기반 테이블 형태로 저장(표준테이블-메타, 구체화된 뷰, 테이블 스냅샷-백업, 외부테이블-데이터저장소에 데이터가 있는?)
--(열기반)외래키 지원  x 이므로  oltp보다 olap 및 데이터 웨어하우스 워크로드에 적합
--(열기반)클라우드 sql로 효율적 사용 가능, 레코드 데이터 분석 워크로드에 최적화
--파티션분할(버킷별로, 시간단위열/정수열/데이터수신시간 3가지 데이터로 테이블 구성)
--프로젝트-데이터셋의 집합, 데이터셋-테이블의 집합, 테이블-데이터들어감, 잡-쿼리/로딩/명령(권한은 프로젝트와 데이터셋단위만 적용/테이블단위는 안됨)
--데잇터셋 권한(리더, 라이터-테이블생성/조회/추가, 오너-데이터셋 업데이트/삭제) 프로젝트 권한(뷰어, 에디터-데이터셋 생성/라이터권한, 오너-데이터셋에 대한 잡 수행)

BigQuery SQL
Ex
-INSERT INTGO box1 VALUES('name', age, 'rank')
-SELECT * FROM 'box1'
-SELECT 이름, 나이 FROM 'box1' WHERE age=33 LIMIT 2
-SELECT 이름, 직급 FROM 'box1' GROUP BY 이름, 직급
-
SELECT
  name, gender,
  SUM(number) AStotal
FROM
  `bigquery-public-data.usa_names.usa_1910_2013`
GROUP BY
  name, gender
ORDER BY
  total DESC
LIMIT
  10


INSERT INTO data_set.box1 VALUES('홍길동', 34, '부장'),
('손오공', 33, '과장'),
('베지터', 29, '차장')



SELECT First_Name, Weight_in_Kgs_ AS Weight FROM `gcloud-hlhl.data_set.100_record` WHERE Gender='M' ORDER BY Weight_in_Kgs_ DESC LIMIT 10



SELECT Emp_ID, User_Name, Password FROM `gcloud-hlhl.data_set.100_record` WHERE Gender='F' and Age_in_Yrs_ >= 27 ORDER BY Emp_ID ASC LIMIT 100




생성, 로드 생성
데이터세트/테이블 생성
-(11)Cloud Composer-워크플로우 관리 서비스(파이프라인 작성, 예약 및 모니터링)
—아파치 에어플로 기반 통합서비스, python기반으로 dag/task 코드 작성 가능
—쿠버네티스(배포환경)+클라우드 sql(메타데이터 저장)+에어플로(웹 서버 호스팅)+Stackdriver(로그관리)
—airflow - python기반 워크플로우 통합도구
—DAG - 방향비순환그래프로 하나의 워크플로우라고 생각하면 됨
—operator - DAG 안의 작업함수로 이를 이용하여 태스크 만듬(bash/python 정도가 있음, 각각 bash/python 실행용)
—저장은 자동으로 gcs서 버킷 만들고 스토리지 FUSE를 사용하여 에어플로 인스턴스와 gcs버킷 매핑함

-(23)CloudFunctions-(AWS의 람다) 클라우드 인프라 및 서비스의 이벤트 연결 함수 작성 가능
—코드가 완전 관리형 환경에서 실행, 프로비저닝/서버관리 필요x
—클라우드 연결/확장에 용이 - gcs파일 업로드,pub/sub 메세지확인 등 gcp 서비스 이용 가능
—cloud이벤트 - 클라우드 환경에서 발생하는 모든 상황
—서버리스 - 서버관리, 소프트웨어구성, 프레임워크 업데이트, 운영체제 패치 등 신경 쓸 필요 x, 사용자코드만 추가o
—배포시 자동으로 프로비저닝함(ETL(파일 생성 및 변경 삭제 데이터처리), Webhook(github, slack 요청/응답), 간단api/BE, IoT에 활용


클라우드 교육
	• VPC 네트워킹 및 GCE
	• Cloud Storage 및 Cloud SQL
	• GKE
	• Cloud Run (APPRUN)
• 서비스 유형 파악
	• 존(zones) 안에 데이터센터, 리전(regions) 안에 존 3개 묶임
	• Region : europe-west2
	• Zone : europe-west2-a, europe-west2-b, europe-west2-c
	• Sla(고가용성, 서비스 수준 협약, ) 때문
		○ SLA는 서비스 수준 협약으로서 일정한 시간동안 어느정도의 수준까지는 동작하는 것을 보장하는 약속이라고 합니다.
	• 버추얼머신(vm)은 존에 단일머신으로 존재
		○ Vm만들면 2개 이상의 존에 분산배치해서 사용하는 것을 권장
		○ 일부 서비스는 여러 지리적 위치에서 실행 가능, 멀티 리전!
	• 리소스 계층화 모델(논리적인 구조 제공)
		○ 우리 조직의 것을 구분할 수 있어야 함ㅈ
		○ 프로젝트가 주요 요소, 구글 클라우드에서 격리 모델로 생성, 보안의 경계, 리소스의 경계를 구성하는 격리 컨테이너로 생각하면 됨
		○ 여러가지 리소스(쿠버나 vm)이 프로젝트에 속하게 됨
		○ 조직 구조에 맞춰서 프로젝트를 만들 수도 있겠죠
		○ 폴더로 프로젝트를 묶어서 프로젝트들을 관리할 수 있음(제어를 위함)
		○ 최상위에는 오거니제이션 노드라는 조직노드가 있음. 폴더들을 포함함
		○ 프로젝트수준, 리소스 수준에서 정책 정하는 것도 가능
		○ 프로젝트가 가장 기본, 서비스 활용의 기반
		○ 개발용, 스테이징용, 제어용 프로젝트 만들어서 사용할 수 있음
		○ 조직 정책 관리자, 프로젝트 생성사 수준으로 제어
		○ 그굴워크스페이스 고객은 자동으로 조직 노드에 속함
	• 온프로미스와 동일하게 구글 클라우드 환경 만들 수 있도록 vpc 제공(구조화된 네트워크 환경, 외부와 격리된 환경)
	• 컴퓨팅엔진 통해 버추얼 머신 배치, 탄력적으로 배치(확대, 축소 - 클라우드 로드벨렌싱)
	• VPC 가상 사설 클라우드, vpc안에 서브넷을 가짐(리전 수준)
	• VPC는 글로벌 리소스, 서비넷은 러전 수준의 리소스
	• ㅇ
	• 선점형은 구글이 선점하기 때문에 중간에 막 멈춤(그래서 90퍼 할인된 가격)
	• 워크로드 유형에 따라 vm머신 (인스턴스)유형 나뉨
	• VPC는 라우터를 프로비저닝할 필요 없음. 한 인스턴스에서 다른 인스턴스로 트래픽 전달
	• VPC 피어링 및 공유(VPC sharing)를 통해 프로젝트 간 통신 가능
	• 로드벨런싱을 통한 인스턴스간 트래픽 분산 처리
	• 여러 서브넷에 분산된. Vm들, 이걸 로드벨런싱(글로벌 수준이라면 글로벌 로드벨런싱)
		○ 리전 수준이라면 리저널 로드벨런싱
		○ 내부만이라면 리저널 인터널 로드벨런싱
	• VPC
		○ Allow-icmp: 외부서 접
		○ Allow-internal : vpc 안에서 ping이건 기본 통신이 됨(내부)
		○ Allow-ssh : ssh 접속
	• 클라우드 스토리지
		○ 객체단위 관리(스키마까지 포함)
		○ 보관 위주에 효과적
		○ 버전관리 기능(이전버전 유지, 롤백가능)
		○ 종류 4가지
		○ 전송 3가지(기본, 헤비용, 정말큰거-appliance)
	• Cloud SQL
		○ 소프트웨어 설치 및 유지 관리 필요 x
		○ 확장 가능 관리형 백업, 자동 복제, 암호화, 방화벽 등 포함
		○ 외부 access 가능, 구글 다른 서비스와 함께 사용 가능
	• Google Kubernetes Engine
		○ 컨테이너는 "pod"라는 그룹에서 실행
		○ Pod 마다 유니크 아이피 가짐
		○ 구글 클라이드의 부하 분산
		○ 클러스터 내 노드의 하위 집합을 지정하기 위한 노드풀(구글 클라우드기능)
		○ 클러스터의 노드 인스턴스 수 자동 조정
		○ 클러스터의 노드 소프트웨어에 대한 자동 업그레이드
		○ 노드 상태 및 가용성을 유지하기 위한 노드자동복구
		○ Google Cloud의 Operations suite을 사용한 로깅 및 모니터링
	• Applications in the Cloud
		○ 앱엔진과 클라우드런으로 적은 관리에 코드 집중 개발 가능
		○ 외부서비스와 통합 활용 가능
		○ 엡엔진은 여러 서비스가 빌트인으로 통합되서 제공
		○ Sdk안에 api라이브러리, 샌드박스 환경, 배포도구도 포함
		○ 서버리스여서 인프라관리 안 해도 되는 장점
		○ 사용 않으면 리소스 0으로 측정(과금 안 이루어짐)



computer engine -> vm 만들기
bespinglobal.com/gcp-start/

Sdk and function test(cloud function)
https://cloud.google.com/functions/docs/create-deploy-http-python?hl=ko


DJANGO REST API 만들기 (참고-> https://jamanbbo.tistory.com/43)

GCP - CloudSQL -> PostgreSQL (버전14)

Django - setting.py에서 설정[local host]

DATABASES={
'default':{
'ENGINE':'django.db.backends.postgresql',
'NAME':'movie',
'USER':'test',
'PASSWORD':'test',
'HOST':'localhost',#35.202.149.157
'PORT':'',
}
}

--접속
#psql postgres
(\connect movie \select * from movies_movie)

'Cloud SQL Admin API'[ssl 사용]

DATABASES={
'default':{
'ENGINE':'django.db.backends.postgresql_psycopg2',
'NAME':'movie',
'USER':'test',
'PASSWORD':'test',
#https://console.cloud.google.com/sql/instances
'HOST':'35.202.149.157',
'PORT':'5432',#atthemomentofthiswritinggooglecloudpostgresqlisusingthedefaultpostgresqlport5432
'OPTIONS':{
'sslmode':'verify-ca',#leavethislineintact
'sslrootcert':'movie_api/server-ca.pem',
"sslcert":"movie_api/client-cert.pem",
"sslkey":"movie_api/client-key.pem",
}
}
}

[ssl 안 사용]
DATABASES={
'default':{
'ENGINE':'django.db.backends.postgresql_psycopg2',
'NAME':'movie',
'USER':'test',
'PASSWORD':'test',
'HOST':'34.72.163.128',
'PORT':'5432',
}

-CloudSQL 생성
-공개 ip 및 네트워크 추가 0.0.0.0/0 ( 참고 : https://yooloo.tistory.com/148 )

-create database movie;
-create role test with login password 'test';

-gcloud에서 확인
# gcloud sql connect movie -d movie --user=test

-쿼리 및 명령어 관련(참고 : https://myinfrabox.tistory.com/234 )


[전반적인 작성 과정]
-Project 생성 ->

-앱 생성

-settings 설정

-model 생성

-serializer 생성
데이터 전달용 dto같은

-view 작성
Viewset으로 crud 기능 제공

-url 작성
Router로 viewset을 router에 연결



-추가를 위한 참고사항
Viewset modify 참고 (-> https://gimkuku0708.tistory.com/43 )



-어드민페이지에서 적용 위한 고민 필요


defchangeform_view(self,request,object_id=None,form_url='',extra_context=None):
print("=============test===========")

print(request.POST['CHART_CD'])

withtransaction.atomic(using=router.db_for_write(self.model)):
returnself._changeform_view(request,object_id,form_url,extra_context)

이 접근 괜춘한듯

POST로 읽힐 데이터 없을 때 대응법 (참고 -> https://cjh5414.github.io/django-keyerror/ )




ㅁ쉐어포인트 연동 --not working.. Selenium?
-파일 올리고 권한 설정
-내 아이디로 접근 및 불러오기
-남는 계정 등록 또는 권한 넘겨주기(파일 접근할 때 인증할 수 있는 방법도 찾아보기)

https://github.com/vgrem/Office365-REST-Python-Client

준비 및 과정
pip install Office365-REST-Python-Client

근데 안 됨..
The error message is pretty intuitive, user credentials auth is not supported when Multi-Factor Authentication (MFA) enabled.
핸드폰 등으로 추가 인증받게 되어있어서 안 되는 것

https://stackoverflow.com/questions/55922791/azure-sharepoint-multi-factor-authentication-with-python


관리계정이 있어야 한다는 ms 답변
https://learn.microsoft.com/en-us/answers/questions/546248/programmatic-use-of-sharepoint-api-and-2factor-aut.html








• Http status 관련
	• Admin 에서 add하면 status code 302 (found) 뜸.. 왜?
		○ 300번대가 redirect
		○ 302 redirect : 의미는 요청한 리소스가 임시적으로 새로운 URL로 이동했음을 나타냄 - 기존 URL이 보유한 페이지랭킹 점수는 그대로 유지하도록 하면서 컨텐트만 새로운 URL에서 조회하도록 해야할 때 사용

# [RESPONSE]
# 200 - 이미 추가가 되어 있는 상태여서 초기화 업데이트를 수행하고 성공 처리
# 201 - 정상적으로 추가 성공
# 400 - 잘못된 요청 메시지
# 401 - UNAUTHORIZED: 호출자의 토큰이 만료되거나, 신원확인이 되지 않는 경우
# 403 - FORBIDDEN: 호출자가 알맞은 호출 권한이 없는 경우
# 500 - INTERNAL SERVER ERROR


