---
layout: post
title:  "Elastic Search ELK"
date:   2023-07-23 21:10:01 +0900
categories: study
---






{% highlight ruby %}


짧은 요약(Abstract) :    
* GCP를 instance에 Elaskticsearch 3총사 Elasticsearch, Logstash, Kibana(ELK)를 설치하고 GCP Cloud SQL의 postgreSQL과 연동한 task를 공유  

[참고문헌: https://sas-study.tistory.com/492]  

{% endhighlight %}  

<br/>

# 사전작업(GCP computeengine, Cloud SQL postgreSQL)  
## (GCP)Compute Engine -> VM 인스턴스 -> 인스턴스 만들기 -> 이름/리즌 선택 -> 머신구성(세개나 돌리기 때문에 커야함.. E2-medium정도로 안되고 E2-standard2추천)  
-> 부팅디스크 CentOS(7)로 변경(참고문헌을 따름)  -> 방화벽 HTTP 트래픽 허용, HTTPS 트래픽 허용에 체크해줌  

## (GCP)SQL -> 인스턴스만들기 -> PostgreSQL(14택함) -> 인스턴스id/pw 설정   

## (터미널로 SSH 접속을 위한 준비, SSH키 만들기) ssh-keygen -t rsa -f ~/.ssh/<key file name> -C <id> -b 2048  
-> cat로 생성된 key file.pub 출력 -> 프로젝트 메타데이터에 SSH키 추가( (GCP)메타데이터->수정->SSH키 항목 추가->저장  
-> ssh -i ~/.ssh/<key file name>@<vm instance IP>  

## (방화벽 설정) (GCP)VPC 네트워크 -> 방화벽 -> 방화벽 규칙 만들기 -> 이름/설명 설정 -> 설정한 이름을 '대상 태그 *' 에 타이핑해서 채워줌 -> '소스 IPv4 범위' 에는  0.0.0.0/0 타이핑해서 채워줌 -> 프로토콜 및 포트 모두 허용  
-> (GCP)VM 인스턴스 -> 수정 -> 네트워크 인터페이스 -> '네트워크 태그'에 앞서 설정한 이름을 타이핑해서 채워줌  

# Elasticsearch 설치  
sudo yum install wget -> https://www.elastic.co/kr/downloads/past-releases#elasticsearch 들어가서 원하는 버전의 Elasticsearch 다운로드 링크 복사  
* 여기서 주의할 점은 Elasticsearch와 Kibana, Logstash의 버전이 같아야한다! 다르면 안 됨!  
-> wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.11.1-linux-x86_64.tar.gz (해당되는 링크)
->tar xfz elasticsearch-7.11.1-linux-x86_64.tar.gz  
->./elasticsearch-7.11.1/bin/elasticsearch  
-> localhost:9200 으로 실행 확인  

## 외부ip에서 접속 가능하게 설정(development->production mode, Bootstrap Checks 통과하게)  
->network.host: 10.178.0.9 (주석 제거하고 본인의 VM의 내부 ip 적어줌)  
* Max File Descriptor 옵션 값 증가  
-> /etc/security/limits.conf 에 가서 <본인 id> - nofile 65535 적어줌  
* vm.max_map-count 옵션 값 증가  
-> /etc/sysctl.conf 에 vm.max_map_count=262144 적어줌  
* discovery 설정  
-> elasticsearch-7.11.1/config/elasticsearch.yml 파일에 아래 항목 적어줌  
cluster.name: "movie-elastic"
node.name: "movie-elastic-node-1"
network.host: ["10.178.0.9"]
discovery.seed_hosts: ["movie-elastic-1"]
cluster.initial_master_nodes: ["movie-elastic-node-1"]
-> sudo shutdown -r 으로 끄고 다시 켜서 curl 34.64.146.91:9200 (본인의 instance 외부 ip)로 실행 확인  

# Kibana 설치  
* elastic search와 마찮가지로 https://www.elastic.co/downloads/kibana에 들어가서 elasticsearch와 동일한 버전의 Kibana 다운로드 링크 복사   
-> wget <링크>로 설치  
->tar -xvf <압축파일명>으로 압축 품  
->kibana_installation_directory/config/kibana.yml에 'elasticsearch.hosts: ["http://34.64.185.118:9200"]' 적어놓음, server.host: "0.0.0.0"도 적어놓음  
-> .kibana-7.17.x-linux-x86_64/bin/kibana 로 kibana 실행(경로문제인지.. 직접 해당 디렉토리 들어가서 ./kibana 해줘야 에러가 안 남.....)    
-> 34.64.146.91:5601 로 접속


# Logstash 설치  
* 위와 같이 https://www.elastic.co/downloads/logstash에서 동일한 버전의 Logstash 설치  
* https://jdbc.postgresql.org/download.html에서 jdbc파일 다운받아서 logstash-core/lib/jars로 옮김  
* 아래처럼 logstash.conf 파일 작성(위치: logstash/config/)  
** 주의할 점은 jdbc driver library path를 정확하게 써야함, 예를 들어 /home/username/logstash-7.17.11/logstash-core/lib/jars/driver.jar 이런식으로 풀로  
** 인덱스 아이디는 일단 임의로 주고, Kibana에서 만들 수 있음(임의로 준 인덱스 아이디와 똑같게)  
input {
    jdbc {
        jdbc_connection_string => "jdbc:postgresql://<host>:<port>/<database>"
        jdbc_user => "<username>"
        jdbc_password => "<password>"
        jdbc_driver_library => "/path/to/your/postgresql/jdbc/driver.jar"
        jdbc_driver_class => "org.postgresql.Driver"
        statement => "SELECT * FROM <your_table>"
    }
}

output {
    elasticsearch {
        hosts => ["http://<elasticsearch_host>:9200"]
        index => "<index_name>"
    }
}

** 참고로 아래 logstash 실행할 때도 path 잘 맞춰서 명령어 쓰시오   
** ./bin/logstash -f /path/to/your/logstash.conf 로 logstash 실행(postgreSQL 데이터 가져옴)  

# Elasticsearch api 호출

curl -X GET "http://34.64.212.49:9200/keywords/_search" -H 'Content-Type: application/json' -d'
{
  "query": {
    "match": {
      "comment": "반전 영화"
    }
  },
  "_source": ["title",  "keywords"]
}'

<br/>

curl -X GET "http://34.64.212.49:9200/infos/_search" -H 'Content-Type: application/json' -d'  
{    
  "query": {  
    "match": {  
      "comment": "반전 영화"  
    }  
  },  
  "_source": ["title",  "keywords", "movie_code", "img_url"]  
}'  

<br/>

### Appendix  
# 사용 근거 및 이유, 관련 프로젝트  
키워드기반 영화추천을 하기 위해서 우선 검색모델을 이용하여 사용자의 입력이 리뷰에 포함된 영화를 찾아줍니다.   
검색모델로 DPR 모델들을 사용해줄 수도 있겠지만 이 경우 컴퓨팅파워가 많이 필요하고 추론 속도가 느리다는 문제점이 있습니다.  
그래서 검색 정확도는 비슷하면서 속도가 빠른 Elasticsearch 프레임워크를 사용하였습니다.  
Elasticsearch의 기본 검색모델은 BM25 기반이지만 프레임워크 차원에서 최적화가 되어있어 속도와 정확도가 좀 더 향상되어 있습니다.  
또한 Elasticsearch를 포함하는 Elastic Stack에는 Logstash와 Kibana가 있는데 이것들을 사용하면 어떤 DB와도 연동이 가능하고 지속적인 DB모니터링과 연동을 가능하게 해줍니다.  
<br/>
다음으로 추출된 영화들의 리뷰에서 키워드 추출 모델을 사용하여 핵심 키워드를 뽑아줍니다.    
단순하게 빈도수를 기반으로 핵심키워드를 추출할 경우 특정 영화의 핵심어가 아닌 경우를 핵심키워드로 뽑아줄 수 있습니다.  
예를 들어, 모든 영화에서 공통적으로 쓰이는 키워드인 “영화”를 핵심 키워드로 뽑아주는 것이죠.  
그래서 이러한 문제를 해결하기 위해 TF-IDF모델과 BERTOPIC 모델을 사용해줍니다.  
해당 모델들은 역빈도수를 반영하기 때문에 모든 곳에서 빈도가 높은 키워드는 제외시켜 줍니다.  
이를 통해 좀 더 해당 영화에 특화된 키워드를 용이하게 추출할 수 있습니다.    
<br/>
마지막으로 추출된 핵심 키워드와 사용자의 입력 사이의 유사도를 기반으로 랭킹을 매겨서 추천해줍니다.  
코사인 유사도로 단순하게 단어 하나하나 비교하는 방법도 있지만 이 경우 연산의 양이 많아지고 추론 속도가 느려지는 문제가 다시 발생합니다.  
그래서 문제점의 해결책으로 문장임베딩을 사용하여 사용자의 입력과 영화의 핵심 키워드 묶음을 직접 한번만 비교해주고 유사도를 찾습니다.  
<br/>  
