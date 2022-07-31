---
layout: post
title:  "Tibero install and link with Python in Ubuntu"
date:   2022-07-31 18:31:19 +0900
categories: study
---




{% highlight ruby %}
짧은 요약 : Tibero install and link with Python in Ubuntu  



{% endhighlight %}



# 설치  

-유저추가(원할 경우에만)  
adduser tibero  
tibero  
enter/enter/enter…  

-환경변수 설정  
cd /home/tibero  
vim .bashrc  

맨 마지막 부분에  

export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64  

export TB_HOME=/home/tibero/Tibero/tibero6  
export TB_SID=tibero  
export LD_LIBRARY_PATH=$TB_HOME/lib:$TB_HOME/client/lib  
export PATH=$PATH:$JAVA_HOME/bin:$TB_HOME/bin:$TB_HOME/client/bin  



-technet에서 unix 64bit용 tar 파일 다운받고 /home/tibero/Tibero 폴더에 옮겨서 압축해제  
tar xvf tibero6-bin-FS07_CS_2005-linux64-199301-opt.tar.gz   

-테크넷에서 tibero6 license 받고 이 license 파일을  /home/tibero/Tibero/tibero6/license에 위치시킴(참고로, 라이센스 신청시 ubuntu의 hostname 잘 써줘야함)  

-/home/tibero/Tibero/tibero6/config에서 gen_tip.sh 파일 실행  
./gen_tip.sh  

-Tibero 서버를 'NOMOUNT 모드'로 기동 및  'sys' 사용자로 접속 및 동작 확인  
tbboot nomount  
tbsql sys/tibero  

-아래 명령어 입력으로, 최초 데이터베이스 생성  
SQL> create database "tibero"   
  user sys identified by tibero   
  maxinstances 8   
  maxdatafiles 100   
  character set MSWIN949   
  national character set UTF16   
  logfile   
    group 1 'log001.log' size 100M,   
    group 2 'log002.log' size 100M,     
    group 3 'log003.log' size 100M   
  maxloggroups 255   
  maxlogmembers 8   
  noarchivelog   
    datafile 'system001.dtf' size 100M autoextend on next 100M maxsize unlimited   
    default temporary tablespace TEMP   
      tempfile 'temp001.dtf' size 100M autoextend on next 100M maxsize unlimited   
      extent management local autoallocate   
    undo tablespace UNDO   
      datafile 'undo001.dtf' size 100M autoextend on next 100M maxsize unlimited   
      extent management local autoallocate;  

-quit으로 종료 후 다시 tbboot로 tibero 가동  






