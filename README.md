# Spark Hadoop Logistic Regression
Example project to use scala, spark, hadoop and mllib for a logistic regression model. 

**Inspirations**
- [Docker Hadoop Spark](https://github.com/Marcel-Jan/docker-hadoop-spark)
- [Docker Spark](https://github.com/big-data-europe)
- [PySpark MLLib Logistic Regression](https://www.youtube.com/watch?v=1a7bB1ZcZ3k)

**Urls**
- [Spark Jobs](http://localhost:4040/executors/)
- [Spark Master](http://localhost:8080/)
- [Browsing HDFS](http://localhost:50070/explorer.html#/)


## Setup Spark
Important make sure to have correct PATH variables and stuff. 

- [Setup Spark](https://medium.com/analytics-vidhya/getting-set-up-with-intellij-git-java-and-apache-spark-c6b6272dc3c0)
- [Setup Spark](https://sparkbyexamples.com/)

I use OpenJDK11 and have the following path variables

__User Variables:__
- JAVA_HOME -> C:\openjdk11
- Path: %JAVA_HOME%\bin

__System Variables:__
- HADOOP_HOME -> F:\dev\spark3-bin-hadoop2.7
- SPARK_HOME -> F:\dev\spark3-bin-hadoop2.7
- Path: %HADOOP_HOME%\bin

To run the project with spark submit and also in IntelliJ I did change the Configuration in IntelliJ with 
"include dependencies with provided scope".

![Setup1](./img/setup1.PNG)


# How to get it working in spark cluster
1. clone this repository `https://github.com/Marcel-Jan/docker-hadoop-spark`
2 start the docker container `docker-compose up`
3. copy the csv to namenode `docker cp fraud.csv namenode:fraud.csv`
4. go to bash shell of namenode `docker exec -it namenode bash`
5. create hdfs directory
```
hdfs dfs -mkdir /data
hdfs dfs -mkdir /data/csv
hdfs dfs -mkdir /data/csv/fraudexample
```
6. copy fraud.csv to HDFS
`hdfs dfs -put fraud.csv /data/csv/fraudexample/fraud.csv`
7.0 build mllib.jar with `sbt package`   
7. copy mllib.jar into spark-master
`docker cp mldemo_2.12-0.1.jar clientId:/mldemo.jar`
8. go to bash shell of spark master `docker exec -it clientId bash`
9. run mldemo in spark master `./spark/bin/spark-submit --class LogisticRegressionDemo --master spark://spark-master:7077 --executor-memory 1G --total-executor-cores 4 ./mldemo.jar 100`