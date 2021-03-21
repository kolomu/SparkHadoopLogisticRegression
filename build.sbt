name := "MLDemo"

version := "0.1"

scalaVersion := "2.12.12"

libraryDependencies += "org.apache.spark" % "spark-core_2.12" % "3.1.1" % "provided"
libraryDependencies += "org.apache.spark" % "spark-sql_2.12" % "3.1.1"  % "provided" // for SimpleApp (can be removed later)
libraryDependencies += "org.apache.spark" % "spark-mllib_2.12" % "3.1.1" % "provided"
// https://mvnrepository.com/artifact/org.scala-lang/scala-reflect
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.12.0"


