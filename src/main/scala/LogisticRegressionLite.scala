import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions.{col, countDistinct}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}

object LogisticRegressionLite {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder
      .master("spark://spark-master:7077")
      .appName("LogisticRegressionExample")
      .getOrCreate()

    spark.sparkContext.setLogLevel("OFF")

    val csvPath  = "hdfs://namenode:9000/data/csv/fraudexample/fraud.csv"
    val df =  spark.read
      .option("header", true) // file contains header columns, we want to refer to them in the select statement
      .option("inferSchema", true) // when setting to true it automatically infers column types based on the data
      .csv(csvPath)
      .select("type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud")

    val result = df.randomSplit(Array(0.7, 0.3), 7)
    val trainDF: DataFrame = result.head
    val testDF: DataFrame = result.tail.head

    val catCols = trainDF.dtypes.filter( value =>  value._2 == "StringType")
    val numCols = trainDF.dtypes.filter( value => value._1 != "isFraud" && value._2 == "DoubleType")

    val string_indexer = new StringIndexer()
      .setInputCol(catCols.head._1)
      .setOutputCol(catCols.head._1 + "_StringIndexer")
      .setHandleInvalid("skip")

    val one_hot_encoder = new OneHotEncoder()
      .setInputCols(Array(catCols.head._1 + "_StringIndexer"))
      .setOutputCols(Array(catCols.head._1 + "_OneHotEncoder"))

    val assemblerInput = numCols.map( value => value._1)
    assemblerInput :+ catCols.map( value => value._1 + "_OneHotEncoder")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(assemblerInput)
      .setOutputCol("VectorAssembler_features")

    val stages = Array(string_indexer, one_hot_encoder, vectorAssembler)
    val pipeline = new Pipeline().setStages(stages)
    val pipelineModel = pipeline.fit(trainDF)
    val pp_df = pipelineModel.transform(testDF)

    val mldata = pp_df.select(
      col("VectorAssembler_features").as("features"),
      col("isFraud").as("label"))

    println("training the model")
    val logisticRegressionModel = new LogisticRegression()
      .fit(mldata)
    println(s"Coefficients: ${logisticRegressionModel.coefficients} Intercept: ${logisticRegressionModel.intercept}")

    val trainingSummary = logisticRegressionModel.binarySummary

    val roc = trainingSummary.roc
    roc.show()

    println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

    spark.stop()
  }
}
