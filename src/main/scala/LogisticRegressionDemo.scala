import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.functions.{col, countDistinct}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}

object LogisticRegressionDemo {

  def main(args: Array[String]): Unit = {
    // create spark session
    val spark: SparkSession = SparkSession.builder
      // .master("local[*]")
      .master("spark://spark-master:7077")
      .appName("LogisticRegressionExample")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    // load dataset
    // val csvPath = getClass.getResource("/fraud.csv").getPath
    val csvPath  = "hdfs://namenode:9000/data/csv/fraudexample/fraud.csv"
    // /user/root/input
    val df =  spark.read
      .option("header", true) // file contains header columns, we want to refer to them in the select statement
      .option("inferSchema", true) // when setting to true it automatically infers column types based on the data
      .csv(csvPath)
      .select("type", "amount", "oldbalanceOrg", "newbalanceOrig", "isFraud")

    df.printSchema()
    // df.show(2)

    // split the dataset into test and training data
    val result = df.randomSplit(Array(0.7, 0.3), 7)
    val trainDF: DataFrame = result.head  // DataFrame = Dataset[Row]
    val testDF: DataFrame = result.tail.head


    // println(s"Train set length ${trainDF.count()} records")
    // println(s"Test set length ${testDF.count()} records")

    // trainDF.show(2)

    // check the data type
    // we need to check the data type because any type of string is treated as categorical variable
    // df.dtypes: Returns all column names and their data types as an array.
    // https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html
    val catCols = trainDF.dtypes.filter( value =>  value._2 == "StringType")
    val numCols = trainDF.dtypes.filter( value => value._1 != "isFraud" && value._2 == "DoubleType")

    // StringIndexer: Converts a single feature to an index feature. http://spark.apache.org/docs/latest/ml-features#stringindexer
    // OneHotEncoder: http://spark.apache.org/docs/latest/ml-features#onehotencoder
    // For more info: http://spark.apache.org/docs/latest/ml-features

    // how many different types are there?
    // trainDF.select(countDistinct("type")).show()

    // group by types
    // trainDF.groupBy("type").count().show()

    // setting up string indexer
    val string_indexer = new StringIndexer()
      .setInputCol(catCols.head._1)
      .setOutputCol(catCols.head._1 + "_StringIndexer")
      .setHandleInvalid("skip")

    // setting up One-Hot-Encoder
    val one_hot_encoder = new OneHotEncoder()
      .setInputCols(Array(catCols.head._1 + "_StringIndexer"))
      .setOutputCols(Array(catCols.head._1 + "_OneHotEncoder"))

    // vector assembling: combines values of input columns into a single vector
    // https://spark.apache.org/docs/latest/ml-features#vectorassembler
    val assemblerInput = numCols.map( value => value._1)
    assemblerInput :+ catCols.map( value => value._1 + "_OneHotEncoder")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(assemblerInput)
      .setOutputCol("VectorAssembler_features")

    val stages = Array(string_indexer, one_hot_encoder, vectorAssembler)
    val pipeline = new Pipeline().setStages(stages)
    val pipelineModel = pipeline.fit(trainDF)
    val pp_df = pipelineModel.transform(testDF)

    // pp_df.select("type", "amount", "oldbalanceOrg", "newbalanceOrig", "VectorAssembler_features").show()

    val mldata = pp_df.select(
      col("VectorAssembler_features").as("features"),
      col("isFraud").as("label"))

    // mldata.show(5, false)

    println("Starting to train the model")
    val logisticRegressionModel = new LogisticRegression()
      .setTol(0.1) // Fix LBFGS Error
      .fit(mldata)
    println(s"Coefficients: ${logisticRegressionModel.coefficients} Intercept: ${logisticRegressionModel.intercept}")

    val trainingSummary = logisticRegressionModel.binarySummary

    // Calculate Area Under the ROC Curve
    // metric for classification problem
    // 1 = good measure of seperability
    // 0 = worst separability
    // https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
    // https://www.graphpad.com/guides/prism/8/curve-fitting/reg_logistic_roc_curves.htm
    // choose cutoff with high sensitivy
    val roc = trainingSummary.roc
    roc.show()

    println(s"areaUnderROC: ${trainingSummary.areaUnderROC}")

    spark.stop()
  }
}
