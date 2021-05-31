import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.sql.{Row, SparkSession}

object DemoCorrelation {

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder
      .master("local[*]")
      .appName("CorrelationExample")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    val data = Seq(
      Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),       // 1, 0, 0, -2
      Vectors.dense(4.0, 5.0, 0.0, 3.0),  // 4, 5, 0, 3
      Vectors.dense(6.0, 7.0, 0.0, 8.0),  // 6, 7, 0, 8
      Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))        // 9, 0, 0, 1
    )

    val df = data.map(Tuple1.apply).toDF("features")
    println(df.show(4))

    val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
    println(s"Pearson correlation matrix:\n$coeff1")

    println("\n")

    val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
    println(s"Spearman correlation matrix:\n$coeff2")

    spark.stop()
  }

}
