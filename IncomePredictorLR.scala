import org.apache.spark.sql.types.{StructType,StringType}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{FloatType,DateType,IntegerType,LongType}
import java.sql.Timestamp
import org.apache.spark.sql.SaveMode
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.storage.StorageLevel
import java.sql.Timestamp
import org.apache.spark.sql.functions.udf
import org.apache.spark.storage.StorageLevel
import sqlContext.implicits._
import sys.process._
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.classification.{SVMModel, SVMWithSGD}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.feature.MinMaxScaler
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.ml.regression.LinearRegression


object LRIncomePredictor{

    def ageGrouper(input:Integer):Integer={
        var group = 0
        var counter = 22
        if(input <= counter){
            return group
        }

        while(counter <= input){
            if(85<=counter){
                return group+1
            }
            counter = counter+7
            group = group +1
        }
        return group
    }

    val ageGrouperUDF = udf((i:Integer) => ageGrouper(i))

    def prediction_10accuracy(predict_income:Double,label_income:Double):Integer = {
            val income = label_income+1
            val ratio = scala.math.abs(predict_income/income-1)
            if(ratio<0.4) 1 else 0
    }

    def IncomePredictor(source1: String, source2 : String):Integer={
        val income_data1 =  spark.read
                         .format("com.databricks.spark.csv")
                         .option("header", "true")
                         .option("inferSchema", "true")
                         .load("file:///notebook/IncomeData-jobinfo.csv")
        val income_data2 =  spark.read
                         .format("com.databricks.spark.csv")
                         .option("header", "true")
                         .option("inferSchema", "true")
                         .load("file:///notebook/ESI_2016_PERSONAS_streamlined.csv")

        val region_data =  spark.read
                         .format("com.databricks.spark.csv")
                         .option("header", "true")
                         .option("inferSchema", "true")
                         .option("encoding","ISO-8859-1")
                         .load("file:///notebook/region_data.csv")

        val income_data = income_data1.select(col("REGION").cast(IntegerType),
                             col("ING_MON_SB").cast(IntegerType),
                             col("Sex (1=man, 2=woman)").alias("Sex").cast(IntegerType),
                             col("Age").cast(IntegerType),
                             col("Education").cast(IntegerType),
                             col("relationship").cast(IntegerType),
                             col("PROVEEDOR").cast(IntegerType))

        val income_dataSource2 = income_data2.select(col("REGION").cast(IntegerType),
                             col("ING_MON_SB").cast(IntegerType),
                             col("Sex (1=man, 2=woman)").alias("Sex").cast(IntegerType),
                             col("Age").cast(IntegerType),
                             col("HABITUALES").cast(IntegerType),
                             col("relationship").cast(IntegerType),
                             col("EFECTIVAS").cast(IntegerType))

        // Split data into training (70%) and test (30%).
        val splits = income_data.randomSplit(Array(0.7, 0.3))
        val training = splits(0)
        val test = splits(1)
        // Split data into training (70%) and test (30%).
        val splits = income_dataSource2.randomSplit(Array(0.7, 0.3))
        val training2 = splits(0)
        val test2 = splits(1)
        val prediction_accuracy = udf((d1:Double,d2:Double)=>prediction_10accuracy(d1,d2))

        //data filtering
        val trainingfilter= training.filter("ING_MON_SB != 0")
        val testfilter= test.filter("ING_MON_SB != 0")

        val lr = new LinearRegression()
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8)

        val assembler = new VectorAssembler()
            .setInputCols(Array("REGION", "Sex","Age","relationship","HABITUALES"))
            .setOutputCol("features")
        val income_data_vector = assembler.transform(training2)
        val test_data_vector = assembler.transform(test2)

        income_data_vector.show()

        val lr = new LinearRegression()
            .setMaxIter(10)
            .setRegParam(0.3)
            .setElasticNetParam(0.8)



        val assembler = new VectorAssembler()
            .setInputCols(Array("REGION", "Sex","Age","Education","PROVEEDOR","relationship"))
            .setOutputCol("features")
        val income_data_vector = assembler.transform(trainingfilter)
        val test_data_vector = assembler.transform(testfilter)

        val vector_data = income_data_vector.select(col("features"),col("ING_MON_SB").alias("label"))
        val test_data  = test_data_vector.select(col("features"),col("ING_MON_SB").alias("label"))
        // Fit the model
        val lrModel = lr.fit(vector_data)
        // Print the coefficients and intercept for linear regression
        println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

        val predictions = lrModel.transform(test_data)
        predictions.count()
        val prediction10 = predictions.select(col("prediction"),
                        col("features"),
                        col("label"))
                        .withColumn("accurate",prediction_accuracy($"prediction",$"label"))

        prediction10.select(col("prediction"),
                        col("features"),
                        col("accurate"),
                        col("label")).where(col("accurate")!==1).show()

        val result = lrModel.evaluate(test_data)

    }



}
