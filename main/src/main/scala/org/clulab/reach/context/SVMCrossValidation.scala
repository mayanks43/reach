package org.clulab.reach.context

import java.io.{File, FileInputStream, ObjectInputStream}

import com.typesafe.config.ConfigFactory
import org.ml4ai.data.classifiers.LinearSVMWrapper
import org.ml4ai.data.utils.{AggregatedRow, Balancer, CodeUtils}

import scala.io.Source

object SVMCrossValidation extends App {

  val config = ConfigFactory.load()
  val configPath = config.getString("contextEngine.params.untrainedSVMPath")
  val svmWrapper = new LinearSVMWrapper(null)
  val unTrainedSVMInstance = svmWrapper.loadFrom(configPath)
  val labelFile = config.getString("svmContext.labelFile")
  val typeOfPaper = config.getString("svmContext.paperType")
  val outPaperDirPath = config.getString("svmContext.contextOutputDir").concat(s"${typeOfPaper}/")
  val fileListUnfiltered = new File(outPaperDirPath)
  val directories = fileListUnfiltered.listFiles().filter(_.isDirectory)
  val rowsSup = collection.mutable.ArrayBuffer[AggregatedRow]()
  val idMap = collection.mutable.HashMap[(String,String,String),AggregatedRow]()
  val metricsMapPerPaper = collection.mutable.HashMap[String,(Double, Double, Double)]()

  for(d<-directories) {
    val rowFiles = d.listFiles().filter(_.getName.contains("Aggregated"))

    val rows = rowFiles.map(file => {
      val pmcid = file.getName.split("_")(1)
      val evtID = file.getName.split("_")(2)
      val ctxID = file.getName.split("_")(3)
      val ctxID2 = ctxID.slice(0,ctxID.length-4)
      val filePath = outPaperDirPath.concat(pmcid).concat(s"/${file.getName}")
      val row = readAggRowFromFile(filePath)
      val tuple = (pmcid,evtID,ctxID2)
      val mapEntry = Map(tuple -> row)
      idMap ++= mapEntry
      row
    })
    rowsSup ++= rows
  }



  val groupedByPaperID = rowsSup.groupBy(row => s"PMC${row.PMCID.split("_")(0)}")


  val folds = collection.mutable.ArrayBuffer[(Seq[AggregatedRow], Seq[AggregatedRow])]()




  groupedByPaperID.keySet.map(s=> {
    val trainingKeys = groupedByPaperID.keySet.filter(_ != s)
    val trainingRows = trainingKeys.map(key => {
      groupedByPaperID(key)
    })
    val testingRows = groupedByPaperID(s)
    val perFold = (testingRows, trainingRows.flatten.toSeq)
    folds += perFold

  })



  val giantTruthLabel = collection.mutable.ListBuffer[Int]()
  val giantPredictedLabel = collection.mutable.ListBuffer[Int]()

  for((test,train) <- folds) {
    val trainingLabelsIds = collection.mutable.ListBuffer[(String,String,String)]()
    train.map(row => {
      val pmcid = row.PMCID.split("_")(0)
      val pmcidReformat = s"PMC${pmcid}"
      val evtCtxPerPaper = idMap.keySet.filter(_._1 == pmcidReformat)
      trainingLabelsIds ++= evtCtxPerPaper
    })


    val intersectingLabels = trainingLabelsIds.toSet.intersect(generateLabelMap(labelFile).keySet)
    val trainingRows = collection.mutable.ListBuffer[AggregatedRow]()
    val trainingLabels = collection.mutable.ListBuffer[Int]()
    for(idTup <- intersectingLabels) {
      val row = idMap(idTup)
      val label = generateLabelMap(labelFile)(idTup)
      trainingRows += row
      trainingLabels += label
    }
    //val balancedTrainingData = Balancer.balanceByPaperAgg(trainingRows, 1)
    //val (trainingRVFDataset, _) = unTrainedSVMInstance.dataConverter(balancedTrainingData,Some(trainingLabels.toArray))
    val (trainingRVFDataset, _) = unTrainedSVMInstance.dataConverter(trainingRows,Some(trainingLabels.toArray))

    unTrainedSVMInstance.fit(trainingRVFDataset)
    giantTruthLabel ++= trainingLabels
    val predictedLabels = unTrainedSVMInstance.predict(test)
    giantPredictedLabel ++= predictedLabels


    // calculating precision per paper as test case.
    val testingLabelsIDs = collection.mutable.ListBuffer[(String,String,String)]()
    test.map(row => {
      val pmcid = row.PMCID.split("_")(0)
      val pmcidReformat = s"PMC${pmcid}"
      val evtCtxPerPaper = idMap.keySet.filter(_._1 == pmcidReformat)
      testingLabelsIDs ++= evtCtxPerPaper
    })
    val intersectingTestingLabels = testingLabelsIDs.toSet.intersect(generateLabelMap(labelFile).keySet)
    val testingRows = collection.mutable.ListBuffer[AggregatedRow]()
    val testingLabels = collection.mutable.ListBuffer[Int]()
    for(idTup <- intersectingTestingLabels) {
      val row = idMap(idTup)
      val label = generateLabelMap(labelFile)(idTup)
      testingRows += row
      testingLabels += label
    }

    val testPaperPMCID = test(0).PMCID
    val testIDReformat = s"PMC${testPaperPMCID.split("_")(0)}"
    val metricsPerTestCase = findMetrics(testingLabels.toArray, predictedLabels)
    val metricsScorePerPaperID = Map(testIDReformat -> metricsPerTestCase)
    metricsMapPerPaper ++= metricsScorePerPaperID

  }

  val metrics = findMetrics(giantTruthLabel.toArray, giantPredictedLabel.toArray)

  val countsTest = CodeUtils.predictCounts(giantTruthLabel.toArray, giantPredictedLabel.toArray)

  println(s"Precision: ${metrics._1.toString.take(5)}")
  println(s"Recall: ${metrics._2}")
  println(s"Accuracy: ${metrics._3.toString.take(5)}")
  println(countsTest("TP") + " : TP count")
  println(countsTest("FP") + " : FP count")
  println(countsTest("TN") + " : TN count")
  println(countsTest("FN") + " : FN count")


  println("Total sample count: " + metricsMapPerPaper.size)
  for((paperID, metrics) <- metricsMapPerPaper) {
    println("Current Paper ID: " + paperID + " \t Precision: " + metrics._1.toString.take(5) + " \t Recall: " + metrics._2 + "\t Accuracy: " + metrics._3.toString.take(5))
  }

  val precisionOverAllPapers = collection.mutable.ListBuffer[Double]()
    metricsMapPerPaper foreach (x => precisionOverAllPapers += x._2._1)
  val precAggrMetrics = findAggrMetrics(precisionOverAllPapers)
  /*println(s"Min precision over 14 papers: ${precAggrMetrics._1}")
  println(s"Max precision over 14 papers: ${precAggrMetrics._2}")*/
  println(s"Avg precision over 14 papers: ${precAggrMetrics._3.toString.take(5)}")

  val recallOverAllPapers = collection.mutable.ListBuffer[Double]()
    metricsMapPerPaper foreach (x => recallOverAllPapers += x._2._2)
  val recAggrMetrics = findAggrMetrics(recallOverAllPapers)
  /*println(s"Min recall over 14 papers: ${recAggrMetrics._1}")
  println(s"Max recall over 14 papers: ${recAggrMetrics._2}")*/
  println(s"Avg recall over 14 papers: ${recAggrMetrics._3.toString.take(5)}")

  val accuracyOverAllPapers = collection.mutable.ListBuffer[Double]()
    metricsMapPerPaper foreach (x => accuracyOverAllPapers += x._2._3)
  val accuracyAggrMetrics = findAggrMetrics(accuracyOverAllPapers)
  /*println(s"Min accuracy over 14 papers: ${accuracyAggrMetrics._1}")
  println(s"Max accuracy over 14 papers: ${accuracyAggrMetrics._2}")*/
  println(s"Avg accuracy over 14 papers: ${accuracyAggrMetrics._3.toString.take(5)}")


  def findMetrics(truth:Array[Int], test:Array[Int]):(Double,Double,Double) = {
    val countsTest = CodeUtils.predictCounts(truth, test)
    val precision = CodeUtils.precision(countsTest)
    val recall = CodeUtils.recall(countsTest)
    val accuracy = accuracyDup(countsTest)
    (precision,recall,accuracy)
  }

  def readAggRowFromFile(file: String):AggregatedRow = {
    val is = new ObjectInputStream(new FileInputStream(file))
    val c = is.readObject().asInstanceOf[AggregatedRow]
    is.close()
    c
  }


  def generateLabelMap(fileName: String): Map[(String,String,String), Int] = {
    val map = collection.mutable.HashMap[(String,String,String), Int]()
    val source = Source.fromFile(fileName)
    val lines = source.getLines()
    val content = lines.drop(1)
    for(c <- content) {
      val array = c.split(",")
      val pmcid = array(0)
      val evtID = array(1)
      val ctxID = array(2)
      val label = Integer.parseInt(array(3))
      val tup = (pmcid,evtID,ctxID)
      map ++= Map(tup -> label)
    }

    map.toMap
  }


  // change this to CodeUtils.accuracy when Enrique finishes merging
  def accuracyDup(preds:Map[String, Int]): Double = {
    if (!((preds("TP") + preds("FP") + preds("FN") + preds("TN")) == 0)) (preds("TP") + preds("TN")).toDouble/(preds("TP") + preds("TN") + preds("FP") + preds("FN")).toDouble
    else 0.0
  }


  def findAggrMetrics(seq:Seq[Double]): (Double,Double,Double) = {
    val min = seq.foldLeft(Double.MaxValue)(Math.min(_,_))
    val max = seq.foldLeft(Double.MinValue)(Math.max(_,_))
    val sum = seq.foldLeft(0.0)(_+_)
    val avg = sum.toDouble/seq.size.toDouble
    (min,max,avg)
  }
}
