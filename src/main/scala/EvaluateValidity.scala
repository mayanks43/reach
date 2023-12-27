
import org.clulab.odin._
import io.circe.generic.auto._
import io.circe.parser._
import io.circe._
import io.circe.syntax._
import scala.io.Source
import org.clulab.reach.PaperReader

object EvaluateValidity extends App {
  def addTabsAndReconcatenate(inputString: String, numberOfSpaces: Int): String = {
    val parts = inputString.split("\n")
    val spacePrefix = " " * numberOfSpaces
    val modifiedParts = parts.map(spacePrefix + _)
    modifiedParts.mkString("\n")
  }

  val fileName = "id_split/train_data_500_results.json"
  val source = Source.fromFile(fileName)
  val jsonString =
      try source.mkString
      finally source.close()
  var validCount = 0
  var totalCount = 0
  decode[List[GeneratedRule]](jsonString) match {
    case Right(rules) =>
      rules.zipWithIndex.foreach { case (rule, index) =>
        val rule_yml = s"""
          |rules:
          |- name: ${rule.rule_name}
          |  label: ${rule.base_type}
          |  pattern: |
          |${addTabsAndReconcatenate(rule.pred0, 4)}
        """.stripMargin
        try {
          ExtractorEngine(rule_yml) // Check if a valid Odin rule
          validCount += 1
        } catch {
          case _: Throwable =>
            println(rule_yml)
            println("Failed")
        }
        totalCount += 1
      }
    case Left(ex) =>
      println(s"Failed to decode JSON from $fileName: ${ex.getMessage}")
  }
  println(s"Valid count: ${validCount}, Total count: ${totalCount}")
}

