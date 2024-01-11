// For direct approach

import org.clulab.odin._
import io.circe.generic.auto._
import io.circe.parser._
import io.circe._
import io.circe.syntax._
import scala.io.Source
import org.clulab.reach.PaperReader

case class miniRule(
    text: String,
    rule: String
)

object CheckValid extends App {
  def addTabsAndReconcatenate(inputString: String, numberOfSpaces: Int): String = {
    val parts = inputString.split("\n")
    val spacePrefix = " " * numberOfSpaces
    val modifiedParts = parts.map(spacePrefix + _)
    modifiedParts.mkString("\n")
  }

  val fileName = "direct_learning/data/val_data_results.json"
  val source = Source.fromFile(fileName)
  val jsonString =
      try source.mkString
      finally source.close()
  var validCount = 0
  var totalCount = 0
  decode[List[miniRule]](jsonString) match {
    case Right(rules) =>
      rules.zipWithIndex.foreach { case (rule, index) =>
        val rule_yml = s"""
          |rules:
          |- name: Sample_rule
          |  label: Negative_activation
          |  pattern: |
          |${addTabsAndReconcatenate(rule.rule, 4)}
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


