
import org.clulab.odin._
import org.clulab.processors.{Sentence, Document}
import scala.collection.mutable
import io.circe.generic.auto._
import io.circe.parser._
import io.circe._
import io.circe.syntax._
import scala.io.Source
import org.clulab.reach.PaperReader
import java.io._
import org.clulab.struct.Interval
import java.io.File
import scala.collection.mutable.Buffer
import org.clulab.struct.GraphMapNames
import java.nio.file.{Files, Paths}


case class GeneratedRule(
  rule_name: String,
  rule: String,
  trigger: List[Int],
  controlled: List[Int],
  controller: List[Int],
  base_rule_name: String,
  base_rule: String,
  base_trigger: List[Int],
  base_controlled: List[Int],
  base_controller: List[Int],
  base_type: String,
  tokens: List[String],
  lemmas: List[String],
  tags: List[String],
  outgoing: List[String],
  incoming: List[String],
  marked_tokens: List[String],
  annotated_tokens: List[String],
  base_text: String,
  marked_text: String,
  processed_text: String,
  pred0: String,
  pred1: String
)

case class ArgStatus(
  controlled: Boolean = false,
  controller: Boolean = false,
)

case class Status(
  any: Boolean = false,
  total: Boolean = false,
  trigger: Boolean = false,
  controlled: Boolean = false,
  controller: Boolean = false,
)

object FindMatches extends App {
  var totalRules = 0
  var anyMatchCount = 0
  var totalMatchCount = 0
  var triggerMatchCount = 0
  var controlledMatchCount = 0
  var controllerMatchCount = 0

  def IntervaltoList(interval: Interval): List[Int] = {
    if (interval == Interval.empty) List.empty[Int]
    else (interval.start to interval.end).toList
  }

  def fetchArguments(b: Mention, rule: GeneratedRule): ArgStatus = {
    var argStatus = ArgStatus()
    b.arguments.foreach { case (argName, ms) =>
      ms.foreach { v =>
        val arg = IntervaltoList(v.tokenInterval)
        if (argName.contains("controlled")) {
          if (arg == rule.controlled) {
            argStatus = argStatus.copy(controlled = true)
          }
        } else if (argName.contains("controller")) {
          if (arg == rule.controller) {
            argStatus = argStatus.copy(controller = true)

          }
        }
        println(s"\t$argName => $arg")
      }
    }
    argStatus
  }

  def displayMention(mention: Mention, rule: GeneratedRule, status: Status): Status = {
    val boundary = s"\t${"-" * 30}"
    var newStatus = status
    mention match {
      case em: EventMention =>
        if (em.foundBy == rule.rule_name) {
          val predTrigger = IntervaltoList(em.trigger.tokenInterval)
          newStatus = newStatus.copy(any=true)
          println(s"trigger: ${rule.trigger} | ${rule.base_trigger}, controlled: ${rule.controlled} | ${rule.base_controlled}, controller: ${rule.controller} | ${rule.base_controller}")
          println(s"Rule => ${em.foundBy}")
          println(s"trigger => ${predTrigger}")
          val argStatus = fetchArguments(em, rule)
          if (rule.trigger == predTrigger) {
            newStatus = newStatus.copy(trigger=true)
            println("Trigger Matched")
          }
          if (argStatus.controlled) {
            newStatus = newStatus.copy(controlled=true)
            println("Controlled Matched")
          }
          if (argStatus.controller) {
            newStatus = newStatus.copy(controller=true)
            println("Controller Matched")
          }
          if (rule.trigger == predTrigger && argStatus.controlled && argStatus.controller) {
            newStatus = newStatus.copy(total=true)
            println("Everything Matched")
          }
          println()
          println(s"Sentence: ${rule.marked_text}")
          println()
          println(s"True: ${rule.rule}")
          println()
          println(s"Predicted: ${rule.pred0}")
          println(s"$boundary\n")
        }
      case _ => ()
    }
    newStatus
  }

  val fileName = "id_split/train_data_500_results.json"
  println(s"Initializing Reach")
  val reachSystem = PaperReader.reachSystem
  println(s"Done initializing Reach")

  println(s"Processing file: $fileName")
  val source = Source.fromFile(fileName)
  val jsonString =
    try source.mkString
    finally source.close()
  val boundary = s"\t${"-" * 30}"
  decode[List[GeneratedRule]](jsonString) match {
    case Right(rules) =>
        rules.zipWithIndex.foreach { case (rule, index) =>
          println(s"Processing rule ${rule.rule_name}")
          val doc = reachSystem.mkDoc(
            text = rule.tokens.mkString(" "),
            docId = "Document" // docId doesn't matter for our purposes
          )
          val stringMentions = reachSystem.extractFrom(doc)
          var status = Status()
          stringMentions.foreach { mention =>
            status = displayMention(mention, rule, status)
          }
          totalRules += 1
          if (status.any) {
            anyMatchCount += 1
            if (status.total) totalMatchCount += 1
            if (status.trigger) triggerMatchCount += 1
            if (status.controlled) controlledMatchCount += 1
            if (status.controller) controllerMatchCount += 1
            println()
            println(s"totalRules: ${totalRules}")
            println(s"anyMatchCount: ${anyMatchCount}")
            println(s"triggerMatchCount: ${triggerMatchCount}")
            println(s"controlledMatchCount: ${controlledMatchCount}")
            println(s"controllerMatchCount: ${controllerMatchCount}")
            println(s"totalMatchCount: ${totalMatchCount}")
            println()
          } else {
            println("No match")
            println(s"Text ${rule.marked_text}\n")
            println(s"True rule ${rule.rule}\n")
            println(s"Predicted rule ${rule.pred0}")
            println(s"Trigger ${rule.base_trigger} Controlled ${rule.base_controlled} Controller ${rule.base_controller}\n")
            println(s"Subtrigger ${rule.trigger} Subcontrolled ${rule.controlled} Subcontroller ${rule.controller}\n")
            println(s"$boundary\n")
          }
        }
    case Left(ex) =>
        println(s"Failed to decode JSON from $fileName: ${ex.getMessage}")
  }
}
