
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
  tokens: List[String],
  pred0: String,
  pred1: String,
  rule: String,
  trigger: List[Int],
  controlled: List[Int],
  controller: List[Int],
  subtrigger: List[Int],
  subcontrolled: List[Int],
  subcontroller: List[Int],
  base_rule_name: String,
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
          if (arg == rule.subcontrolled) {
            argStatus = argStatus.copy(controlled = true)
          }
        } else if (argName.contains("controller")) {
          if (arg == rule.subcontroller) {
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
          println(s"trigger: ${rule.trigger} | ${rule.subtrigger}, controlled: ${rule.controlled} | ${rule.subcontrolled}, controller: ${rule.controller} | ${rule.subcontroller}")
          println(s"Rule => ${em.foundBy}")
          println(s"trigger => ${predTrigger}")
          val argStatus = fetchArguments(em, rule)
          if (rule.subtrigger == predTrigger) {
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
          if (rule.subtrigger == predTrigger && argStatus.controlled && argStatus.controller) {
            newStatus = newStatus.copy(total=true)
            println("Everything Matched")
          }
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

  val fileName = "results.json"
  println(s"Initializing Reach")
  val reachSystem = PaperReader.reachSystem
  println(s"Done initializing Reach")

  println(s"Processing file: $fileName")
  val source = Source.fromFile(fileName)
  val jsonString =
    try source.mkString
    finally source.close()

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
            println(s"totalMatchCount: ${totalMatchCount}")
            println(s"triggerMatchCount: ${triggerMatchCount}")
            println(s"controlledMatchCount: ${controlledMatchCount}")
            println(s"controllerMatchCount: ${controllerMatchCount}")
            println()
          }
        }
    case Left(ex) =>
        println(s"Failed to decode JSON from $fileName: ${ex.getMessage}")
  }
}
