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

case class BioRule(
    rule_name: String,
    rule: String,
    sentence_tokens: List[String],
    `type`: String,
    trigger_indices: List[Int],
    controlled_indices: List[Int],
    controller_indices: List[Int]
)

case class ModifiedBioRule(
    rule_name: String,
    rule: String,
    sentence_tokens: List[String],
    `type`: String,
    trigger_indices: List[Int],
    controlled_indices: List[Int],
    controller_indices: List[Int],
    lemmas: Buffer[String] = Buffer.empty[String],
    tags: Buffer[String] = Buffer.empty[String],
    outgoing: Buffer[String] = Buffer.empty[String],
    incoming: Buffer[String] = Buffer.empty[String]
)

case class SubRule(
    rule_name: String,
    trigger: List[Int] = List.empty[Int],
    controlled: List[Int] = List.empty[Int],
    controller: List[Int] = List.empty[Int]
)

case class ExplodedRule(
    rule: ModifiedBioRule,
    subrules: Seq[SubRule]
)

object Filter extends App {
  def IntervaltoList(interval: Interval): List[Int] = {
    if (interval == Interval.empty) List.empty[Int]
    else (interval.start to interval.end).toList
  }

  def getSubRule(
      mention: Mention,
      trigger_indices: List[Int],
      controlled_indices: List[Int],
      controller_indices: List[Int]
  ): Option[SubRule] = {
    mention match {
      case em: EventMention
          if IntervaltoList(em.trigger.tokenInterval) == trigger_indices =>
        val (controlledIndices, controllerIndices) =
          em.arguments.foldLeft((List.empty[Int], List.empty[Int])) {
            case ((controlled, controller), (argName, ms)) if ms.nonEmpty =>
              val headInterval = ms.head.tokenInterval
              argName match {
                case "controlled"
                    if IntervaltoList(headInterval) == controlled_indices =>
                  (controlled ++ headInterval, controller)
                case "controller"
                    if IntervaltoList(headInterval) == controller_indices =>
                  (controlled, controller ++ headInterval)
                case _ => (controlled, controller)
              }
            case (acc, _) => acc
          }

        if (controlledIndices.nonEmpty && controllerIndices.nonEmpty) {
          Some(
            SubRule(
              rule_name = em.foundBy,
              trigger = IntervaltoList(em.trigger.tokenInterval),
              controlled = controlledIndices,
              controller = controllerIndices
            )
          )
        } else {
          None
        }

      case _ => None
    }
  }

  def getListOfFiles(dir: String): List[File] = {
    val folder = new File(dir)
    if (folder.exists && folder.isDirectory) {
      folder.listFiles.toList
    } else {
      List[File]()
    }
  }

  def getsentenceDetails(sentence: Sentence, rule: BioRule): ModifiedBioRule = {
    val (words, lemmas, tags, entities, deps) = (
      sentence.words,
      sentence.lemmas.get,
      sentence.tags.get,
      sentence.entities.get,
      sentence.graphs.get("universal-enhanced")
    )
    val new_rule = new ModifiedBioRule(
      rule_name = rule.rule_name,
      rule = rule.rule,
      sentence_tokens = rule.sentence_tokens,
      `type` = rule.`type`,
      trigger_indices = rule.trigger_indices,
      controlled_indices = rule.controlled_indices,
      controller_indices = rule.controller_indices
    )
    for (i <- words.indices) {
      new_rule.tags += tags(i)
      new_rule.lemmas += lemmas(i)
      val incoming = new StringBuilder
      val outgoing = new StringBuilder
      deps match {
        case Some(x) => {
          for (e <- x.getIncomingEdges(i)) {
            incoming.append(" " + e)
          }
          for (e <- x.getOutgoingEdges(i)) {
            outgoing.append(" " + e)
          }
          new_rule.incoming += incoming.toString
          new_rule.outgoing += outgoing.toString
        }
        case None => {
          new_rule.incoming += ""
          new_rule.outgoing += ""
        }
      }
    }
    new_rule
  }

  val processedRuleName = ""
  val fileNames = getListOfFiles("rules")
  println(s"Initializing Reach")
  val reachSystem = PaperReader.reachSystem
  println(s"Done initializing Reach")
  val explodedRules: Buffer[ExplodedRule] = Buffer.empty[ExplodedRule]

  fileNames.foreach { fileName =>
    if (fileName.getName.endsWith(".json")) {
      println(s"Processing file: $fileName")
      val source = Source.fromFile(fileName)
      val jsonString =
        try source.mkString
        finally source.close()

      decode[List[BioRule]](jsonString) match {
        case Right(rules) =>
          rules.zipWithIndex.foreach { case (rule, index) =>
            println(s"Processing rule number $index")
            if (
              rule.rule_name == processedRuleName && rule.rule != "MISSING VAL"
            ) {
              val doc = reachSystem.mkDoc(
                text = rule.sentence_tokens.mkString(" "),
                docId = "Document" // docId doesn't matter for our purposes
              )
              val new_rule = getsentenceDetails(doc.sentences.head, rule)
              val stringMentions = reachSystem.extractFrom(doc)
              if (stringMentions.nonEmpty) {
                val subRules = stringMentions.flatMap(
                  (
                      mention =>
                        getSubRule(
                          mention = mention,
                          trigger_indices = new_rule.trigger_indices,
                          controlled_indices = new_rule.controlled_indices,
                          controller_indices = new_rule.controller_indices
                        )
                  )
                )
                if (subRules.nonEmpty) {
                  explodedRules += ExplodedRule(new_rule, subRules)
                }
              }
            }
          }
        case Left(ex) =>
          println(s"Failed to decode JSON from $fileName: ${ex.getMessage}")
      }
    }
  }

  val directoryPath = Paths.get("training_data/")
  if (!Files.exists(directoryPath)) {
      Files.createDirectories(directoryPath)
  }

  val file = new File("training_data/" + processedRuleName + ".json")
  val writer = new PrintWriter(file)
  val json = explodedRules.asJson

  try {
    writer.println(json)
  } finally {
    writer.close()
  }
}
