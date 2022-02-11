name := "spark-tree-plotting"

version := "0.3"

scalaVersion := "2.12.10"

libraryDependencies += "net.liftweb" % "lift-json_2.12" % "3.5.0"

libraryDependencies += "org.apache.spark" % "spark-core_2.12" % "3.1.0"

libraryDependencies += "org.apache.spark" % "spark-mllib_2.12" % "3.1.0"

spShortDescription := "A simple tool for plotting Spark ML's Decision Trees"

spDescription := """This module provides a simple tool for plotting an easy to understand graphical representation
                    |of Spark ML's DecisionTreeClassificationModels, very similar to the one Python's Scikit-Learn provides.
                    |Given a DecisionTreeClassificationModel, spark_tree_plotting generates a JSON file with 
                    |the relevant metadata in order to plot the tree. Moreover, a simple JSON-to-DOT python
                    |script allows you to plot trees in PySpark in a very simple manner (just as in Scikit-Learn)""".stripMargin

licenses += "MIT" -> url("https://opensource.org/licenses/MIT")

spIncludeMaven := false

// Resulting name for the assembly jar
assemblyJarName in assembly := "spark-tree-plotting_0.3.jar"

// Do not include the Scala library itself in the jar 
assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

assemblyMergeStrategy := {
  case m if m.toLowerCase.endsWith("manifest.mf") => MergeStrategy.discard
  case m if m.toLowerCase.matches("meta-inf.*\\.sf$") => MergeStrategy.discard
  case "log4j.properties" => MergeStrategy.discard
  case m if m.toLowerCase.startsWith("meta-inf/services/") => MergeStrategy.filterDistinctLines
  case "reference.conf" => MergeStrategy.concat
  case _ => MergeStrategy.first
}
