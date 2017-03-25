name := "spark-tree-plotting"

version := "0.2"

scalaVersion := "2.11.8"

libraryDependencies += "net.liftweb" % "lift-json_2.11" % "3.0.1"


assemblyShadeRules in assembly := Seq(
    ShadeRule.rename("net.liftweb.json.**" -> "org.lift.web.library.json.@1").inAll
)

// Spark Packages config
spName := "julioasotodv/spark-tree-plotting"

sparkVersion := "2.0.0"

sparkComponents += "mllib"

credentials += Credentials(Path.userHome / ".ivy2" / ".sbtcredentials")

spShortDescription := "A simple tool for plotting Spark ML's Decision Trees"

spDescription := """This module provides a simple tool for plotting an easy to understand graphical representation
                    |of Spark ML's DecisionTreeClassificationModels, very similar to the one Python's Scikit-Learn provides.
                    |Given a DecisionTreeClassificationModel, spark_tree_plotting generates a JSON file with 
                    |the relevant metadata in order to plot the tree. Moreover, a simple JSON-to-DOT python
                    |script allows you to plot trees in PySpark in a very simple manner (just as in Scikit-Learn)""".stripMargin

licenses += "MIT" -> url("https://opensource.org/licenses/MIT")

spIncludeMaven := false


// Resulting name for the assembly jar
jarName in assembly := "spark-tree-plotting_0.2.jar"

// Do not include the Scala library itself in the jar 
assemblyOption in assembly := (assemblyOption in assembly).value.copy(includeScala = false)

