package com.jasoto.spark.ml

case class decisionNode(id:Int,
                        featureIndex:Option[Int], 
                        gain:Option[Double], 
                        impurity:Double,
                        threshold:Option[Double], // Continuous split
                        nodeType:String, // Internal or leaf
                        splitType: Option[String], // Continuous and categorical
                        leftCategories: Option[Array[Double]], // Categorical Split
                        rightCategories: Option[Array[Double]], // Categorical Split
                        prediction:Double,
                        leftChild:Option[decisionNode],
                        rightChild:Option[decisionNode])

object Functions {

def get_node_type(node:org.apache.spark.ml.tree.Node):String = node match {
    case internal: org.apache.spark.ml.tree.InternalNode => "internal"
    case other => "leaf"
    }

def get_split_type(split:org.apache.spark.ml.tree.Split):String = split match {
    case continuous: org.apache.spark.ml.tree.ContinuousSplit => "continuous"
    case other => "categorical"
  }

}

class SparkMLTree(tree:org.apache.spark.ml.classification.DecisionTreeClassificationModel) {
    def get_decision_rules(node: org.apache.spark.ml.tree.Node, id:Int=0):decisionNode = {
        val node_type = Functions.get_node_type(node)

        val gain:Option[Double] = node_type match {
            case "internal" => Some(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].gain)
            case "leaf" => None
        }

        val feature_index:Option[Int] = node_type match {
            case "internal" => Some(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].split.featureIndex)
            case "leaf" => None
        }

        val split_type:Option[String] = node_type match {
            case "internal" => Some(Functions.get_split_type(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].split))
            case "leaf" => None
        }

        val node_threshold:Option[Double] = split_type match {
            case Some("continuous") => Some(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].split.asInstanceOf[org.apache.spark.ml.tree.ContinuousSplit].threshold)
            case Some("categorical") => None
            case other => None
            }

        val (left_categories:Option[Array[Double]],
             right_categories:Option[Array[Double]]) = split_type match {
            case Some("categorical") => (Some(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].split.asInstanceOf[org.apache.spark.ml.tree.CategoricalSplit].leftCategories),
                                         Some(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].split.asInstanceOf[org.apache.spark.ml.tree.CategoricalSplit].rightCategories)
                                         )
            case other => (None,None)
        }

        val left_child: Option[decisionNode] = node_type match {
            case "internal" => Some(get_decision_rules(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].leftChild, id+1))
            case other => None
            }

        val right_child: Option[decisionNode] = node_type match {
            case "internal" => Some(get_decision_rules(node.asInstanceOf[org.apache.spark.ml.tree.InternalNode].rightChild, id+2))
            case other => None  
        }

        decisionNode(id = id,
                     featureIndex = feature_index,
                     gain = gain,
                     impurity = node.impurity,
                     threshold = node_threshold,
                     nodeType = node_type,
                     splitType = split_type,
                     leftCategories = left_categories, // Categorical Split
                     rightCategories = right_categories, // Categorical Split
                     prediction = node.prediction,
                     leftChild = left_child,
                     rightChild = right_child
                     )
    }
    
    def toJsonPlotFormat():String = {
        implicit val formats = net.liftweb.json.DefaultFormats
        net.liftweb.json.Serialization.writePretty(get_decision_rules(tree.rootNode))
    }
    
}

object Implicits {
    implicit def genTree(tree:org.apache.spark.ml.classification.DecisionTreeClassificationModel) = new SparkMLTree(tree)
}
