from json import loads
from pyspark import SparkContext

def generate_color_brew(n):
    hue_step = 360 / float(n)
    return [color * hue_step / 360.0 for color in range(n)]


def node_to_str(node, featureNames, categoryNames, classNames, numClasses, 
                nodeList, filled, round_leaves):
    # classNames preparation:
    if classNames is None:
        class_name = node["prediction"]
        class_name_str = "Class #" + str(int(node["prediction"]))
    else:
        class_names = dict(enumerate(classNames))
        class_name = class_names[node["prediction"]]
        class_name_str = str(class_name)
        
    # Names preparation (featureNames, categoryNames):
    if node["nodeType"] == "internal":
        
        if featureNames is None:
            feature_name = node["featureIndex"]
            feature_name_str = "feature #" + str(feature_name)
        else:
            featureNames = dict(enumerate(featureNames))
            feature_name = featureNames[node["featureIndex"]]
            feature_name_str = str(feature_name)
        
        if categoryNames is None:
            if node["splitType"] == "categorical":
                categories = "categories# " + "{" + ",".join(str(n) for n in node["leftCategories"]) + "}"
        else:
            if node["splitType"] == "categorical":
                try:
                    category_names = dict(enumerate(categoryNames[feature_name]))
                    categories = "{" + ",".join(category_names[n] for n in node["leftCategories"]) + "}"
                except KeyError:
                    categories = "categories# " + "{" + ",".join(str(n) for n in node["leftCategories"]) + "}"
        
        # For continuous split:
        if node["splitType"] == "continuous":
            graph_string = """ "%s <= %.4f\\nImpurity = %.4f\\nGain = %.4f\\nPrediction = %s" """ % (feature_name_str,
                                                                                         node["threshold"],
                                                                                         node["impurity"],
                                                                                         node["gain"],
                                                                                         class_name_str
                                                                                        )
        # For categorical split:
        else:
            graph_string = """ "%s in %s\\nImpurity = %.4f\\nGain = %.4f\\nPrediction = %s" """ % (feature_name_str,
                                                                                         categories,
                                                                                         node["impurity"],
                                                                                         node["gain"],
                                                                                         class_name_str
                                                                                        )
    # Leaf node:
    else:
        graph_string = """ "Impurity = %.4f\\nPrediction = %s" """ % (node["impurity"],
                                                                 class_name_str
                                                               )
        if round_leaves is True:
            nodeList.append(graph_string + "[shape=ellipse]") # Change leaf shape
    
    # Color adding:
    if filled is True:
        h = generate_color_brew(numClasses)[int(node["prediction"])]
        s = node["impurity"]
        nodeList.append(graph_string + ' [fillcolor="%.4f,%.4f,%.4f"]' % (h,s,1.0))
    return graph_string

def get_num_classes(node, current_classes=None):
    if current_classes is None:
        current_classes = set()
    else:
        current_classes = current_classes.copy()
    if node["nodeType"] == "internal":
        current_classes.add(node["prediction"])
        return (get_num_classes(node["leftChild"], current_classes=current_classes) 
                | get_num_classes(node["rightChild"], current_classes=current_classes)
               )
    else:
        current_classes.add(node["prediction"])
        return current_classes

def relations_to_str(node, featureNames=None, categoryNames=None, classNames=None, 
                     numClasses=None, nodeList=None, filled=True, round_leaves=True):
    nodes_to_explore = [node]
    relations = []
    while len(nodes_to_explore) > 0:
        if len(nodes_to_explore) == 0:
            break
        current_node = nodes_to_explore.pop()
        if current_node["nodeType"] == "leaf":
            continue
        relations.append(node_to_str(current_node, featureNames, categoryNames, 
                                     classNames, numClasses, nodeList, filled, round_leaves) 
                         + "->" 
                         + node_to_str(current_node["leftChild"], featureNames, categoryNames, 
                                       classNames, numClasses, nodeList, filled, round_leaves) 
                         + '[labeldistance=2.5, labelangle=45., headlabel="True"]' 
                         + "\n")
        nodes_to_explore.append(current_node["leftChild"])
        
        relations.append(node_to_str(current_node, featureNames, categoryNames, 
                                     classNames, numClasses, nodeList, filled, round_leaves) 
                         + "->" 
                         + node_to_str(current_node["rightChild"], featureNames, categoryNames, 
                                       classNames, numClasses, nodeList, filled, round_leaves) 
                         + '[labeldistance=2.5, labelangle=-45., headlabel="False"]' 
                         + "\n")
        nodes_to_explore.append(current_node["rightChild"])
    return relations


def export_graphviz(DecisionTreeClassificationModel, featureNames=None, categoryNames=None, classNames=None,
                   filled=True, rounded_corners=True, round_leaves=True):
    sc = SparkContext.getOrCreate()
    tree_dict = loads(sc._jvm.com.jasoto.spark.ml.SparkMLTree(DecisionTreeClassificationModel._java_obj).toJsonPlotFormat())

    num_classes = len(get_num_classes(tree_dict))
    node_list = []
    graph = relations_to_str(tree_dict, 
                             featureNames=featureNames, 
                             categoryNames=categoryNames, 
                             classNames=classNames, 
                             numClasses=num_classes,
                             nodeList=node_list,
                             filled=filled,
                             round_leaves=round_leaves)
    node_properties = "\n".join(node_list)
    filled_and_rounded = []
    if filled:
        filled_and_rounded.append("filled")
    if rounded_corners:
        filled_and_rounded.append("rounded")
    dot_string = """digraph Tree {
                    node [shape=box style="%s"]
                    subgraph body {
                    %s
                    %s}
                    }""" % (",".join(filled_and_rounded), "".join(graph), node_properties)
    return dot_string
    