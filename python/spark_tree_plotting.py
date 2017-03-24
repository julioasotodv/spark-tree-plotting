from json import loads, dumps
from pyspark import SparkContext

def generate_color_brew(n):
    hue_step = 360 / float(n)
    return [color * hue_step / 360.0 for color in range(n)]


def node_to_str(node, featureNames, categoryNames, classNames, numClasses, 
                nodeList, filled, round_leaves, colorBrew):
    # classNames preparation:
    if classNames is None:
        class_name = node["prediction"]
        class_name_str = "Class #" + str(int(node["prediction"]))
    else:
        class_names = dict(enumerate(classNames))
        class_name = class_names[node["prediction"]]
        class_name_str = str(class_name)
    
    attributes = []

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
            label = """ label="%s <= %.4f\\nImpurity = %.4f\\nGain = %.4f\\nPrediction = %s" """ % (feature_name_str,
                                                                                                    node["threshold"],
                                                                                                    node["impurity"],
                                                                                                    node["gain"],
                                                                                                    class_name_str
                                                                                                   )
        # For categorical split:
        else:
            label = """ label="%s in %s\\nImpurity = %.4f\\nGain = %.4f\\nPrediction = %s" """ % (feature_name_str,
                                                                                                  categories,
                                                                                                  node["impurity"],
                                                                                                  node["gain"],
                                                                                                  class_name_str
                                                                                                  )
    # Leaf node:
    else:
        label = """ label="Impurity = %.4f\\nPrediction = %s" """ % (node["impurity"],
                                                                     class_name_str
                                                                    )
        if round_leaves is True:
            attributes.append("shape=ellipse")
            #nodeList.append(graph_string + "[shape=ellipse]") # Change leaf shape
    
    attributes.append(label)

    # Color adding:
    if filled is True:
        h = colorBrew[int(node["prediction"])]
        s = node["impurity"]
        attributes.append('fillcolor="%.4f,%.4f,%.4f"' % (h,s,1.0))
        #nodeList.append(graph_string + ' [fillcolor="%.4f,%.4f,%.4f"]' % (h,s,1.0))

    nodeList.append("%s [%s]" % (node["id"],
                                 ",".join(attributes)))

    return str(node["id"])

def get_num_classes(node):
    nodes_to_explore = [node]
    classes = set()
    while len(nodes_to_explore) > 0:
        if len(nodes_to_explore) == 0:
            break
        current_node = nodes_to_explore.pop()
        classes.add(current_node["prediction"])
        
        if current_node["nodeType"] == "internal":
            nodes_to_explore.append(current_node["leftChild"])
            nodes_to_explore.append(current_node["rightChild"])
    return int(max(list(classes)) + 1)

def add_node_ids(node):
    nodes_to_explore = [node]
    counter = -1
    while len(nodes_to_explore) > 0:
        if len(nodes_to_explore) == 0:
            break
        current_node = nodes_to_explore.pop()
        counter += 1
        current_node["id"] = counter
        #classes.add(current_node["prediction"])
        
        if current_node["nodeType"] == "internal":
            nodes_to_explore.append(current_node["rightChild"])
            nodes_to_explore.append(current_node["leftChild"])
    return node

def relations_to_str(node, featureNames=None, categoryNames=None, classNames=None, 
                     numClasses=None, nodeList=None, filled=True, roundLeaves=True,
                     color_brew=None):
    nodes_to_explore = [node]
    relations = []
    while len(nodes_to_explore) > 0:
        if len(nodes_to_explore) == 0:
            break
        current_node = nodes_to_explore.pop()
        if current_node["nodeType"] == "leaf":
            continue
        relations.append(node_to_str(current_node, featureNames, categoryNames, 
                                     classNames, numClasses, nodeList, filled, roundLeaves, color_brew) 
                         + "->" 
                         + node_to_str(current_node["leftChild"], featureNames, categoryNames, 
                                       classNames, numClasses, nodeList, filled, roundLeaves, color_brew) 
                         + '[labeldistance=2.5, labelangle=45., headlabel="True"]' 
                         + "\n")
        nodes_to_explore.append(current_node["leftChild"])
        
        relations.append(node_to_str(current_node, featureNames, categoryNames, 
                                     classNames, numClasses, nodeList, filled, roundLeaves, color_brew) 
                         + "->" 
                         + node_to_str(current_node["rightChild"], featureNames, categoryNames, 
                                       classNames, numClasses, nodeList, filled, roundLeaves, color_brew) 
                         + '[labeldistance=2.5, labelangle=-45., headlabel="False"]' 
                         + "\n")
        nodes_to_explore.append(current_node["rightChild"])
    return relations

def generate_tree_json(DecisionTreeClassificationModel, withNodeIDs=False):
    sc = SparkContext.getOrCreate()

    json_tree = sc._jvm.com.jasoto.spark.ml.SparkMLTree(DecisionTreeClassificationModel._java_obj).toJsonPlotFormat()

    if withNodeIDs:
        json_tree = dumps(add_node_ids(loads(json_tree)), indent=2)
        
    return json_tree

def export_graphviz(DecisionTreeClassificationModel, featureNames=None, categoryNames=None, classNames=None,
                   filled=True, roundedCorners=True, roundLeaves=True):

    tree_dict = loads(generate_tree_json(DecisionTreeClassificationModel, withNodeIDs=False))
    num_classes = get_num_classes(tree_dict)
    color_brew = generate_color_brew(num_classes)
    node_list = []
    tree_dict_with_id = add_node_ids(tree_dict)

    graph = relations_to_str(tree_dict_with_id,
                             featureNames=featureNames, 
                             categoryNames=categoryNames, 
                             classNames=classNames, 
                             numClasses=num_classes,
                             nodeList=node_list,
                             filled=filled,
                             roundLeaves=roundLeaves,
                             color_brew=color_brew)
    node_properties = "\n".join(node_list)
    filled_and_rounded = []
    if filled:
        filled_and_rounded.append("filled")
    if roundedCorners:
        filled_and_rounded.append("rounded")
    dot_string = """digraph Tree {
                    node [shape=box style="%s"]
                    subgraph body {
                    %s
                    %s}
                    }""" % (",".join(filled_and_rounded), "".join(graph), node_properties)
    return dot_string
    