# spark-tree-plotting

![Iris](https://i.imgur.com/FcqKe9w.png)

This module provides a simple tool for plotting an easy to understand graphical representation of Spark ML's DecisionTreeClassificationModels, very similar to the one Python's Scikit-Learn provides.
Given a `DecisionTreeClassificationModel`, spark_tree_plotting generates a JSON file with the relevant metadata in order to plot the tree. Each split node (either continuous or categorical) is parsed with its impurity measure and gain.

Moreover, a simple JSON-to-DOT Python function allows you to plot trees in PySpark in a very simple manner (just as in Scikit-Learn).

## Demo
A Jupyter notebook can be found [here]().

## Requirements
- Apache Spark (versions higher than 2.0 are supported).
- Optionally, `pydot3` in the Spark driver if you plan to plot trees with th built-in functionality under PySpark. You can easily install it with `pip install pydot3`.

## Installation

### Online
If your Spark cluster has Internet access, the easiest way to use `spark-tree-plotting` is through [spark-packages.org](https://spark-packages.org/). There are two ways of doing so:

The first one is just through your terminal, But you will need to do it for every new Spark app that you launch:
```bash
~$ spark-shell/pyspark/spark-submit --packages julioasotodv:spark-tree-plotting:0.2
```

The other one is adding the following line at the end of your `spark-defaults.conf` file. Once done, all new Spark apps will be able to use the package:

```bash
spark.jars.packages    julioasotodv:spark-tree-plotting:0.2
```


### Offline
However, lots of clusters do not have Internet access nowadays. To manually install `spark-tree-plotting`, you will need `sbt` and `git` in order to build a jar file.

Once you have `sbt`, just follow these steps:
```bash
git clone https://github.com/julioasotodv/spark-tree-plotting.git
cd spark-tree-plotting
sbt assembly
```
This will generate a jar file in `target/Scala-2.X/spark-tree-plotting_0.2.jar`.

If you just need to create a JSON file out of your trees, you can just add the following line to your `spark-defaults.conf` file:
```
spark.jars	/path/to/the/spark-tree-plotting_0.2.jar
```

Ohterwise, if you plan to use the Python plotting utilities, you will need to start your Spark sessions as follows:
```
spark-shell/pyspark/spark-submit \
	--jars /path/to/the/spark-tree-plotting_0.2.jar \
	--driver-class-path /path/to/the/spark-tree-plotting_0.2.jar \
	--py-files /path/to/the/spark-tree-plotting_0.2.jar \
```

## Usage

### Tree plotting
For now, you can only plot trees directly through PySpark:

```python
from spark_tree_plotting import plot_tree  # requires pydot3

your_dtree_classification_model = (create a DecisionTreeClassificationModel)

png_string = plot_tree(your_dtree_classification_model,
                       featureNames=list_of_feature_names,
                       categoryNames=dict_of_category_names_for_categorical_features,
                       classNames=list_of_class_names,
                       filled=True,
                       roundedCorners=True,
                       roundLeaves=True)
```
Now you can do whatever you want with `png_string`, which is a binary PNG that can be written to disk or displayed directly through Jupyter for instance. [A Jupyter notebook](https://nbviewer.jupyter.org/github/julioasotodv/spark-tree-plotting/blob/master/examples/Example_covertype_dataset.ipynb) showing this functionality can be found in the `examples` folder.

If you just want the DOT-format string:

```python
from spark_tree_plotting import export_graphviz

dot_string = export_graphviz(your_dtree_classification_model,
                             featureNames=list_of_feature_names,
                             categoryNames=dict_of_category_names_for_categorical_features,
                             classNames=list_of_class_names,
                             filled=True,
                             roundedCorners=True,
                             roundLeaves=True)
```
`dot_string` can be processed with any DOT parsing library.

### Get JSON from tree
But maybe you just need the tree in a JSON format, if you want to develop your own visualization library in whatever language you love (perhaps D3?). You can do this either from Python or Scala:

#### Python

```python
from spark_tree_plotting import generate_tree_json

tree_json = generate_tree_json(your_dtree_classification_model,
                               withNodeIDs=True # each tree with node ID
                               )
```

#### Scala
```scala
import com.jasoto.spark.ml.SparkMLTree

val tree_json = new SparkMLTree(your_dtree_classification_model).toJsonPlotFormat()
```
The same can be done through implicits:

```scala
import com.jasoto.spark.ml._

val tree_json = your_dtree_classification_model.toJsonPlotFormat()
```

