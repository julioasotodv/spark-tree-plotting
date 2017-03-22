# spark-tree-plotting

This module provides a simple tool for plotting an easy to understand graphical representation of Spark ML's DecisionTreeClassificationModels, very similar to the one Python's Scikit-Learn provides.
Given a `DecisionTreeClassificationModel`, spark_tree_plotting generates a JSON file with
the relevant metadata in order to plot the tree. Moreover, a simple JSON-to-DOT python
script allows you to plot trees in PySpark in a very simple manner (just as in Scikit-Learn).
