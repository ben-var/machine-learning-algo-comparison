Overview: The entire program was generated in python==3.6.5, and sklearn/numpy/etc were all installed
using the default pip command (e.g., pip install sklearn, etc...).

The datasets were obtained via UCI's machine learning dataset repository,
Bank Dataset: https://archive.ics.uci.edu/ml/datasets/bank+marketing
	i. Notes for bank dataset:
		a. One column was removed per instructions on the dataset download page to make
		   the classification more difficult. "14 - previous:"
Letter Dataset: https://archive.ics.uci.edu/ml/datasets/letter+recognition
	i. Notes for letter dataset:
		a. the only modification to this dataset was converting the classes to numerical
		   representation (A --> 0, B --> 1, etc.), and moving the class column to the end.

Critical Files:

1. util.py - contains all the supporting code that applies to multiple classifiers (not a script)
2. Classifier Scripts:
	i.   DecisionTreeLearner.py
	ii.  kNNLearner.py
	iii. NeuralNetLearner.py
	iv.  SVMLearner.py
	v.   BoostLearner.py
These files are used to generate validation curves, learning curves, and perform benchmark tests.

Call them by running 'python ___Learner.py'.

Each script is modified whenever a specific parameter must be run. For example, the method
"make_val_curve()" from the util.py file is used to generate a validation curve for a provided
classifier using standard sklearn functions (tweaked to fit the project). If you would like to
generate a validation curve for base estimators for an ensemble learner, please use the
"ensemble_val_curve" method from util.py. It will take as input an array of base learners to
test over the curve.

Another important method is plot learning curve. I will provide some documentation in the util.py file
via comments to make the method easier to understand. There are also examples scattered across
the classifier scripts. Most of the parameters I added were for custom file naming and such.

3. gen_final_stats.py - contains the script to generate ROC curves, and time/error dicts used
at the end of the analysis.

4. data/ folder
	i.   bank-additional-full-clean.csv - contains the cleaned (numericalized)
	     data for the bank dataset.
	ii.  bank-additional-full.csv - the raw datafile that was converted to csv
	     prior to cleaning (was initially semi-colon separated)
	iii. letter-recognition.csv - the raw datafile that was converted from
	     "letter-recognition.data" by importing into Microsoft Excel
	iv.  letter-recognition.csv - the cleaned data for the letter dataset
	v.   names files are information about the datasets
	vi.  time_error_data has some data used in the Report-Data.xslx file
	vii. Report-Data xslx contains some chart data.

5. graphs/ is not included, because the graphs I deemed as important were included in
   the report instead. To simplify the directory, I am not going to upload all
   the graphs onto github.

A combination of python scripts and Excel were used to clean the data. Most of the
data cleaning code was written by me in the util.py file. Some of load_data.csv
was adapted from starter code in the CS6601 course.

Sklearn code was adapted from the sklearn documentation or otherwise coded from scratch.
