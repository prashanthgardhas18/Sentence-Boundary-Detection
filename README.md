# Sentence-Boundary-Detection
•	Identifying the period occurrences to detect the EOS/NEOS, for each such period occurrences extracted the 8 features (Left word, Right word, Left word < 3, Right word < 3, is left word is stop word, is right word is stop word, is left word is capitalized, is right word is capitalized).
•	Using the sklearn library, applied a decision tree classifier on the 8 feature vectors (training data) to predict the labels (EOS/NEOS) and the model predicted 82% accuracy of the system on test data.
