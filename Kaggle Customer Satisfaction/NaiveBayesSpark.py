from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint


def parseLine(line):
	parts = line.split(',')
	label = float(parts[-1])
	parts.pop()
	features = Vectors.dense([float(x) for x in parts[1:]])
	
	return LabeledPoint(label,features)


train_data = sc.textFile("kaggle/customer/train.csv").filter(lambda line : line[0].isdigit()).map(parseLine)

training, test = train_data.randomSplit([0.7,0.3],seed=0)

model = NaiveBayes.train(training,1.0)

predictionAndLabel = test.map(lambda p : (model.predict(p.features),p.label))
accuracy = 1.0 * (predictionAndLabel.filter(lambda (x,v): x == v).count() / test.count())

