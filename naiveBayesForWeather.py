import csv
import random
import math

def loadCsv(filename):
	lines = csv.reader(open(filename, "rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset

def splitDataset(dataset, splitRatio):
	trainSize = int(len(dataset) * splitRatio)
	trainSet = []
	copy = list(dataset)
	while len(trainSet) < trainSize:
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet, copy]

def separateByClass(dataset):
	separated = {0: [], 1: []}
	for i in range(len(dataset)):
		vector = dataset[i]
		num = vector[-1];
		if(num>=32):
			num = num - 32
		if(int(num/16) == 1):
			separated[1].append(vector)
		else:
			separated[0].append(vector)

	return separated

def mean(numbers):
	return sum(numbers)/float(len(numbers))
 
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
	return math.sqrt(variance)

def summarize(dataset):
	summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

def summarizeByClass(dataset):
	separated = separateByClass(dataset)
	summaries = {}
	for classValue, instances in separated.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

def calculateProbability(x, mean, stdev):
	exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
	return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
 
def calculateClassProbabilities(summaries, inputVector):
	probabilities = {}
	for classValue, classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean, stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x, mean, stdev)
	return probabilities
			
def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel
 
def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

def getEvents(testSet, predictions):
	events = {"tp": 0, "fp": 0, "tn": 0, "fn": 0};
	for i in range(len(testSet)):
		num = testSet[i][-1]
		if(num>=32):
			num = num - 32
		if(num>=16):
			classValue = 1
		else:
			classValue = 0

		if classValue == 0 and predictions[i] == 0:
			events["tn"] += 1
		elif classValue == 0 and predictions[i] == 1:
			events["fp"] += 1
		elif classValue == 1 and predictions[i] == 0:
			events["fn"] += 1
		elif classValue == 1 and predictions[i] == 1:
			events["tp"] += 1
	return events

def getAccuracy(testSet, predictions):
	correct = 0
	for i in range(len(testSet)):
		num = testSet[i][-1]
		if(num>=32):
			num = num - 32
		if(int(num/16) == 1):
			classValue = 1
		else:
			classValue = 0
		if classValue == predictions[i]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0

def meanAccuracy(filename, splitRatio, dataset):
	accuracyList = []
	for i in range (100):
		trainingSet, testSet = splitDataset(dataset, splitRatio)
		# prepare model
		summaries = summarizeByClass(trainingSet)
		# test model
		predictions = getPredictions(summaries, testSet)
		accuracy = getAccuracy(testSet, predictions)
		accuracyList.append(accuracy)
	return mean(accuracyList)	

def main():
	filename = 'dataset3.csv'
	splitRatio = 0.95
	dataset = loadCsv(filename)
	trainingSet, testSet = splitDataset(dataset, splitRatio)
	print('\nSplit {0} rows into train={1} and test={2} rows').format(len(dataset), len(trainingSet), len(testSet))
	# prepare model
	summaries = summarizeByClass(trainingSet)
	# test model
	predictions = getPredictions(summaries, testSet)
	events = getEvents(testSet, predictions)
	print "\nTest set is of %s values" % str(len(testSet))
	print "True Positives: %s" % (str(events["tp"])) 
	print "True Negatives: %s" % (str(events["tn"])) 
	print "False Positives: %s" % (str(events["fp"])) 
	print "False Negatives: %s" % (str(events["fn"]))

	precision = float(events["tp"])/(events["tp"] + events["fp"])
	recall = float(events["tp"])/(events["tp"] + events["fn"])

	print "\nPrecision: %s" % (str(precision))
	print "Recall: %s" % (str(recall))

	accuracy = getAccuracy(testSet, predictions)

	print "\nNumber of Correct Predictions in testset of %s: %s" % (str(len(testSet)), str(events["tp"] + events["tn"]))
	print('Accuracy: {0}%').format(accuracy)
	print "Mean Accuracy of 100 random measurements: %s \n\n\n" % str(meanAccuracy(filename, splitRatio, dataset))

main()