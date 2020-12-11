import json

def testData(name):
	x_test = {}
	y_test = {}
	with open(name, 'r') as f:
		for line in f:
			line = line.split('\t')
			x_test[int(line[0])] = line[1].lower()
			y_test[int(line[0])] = line[2]
	return x_test, y_test
