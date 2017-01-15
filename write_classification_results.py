import os
resultsPath = "/home/andras/data/datasets/FL32/FlickrLogos-v2/fl_devkit/results"
threshold = 0.7


resultValues = dict()
resultClasses = dict()
for subdir, dirs, files in os.walk(resultsPath):
	for file in files:
		with open(os.path.join(subdir, file), 'r') as classFile:
			lines = classFile.readlines()

		for data in lines:
			data = data.split(' ')
			if (not data[0] in resultValues) or resultValues[data[0]] < data[1]:
				resultValues[data[0]] = data[1]
				resultClasses[data[0]] = (file.split('_')[-1]).split('.')[0]

with open(os.path.join(resultsPath, "classification.txt"), 'w') as resultsFile:
	for key in resultValues:
		value = resultValues[key]
		if float(value) < threshold:
			imageClass = "no-logo"
			value = 1.0
		else:
			imageClass = resultClasses[key]
		resultsFile.write(str(key) + '\t' + imageClass + '\t' + str(value) + '\n')
