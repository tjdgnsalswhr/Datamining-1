import csv

f = open("cancer_test.csv", "r")
reader =  csv.reader(f)
for line in reader:
	print(line)
f.close
