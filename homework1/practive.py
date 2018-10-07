import numpy

group = numpy.full((4,4),1)
group2 = numpy.full((4,4),3)
print(group)
print(group2)


group3 = group - group2

print(group3)

group4 = group3**2

print(group4)

a = numpy.sum(group4)

print(a)

b = numpy.sqrt(36)

print(b)
