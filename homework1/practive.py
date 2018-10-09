import numpy

group = numpy.full((2,2),0)
group2 = numpy.full((2,2),1)
group3 = numpy.vstack((group,group2))
group3 = numpy.vstack((group3,group2))
print(group3)

