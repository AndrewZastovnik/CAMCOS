import scipy as sc
import numpy as np

class factor_weighting():
    """
    Several weighting functions for a matrix A
    """
    def __init__(self,A):
        # A is the matrix you want to find weights for
        self.col_occurance = np.array((A>0).sum(axis=0)).flatten()/A.shape[0]

    def threashold(self,c,start,end):
        # Hard threashold
        # c is the number of clusters
        weights = (self.col_occurance <1/c+end)*(self.col_occurance >1/c - start)
        return sc.sparse.diags(weights,dtype=np.bool)

    def linear(self,c,start,end):
        m  = -1/(start - 1/c)
        b = - m / c + 1
        m2 = 1/(1/c-end)
        b2 = 1 - m2/c
        def g(x):
            if x < 1/c:
                return m*x+b
            else:
                return m2*x+b2
        g = np.vectorize(g)
        return sc.sparse.diags(g(self.col_occurance))

    def beta(self,c,b):
        a  = (b/c -2/c + 1) / (1-1/c)
        return sc.sparse.diags(sc.stats.beta.pdf(self.col_occurance,a,b))


#pyspark functions

#spark = sparksession()

#df = spark.read.parquet("/home/andrew/Documents/CAMCOS/Verizon/Verizon data/agg/oneday/feature/day=2016-11-25")
#df2 = df.toPandas()
#df2.to_csv('/home/andrew/downloads/example.csv')

#df = spark.read.parquet("/home/andrew/Documents/CAMCOS/Verizon/Verizon data/agg/oneday/profile/date=20161125")
#df2 = df.toPandas()
#df2.to_csv('/home/andrew/downloads/example.csv')

#df = spark.read.parquet("/home/andrew/Documents/CAMCOS/Verizon/Verizon data/agg/agg/hist/date=20170202")
#df2 = df.toPandas()
#df2.to_csv('/home/andrew/downloads/example.csv')


#df = spark.read.parquet("/home/andrew/Documents/CAMCOS/Verizon/Verizon data/agg/agg/profileAgg/date=20161125")

#converte to csv
import os
for dir in os.