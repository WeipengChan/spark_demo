from pyspark import SparkContext
from pyspark import sql
import os
os.environ['PYSPARK_PYTHON'] = '/home/ubuntu/.pyenv/versions/3.6.2/bin/python'

def plt_show(df):
    import numpy as np
    import matplotlib.pyplot as plt
    n_groups = 5
    x = df.groupBy("rating").count().select('count')
    xx = x.rdd.flatMap(lambda x: x).collect()
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 1
    opacity = 0.4
    rects1 = plt.bar(index, xx, bar_width,alpha=opacity,color='b', label='ratings')

    plt.xlabel('ratings')
    plt.ylabel('Counts')
    plt.title('Distribution of ratings')
    plt.xticks(index , ('1.0', '2.0', '3.0', '4.0', '5.0'))
    plt.legend()
    plt.tight_layout()
    plt.show()



sc = SparkContext("local", "First App")

coll = list(["a", "b", "c", "d", "e"])
rdd_from_coll = sc.parallelize(coll)

#creating RDD from a referenced file
rdd_from_Text_File = sc.textFile("./testdata.txt")

#Data loading
data = sc.textFile("./ml-100k/u.data")

#loaded data will be a spark RDD type, run the below command to findout the data type of data object.
#print(type(data),data.count(),data.first())
print(data.take(5))

# total length of the data loaded is given by:
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
ratings = data.map(lambda l: l.split('\t'))\
	    .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
print(ratings.take(5))

#创建df
sql_ctx = sql.SQLContext(sc)
df =  sql_ctx.createDataFrame(ratings, ['UserID', 'product',"Rating"])
#df.select('user').distinct().show(100)

user_count = df.groupBy("UserID" ).count()
print(type(user_count))

#漂亮的直方图
#plt_show(df)
#

df.stat.crosstab("UserID", "Rating").show()

#分割训练集
(training, test) = ratings.randomSplit([0.8, 0.2])

# Build the recommendation model using Alternating Least Squares
#setting rank and maxIter parameters
#设置模型以及参数
rank = 10
numIterations = 10
model = ALS.train(training, rank, numIterations)
testdata = test.map(lambda p: (p[0], p[1]))

#prediction
pred_ind = model.predict(119, 392)
print(pred_ind)

predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
print(predictions.take(5))
ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
print(ratesAndPreds.take(5))

MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
from math import sqrt
rmse = sqrt(MSE)
print("Root-mean-square error = " + str(rmse))

#understanding
(training, test) = df.randomSplit([0.8, 0.2])
print(training.count(),test.count())
from pyspark.ml.recommendation import ALS
als = ALS(userCol="UserID", itemCol="product", ratingCol="Rating")
#create pipeline object and setting the created als model as a stage in the pipeline
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[als])
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
paramMapExplicit = ParamGridBuilder() \
                    .addGrid(als.rank, [8, 12]) \
                    .addGrid(als.maxIter, [10, 15]) \
                    .addGrid(als.regParam, [1.0, 10.0]) \
                    .build()
from pyspark.ml.evaluation import RegressionEvaluator
#calling RegressionEvaluator() method with evaluation metric set to rmse and evaluation column set to Rating
evaluatorR = RegressionEvaluator(metricName="rmse", labelCol="Rating")
cvExplicit = CrossValidator(estimator=als, estimatorParamMaps=paramMapExplicit, evaluator=evaluatorR,numFolds=2)
cvModel = cvExplicit.fit(training)
preds = cvModel.transform(test)
print(preds.take(20))
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Rating",predictionCol="prediction")

preds = preds.na.drop()#nan 是由于一些测试集的用户没有在训练集引起的

rmse = evaluator.evaluate(preds)

print("Root-mean-square error = " + str(rmse))





#妥妥的UBCF
# recommedItemsToUsers = model.recommendProductsForUsers(10)
# r_count = recommedItemsToUsers.count()
# print(r_count)
