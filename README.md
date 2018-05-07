# IncomePredictor
Compare and analysis on using three different supervised machine learning algorithms to build an income predictor

Solution 1: Linear regression

Regression models are used to test if a relationship exists between variables; that is, to use one variable to predict another.  However, there is some random error that cannot be predicted. There is a simple formula to represent the basic relationship between the input and output in the linear regression model [2].
The Simple Linear Regression formula: Y = β0+ β1X + error
Where, 
Y = dependent variable (response) 	
X = independent variable (predictor / explanatory)
β0= intercept (value of Y when X = 0)
β1= slope of the regression line  
Error = random error
Sample data or labeled data are used to estimated the true values for the intercept and slope of the formula. Using the sample data, Y = b0 + b1X, and Y = predicted value of Y. The difference between the actual value of Y and the predicted value (using sample data) is known as the error. 
So, Error = (actual value) – (predicted value) 
The first solution has used linear regression to build the machine learning model. The total count of all the available data is 101050. During the data preparation process, the whole dataset has been randomly split with a 70% and 30% ratio, which means that the model is trained on 70% of the available data, and being evaluated on the rest 30%. Training the linear regression model took around 20s and evaluating the model took around 10s. 
The resultant coefficients of the model were: [5134.983665365605, -196738.8587523364, 
-5375.06711782812, -4161.871545497067,0.0, -32675.9606318151]   
Intercept: 1153608.7303823817
From the cross-validation result, the percentage that the linear regression model was able to successfully predict the income of a person is 40%. The result means that out of 30000 people there were 15000 people’s income got successfully predicted. In Figure 1, it is clear that the result is better than random, because excluding all the outliers, the model was able to successfully predict most of the middle-class people’s income. After analyzing all the important perspectives, the summarized results are show in Table 1.

![alt text](https://github.com/YusufLiu/IncomePredictor/blob/master/LRResult.png)

Figure 1 Distribution of LR prediction 

![alt text](https://github.com/YusufLiu/IncomePredictor/blob/master/Summarized%20Result%20for%20LR.png)

Table 1. Summarized result for LR 

From the summarized table, the linear regression model received a score of 4.0 for performance a score of (10-4) *(10-6)/10 = 2.4 for cost, and a score of (10 – 1.5) *(10-1)/10 =7.7 for response time. Thus, the final score for random forest model is show in Table 2.

![alt text](https://github.com/YusufLiu/IncomePredictor/blob/master/Scoring%20Matrix%20for%20LR.png)

Table 2. Scoring matrix for LR 

Solution 2: Random Forest Regression

One of the most fundamental and popular method for various machine learning task is decision trees. Tree learning “comes closest to meeting the requirements for serving as an off-the-shelf procedure for data mining”, say Hastie et al. [5]. Decision tree is invariant under scaling and various other transformations of feature values, and it is robust to inclusion of irrelevant features, and produces inspectable models. Random Forest is based on having multiple decision trees, and ensemble those tree’s result to get a regression value.
Given a standard training set D of size n, bagging generates m new training sets D_i, each of size n′, by sampling from D uniformly and with replacement. This kind of sample is known as a bootstrap sample. The m models are fitted using the above m bootstrap samples and combined by averaging the output (for regression) or voting (for classification). 
Random Forest Algorithm [3]: 
Given a training set with n samples {X，Y} 
for b=1, …, B: 
1.From X there are n samples returned to form a collection {Xb, Yb}; 
2. In {Xb, Yb} train the decision tree (or regression trees); 
End
After the training is done, take the average of all models as output (or vote with the majority vote).
The second solution has used random forest regression to build the machine learning model. The total count of all the available data is still the same, which is 101050. During the data preparation process, the team decided keep the 70% and 30% ratio of training data, and testing data. Training the linear regression model took around 40s and evaluating the model took around 10s. The model has set a limit of 10 trees in the random forest model.
From the cross-validation result, the percentage that the linear regression model was able to successfully predict the income of a person is 75%. The result means that out of 30000 people there were 21500 people’s income got successfully predicted. In Figure 2, it is clear that the result is better than linear regression, because excluding all the outliers, the model was able to successfully predict most of the middle-class people’s income and the lower-class people’s income. After analyzing all the important perspectives, the summarized results are show in Table 3. 

![alt text](https://github.com/YusufLiu/IncomePredictor/blob/master/RFResult.png)

Figure 2 Distribution of RF prediction  

![alt text](https://github.com/YusufLiu/IncomePredictor/blob/master/Summarized%20Result%20for%20RF.png)

Table 3. Summarized results for RF 

From the summarized table, the random forest regression model received a score of 7.5 for performance a score of (10-8) *(10-6)/10 = 0.8 for cost, and a score of (10 – 4)*(10-1)/10 =5.4 for response time. Thus, the final score for random forest model is show in Table 4.

![alt text](https://github.com/YusufLiu/IncomePredictor/blob/master/Scoring%20Matrix%20for%20RF.png)

Table 4. Scoring matrix for RF 







