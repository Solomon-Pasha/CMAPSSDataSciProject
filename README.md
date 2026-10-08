A python data analysis project for the CMAPSS data (https://data.nasa.gov/docs/legacy/CMAPSSData.zip). Aiming to find out what changes in which sensors correlate with engine failure.

I have written several functions in the quality.py file to check for any issues in the existing data and ensure its integrity, including ensuring that the test data and its real results match exactly in length and ids as well as checking that there are no empty rows or duplicate records to unnecessarily bias the analysis.

To filter out columns which may have no major change over any engines lifespan and thus be effectively irrelevant to finding what foreshadows an engine nearing RUL I used the noise_vs_drift function, which analyses which columns actually change over a engines lifespan.

I used the describe function to generate some basic information on the dataset in the form of the following table
[text](results/summary.txt)

Due to this information and the folowing gathered from the driftVsNoise function:
[text](results/driftVsNoise.txt)
I decided to ignore columns 3, 4, 5, 6, 24, 10, 23, 21 and 15 in my model because they are effectively constant and thus would simply waste space in the model. Any changes in them are either random noise or imperceptibly different from random noise. While columns 3 and 4 displayed high relative standard deviation, this was primarily due to their extremely small mean being inordinately effected by minor fluctuations revealed when noise and deviation are compared.

Notably the datasets readme.txt file is inconsistent in it's notation, the second number resets between operational setting 3 and sensor measurement 1 but the final sensor measurement is labelled 26 when it should be 21.

To further see how the different values act as the engine approaches failure I used matplotlib to produce a graph showing it's change. The results of this are shown in the dataMeasurements png file. Notably several datapoints diverge near the end of their lives, sensors 14 and 9, where some engines spike and others display a gradual decrease, this will change how they have to be handled if I involve them in my regression algorithm.

Some of the sensors were very close together in pattern, to try and check if this shows a real redundancy that could be removed to speed up later model building and running I made a correlation heat map:
![alt text](Figure_2.png)
Which clearly shows that variables 9 and 14 have a near correlation, meaning that using more than one of them in the later regression model would be a waste. Sensors 2, 3, 4, 8, 11, 13, 15 and 17 act as a group with a strong positive correlation which we know from the earlier graph means they are all increasing as the engine degrades (8 and 13 being most strongly correlated) whereas sensors 7, 12, 20 and 21 show the opposite behaviour as a group, sensors 9 and 14 continue to be effective outlier columns which only correlate with each other.

I decided to test out several different regression models here to see how they modelled the data differently. I have decided to test the model first using all available data and then using only the last 125 datapoints of each engine, this is to avoid confusing the model with large stretches of very similar data where the engine has yet to degrade, which means no information about how much longer it has left is being signalled. 

I then used several scikit based models (Linear regression, Linear regression with PCA, Ridge Regression and Random forest) to try and predict the remaining turns based on the current turn without sensors 14 and 9 due to their inconsistency. I first used 5-fold cross validation to check the models on only the training data, getting the following data, the std was calculated with ddof=0 but ddof=1 may have been more appropriate given the small sample size:
                 model       mean       std      fold1      fold2      fold3      fold4      fold5
0           pureLinear  22.900115  1.457979  24.993081  22.799043  22.902990  20.445296  23.360163
1      ridgeRegression  22.900086  1.459862  24.997095  22.800665  22.901625  20.442716  23.358331
2  linearRegressionPCA  23.463256  1.534430  25.711958  23.488730  23.392046  20.898307  23.825240
3         randomForest  21.370104  1.171591  22.849805  21.384305  21.779717  19.257756  21.578940
Where each value is in cycles based on RMSE. But later changed the data used to be based on a sliding window average from each row to make random noise less of a factor, producing the following results:
[text](results/without_9_14/crossValidation.txt)
Maintaing the gaps between them roughly while showing the biggest improvement on random forest, it is possible I could get better results by using more tree branches or reducing the probability of each column being counted on a split to less than one to make high variance variables less significant but I deemed that beyond the scope of the project.
I then used this on the actual data recieveing:
[text](results/without_9_14/testResults.txt) without 9 and 14
[text](results/with_9_14/testResults.txt) with 9 and 14
Showing a increase of about 0.9 cycles between the average error on training and test data, possibly due to the fact is used only the last value rather than taking a mean over several hundred. randomForest was most strongly effected suggesting it was more accurate near the beginning than the others.

To ensure the differences seen here weren't just flukes of the test set specifically used I checked the difference between real and predicted values over several thousand randomly picked rows and took the minimum and maximum over the middle 95%, this got me the following:
Linear minus RF RMSE: 95% interval 2.07 to 2.51
Which suggests the difference between predictions based on linear and random forest regression have a consistent and significant difference between them, as shown by the difference between max and min varying over only 0.5 cycles for the full run. 

In order to ensure I made the right decision in removing sensors 9 and 14 I decided to run the cross validation with them, to get a clear succinct impression of how they effect prediction accuracy:
[text](results/with_9_14/crossValidation.txt)
It seems my initial assertion was wrong given we see a consistent decrease of 0.1 in all RMSE (Though this is too small to necessarily not just be random).

In order to make this data more understandable I decided to use matplot lib to plot it, producing the following graphs:
![alt text](results/with_9_14/lastCyclePredictions.png) with 9 and 14
![alt text](results/without_9_14/lastCyclePredictions.png) without 9 and 14
Also showing that the models are primarily flawed near the end of the database

In order to make the tables more readable I have switched to using markdown tables and restructured the README to refer to the relevant file when necessary.

 I also made some mistakes earlier in development which influenced my conclusions, these were in the error section for generating the error of different regression methods. Now it shows that both random forest and linear regression (Under both sensor sets) have a range which cover 0:
 [text](results/with_9_14/bootstrap.txt) with 9 and 14
 [text](results/without_9_14/bootstrap.txt) without 9 and 14
  suggesting that the difference is quality is inconsistent and may be worse or better depending on the value.