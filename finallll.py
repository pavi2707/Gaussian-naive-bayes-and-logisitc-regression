import pandas as pd 
import numpy as np 
import math
from matplotlib import pyplot as plt




def LogisticRegression(traindata):
	
	fst_col = np.ones((len(traindata),1))
	traindata = np.hstack((fst_col,traindata))
	
	weights = np.random.rand(5,1)*0.01	

	for i in range(10000):	
		#weight calculating and updating
		w=np.dot(traindata[:,:5],weights)
		y_hat=(1/(1+np.exp(-w)))
		t=traindata[:,-1].reshape(len(traindata),1)
		tdata_use=traindata[:,:5].reshape(len(traindata),5)
		error=t-y_hat
		dw=(np.dot(np.transpose(error),tdata_use))/len(traindata)
		weights=weights+0.01*(np.transpose(dw))
	#prediction	
	testing=np.dot(testdata[:,:5],weights)
	prediction=(1/(1+np.exp(-testing)))
	finalpred=[]
	for i in range(len(prediction)):
		finalpred.append(1 if prediction[i]>0.5 else 0)
	
	finalpred = np.array(finalpred)
	comp = testdata[:,-1]-finalpred
	cnt=0
	for i in range(len(comp)):
		if comp[i] ==0:
		 cnt = cnt + 1 
	accuracy=cnt/len(comp)*100
	#print(accuracy)
	return accuracy


def GaussianNaiveBayes(traindata):
#Splitting the data classvise
	t_zerod = traindata[traindata['lable'] == 0]
	t_oned = traindata[traindata['lable'] == 1]
#prior for 0 and 1
	priorone = len(t_oned)/(len(traindata))
	priorzero = len(t_zerod)/(len(traindata))



	#For class zero mean and SD
	t_zerodmean = []

	for i in range(len(columns)):
		tmean = t_zerod[columns[i]].mean()
		t_zerodmean.append(tmean)

	t_zerodSD = []
	for i in range(len(columns)):
		TempSD = t_zerod[columns[i]].std()
		t_zerodSD.append(TempSD)

	#For class zero mean and SD
	t_onedmean = []
	for i in range(len(columns)):
		tmean = t_oned[columns[i]].mean()
		t_onedmean.append(tmean)

	t_onedSD = []
	for i in range(len(columns)):
		TempSD = t_oned[columns[i]].std()
		t_onedSD.append(TempSD)

	#likelihood 
	def calculateProbability(x, mean, stdev):
		exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
		return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent


	#likelihood for zero
	probability = 1
	prob_zero = []
	for i in range(len(testdata)):
		probability = 1
		for j in range(len(columns)):
			c_prob = calculateProbability(testdata.iloc[i][j], t_zerodmean[j], t_zerodSD[j])
			probability = probability * c_prob
		prob_zero.append(probability)

	#likelihood for 1
	prob_one = []
	for i in range(len(testdata)):
		probability = 1
		for j in range(len(columns)):
			c_prob = calculateProbability(testdata.iloc[i][j], t_onedmean[j], t_onedSD[j])
			probability = probability * c_prob
		prob_one.append(probability)

	#final p(y=1|x)
	final_prob = []
	for i in range(len(testdata)):
		num = prob_one[i] * priorone
		denm = (prob_zero[i] * priorzero) + num
		final_prob.append(num/denm)

	#final p(y=0|x)
	final_probz = []
	for i in range(len(testdata)):
		num = prob_zero[i] * priorzero
		denm = (prob_one[i] * priorone) + num
		final_probz.append(num/denm)

	finalprob = []
	for i in  range(len(testdata)):
		if(final_prob[i] > final_probz[i]):
			finalprob.append("1")
		else:
			finalprob.append("0")

	out = [int(labelCol[i])==int(finalprob[i]) for i in range(len(labelCol))]

	return(np.mean(out)*100)

df=pd.read_csv('data.txt',sep=",",header=None)
train=df.sample(frac=1,random_state=200)
train.columns = ['feature1','feature2','feature3','feature4','lable']
lb=train['lable']
traindata = train.head(int(len(train)*(2/3)))
testdata = train.tail(int(len(train)*(1/3)))
columns=list(train)
columns.remove('lable')

#print(testdata)
labelCol = list(testdata['lable'])
del testdata['lable']





fractions = [0.01, 0.02, 0.05, 0.1, 0.625, 1]

finalaverages = []
for fraction in fractions:
	print("For  GNB fraction ",fraction)
	
	for i in range(5):
		averages = []
		traindataSubset = traindata.sample(frac=fraction,random_state=200)
		tempavg = GaussianNaiveBayes(traindataSubset)
		averages.append(tempavg)
	
	print(np.array(averages).mean())
	finalaverages.append(sum(averages)/len(averages))
testdata=np.array(testdata)



df=pd.read_csv('data.txt',sep=",",header=None)
train=df.sample(frac=1,random_state=200)
train.columns = ['feature1','feature2','feature3','feature4','lable']
#splitting the test and train data
traindata = train.head(int(len(train)*(2/3)))
testdata = train.tail(int(len(train)*(1/3)))
testdata=np.array(testdata)

first_col = np.ones((len(testdata),1))
testdata=np.hstack((first_col,testdata))
averages=[]
frac=[0.01, 0.02, 0.05, 0.1, 0.625, 1]
for i in frac:
	acc =0
	for j in range(5):
		train_data = traindata.iloc[:int(i*len(traindata))]
		train_data = np.array(train_data)
		acc += LogisticRegression(train_data)
	print("average accuracy for logistic regression", acc/5)
	averages.append(acc/5)




fig,b = plt.subplots()
plt.title("Learning curve")
plt.xlabel("Size Ratio")
plt.ylabel(" Average  Accuracy Results")
b.plot(fractions,finalaverages,'*-',label = " Gaussian Naive Bayes")
b.plot(frac,averages,'*-',label = " Logistic Regression")
b.legend()
plt.show()


ts1 = traindata.head(int(len(traindata)*(1/2)))
ts2 = traindata.tail(int(len(traindata)*(1/2)))
ts3 = train.tail(int(len(train)*(1/3)))
print("ts1",ts1)
print("ts2",ts2)
print("ts3",ts3)
t_one1 = ts1[ts1['lable'] == 1]
t_one2 = ts2[ts2['lable'] == 1]
t_one1=np.array(t_one1)
t_one2=np.array(t_one2)
finalmean=[]

finalvar=[]

newtrain1=np.vstack((t_one1,t_one2))
print("new train",newtrain1)
for j in range(newtrain1.shape[1]-1):
	finalmean.append(sum(newtrain1[j])/len(newtrain1))
	t_sum=0
	nt=newtrain1[:,j]
	for i in range(len(newtrain1)):
		t_sum = t_sum+ pow(nt[i],2)
	finalvar.append(math.sqrt(t_sum/len(newtrain1)))
#finalmean=np.array(finalmean)
#finalvar= np.array(finalvar)
print("mean, var",finalmean, finalvar)
#print(finalmean.shape)
data=[]
for k in range(len(finalmean)):
	data.append(np.random.normal(finalmean[k],finalvar[k],400))


data= np.array(data)
data=np.transpose(data)

newmean=[]
newvar=[]
for j in range(data.shape[1]):
	newmean.append(sum(data[j])/len(data))
	t_sum=0
	nt=data[:,j]
	for i in range(len(data)):
		t_sum = t_sum+ pow(nt[i],2)
	newvar.append(t_sum/len(data))
print("mean, var",newmean, newvar)
