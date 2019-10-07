# Get training and test features for r = 10
train10, test10 = generateFMatrix(trainMinusAv, testMinusAv, 10) 

# create a logistic regression model
lr = LogisticRegression(multi_class = 'ovr').fit(train10, train_labels)
# get predictions
preds = lr.predict(test10) 

# calculate and print model accuracy
accuracy = metrics.accuracy_score(preds, test_labels) * 100
print('The accuracy of our model for r = 10 is', accuracy, '%')

accuracies = [] # list to store accuracies for r = 1,....,200

for r in rValues:
    # get matrices
    trainF, testF = generateFMatrix(trainMinusAv, testMinusAv, r) 
    # create logistics regression model
    lr = LogisticRegression(multi_class = 'ovr').fit(trainF, train_labels)
    # predict
    preds = lr.predict(testF) 
    # find accuracy
    accuracies.append(metrics.accuracy_score(test_labels, preds)*100) 

# create accuracy plot, plotting r against accuracy
plt.plot(rValues, accuracies, label='Classification Accuracy') 
plt.xlabel("Rank (r)") 
plt.ylabel("Rank-r Accuracy (%)"); 

# save images for report
plt.savefig('Accuracy.png')