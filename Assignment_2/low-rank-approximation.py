# r is r=1,2,...,200
rValues = [n for n in range(1, 201)]
# list to store Frobenius Norm error values
errList = []

# calculate the low rank approximation error for every rank in r=[1,200]
for r in rValues: 
    # calculate matrix products
    SVt_product = np.dot(np.diag(S[:r]), VTrans[:r,:])
    Xr = np.dot(U[:,:r], SVt_product) 
    # calculate difference between X and Xr
    diff = np.subtract(trainMinusAv, Xr) # change to train_data????
    # calculate Frobenius Norm of the difference
    errList.append(np.linalg.norm(diff, 'fro'))

# plot r against error
plt.plot(rValues, errList, label='Rank-r Approximation Error') 
plt.xlabel("Rank (r)") 
plt.ylabel("Approximation Error");

# save images for report
plt.savefig('Low_Rank.png')