from scipy import stats


rvs1 = stats.norm.rvs(loc=5,scale=10,size=500)
rvs2 = stats.norm.rvs(loc=5,scale=10,size=500)
stats.ttest_ind(rvs1,rvs2)

print(rvs1)



list1 = [5,23312,4,123,23,412,41,25,14,123,12]
list2 = [5,2312,44,13,21,412,41,25,14,123,12]

print(stats.ttest_ind(list1,list2))