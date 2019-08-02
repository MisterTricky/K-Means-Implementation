#K-Means clustering implementation

#Some hints on how to start have been added to this file.
#You will have to add more code that just the hints provided here for the full implementation.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# ====
# Define a function that computes the distance between two data points
def dist(xi,yi,xii,yii):
    answer = ((xii-xi)**2+(yii-yi)**2)
    return answer



# ====
# Define a function that reads data in from the csv files  HINT: http://docs.python.org/2/library/csv.html
def openCSV(filename):
    text=pd.read_csv(filename+'.csv')
    return text


# ====
# Write the initialisation procedure
dataset = openCSV('dataBoth')
X=dataset.iloc[:,[1,2]].values

#How many clusters?
clus=int(input('How many clusters are there?'))


# ====
# Implement the k-means algorithm, using appropriate looping

#fitting kmeans to the dataset
kmeans=KMeans(n_clusters=clus,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(X)
cenX=kmeans.cluster_centers_[:,0]
cenY=kmeans.cluster_centers_[:,1]

centroids=[]
cluster_1=[]
cluster_2=[]
cluster_3=[]
cluster_4=[]
cluster_5=[]


for i in range(clus):
    centroids.append([cenX[i],cenY[i]])
    
for i in range(len(X)):
    if(clus==1):
        cluster_1.append(dataset['Countries'][i]+", ")
    if(clus==2):
        dist_1=dist(X[i][0],X[i][1],centroids[0][0],centroids[0][1])
        dist_2=dist(X[i][0],X[i][1],centroids[1][0],centroids[1][1])

        if(dist_1<dist_2):
            cluster_1.append(dataset['Countries'][i]+", ")
        else:
            cluster_2.append(dataset['Countries'][i]+", ")
    elif(clus==3):
        dist_1=dist(X[i][0],X[i][1],centroids[0][0],centroids[0][1])
        dist_2=dist(X[i][0],X[i][1],centroids[1][0],centroids[1][1])
        dist_3=dist(X[i][0],X[i][1],centroids[2][0],centroids[2][1])
        
        if(dist_1<dist_2 and dist_1<dist_3):
            cluster_1.append(dataset['Countries'][i]+", ")
        elif(dist_2<dist_1 and dist_2<dist_3):
            cluster_2.append(dataset['Countries'][i]+", ")
        elif(dist_3<dist_1 and dist_3<dist_2):
            cluster_3.append(dataset['Countries'][i]+", ")
            
    elif(clus==4):
        dist_1=dist(X[i][0],X[i][1],centroids[0][0],centroids[0][1])
        dist_2=dist(X[i][0],X[i][1],centroids[1][0],centroids[1][1])
        dist_3=dist(X[i][0],X[i][1],centroids[2][0],centroids[2][1])
        dist_4=dist(X[i][0],X[i][1],centroids[3][0],centroids[3][1])
        
        if(dist_1<dist_2 and dist_1<dist_3 and dist_1<dist_4):
            cluster_1.append(dataset['Countries'][i]+", ")
        elif(dist_2<dist_1 and dist_2<dist_3 and dist_2<dist_4):
            cluster_2.append(dataset['Countries'][i]+", ")
        elif(dist_3<dist_1 and dist_3<dist_2 and dist_3<dist_4):
            cluster_3.append(dataset['Countries'][i]+", ")
        elif(dist_4<dist_1 and dist_4<dist_2 and dist_4<dist_3):
            cluster_4.append(dataset['Countries'][i]+", ")
            
    elif(clus==5):
        dist_1=dist(X[i][0],X[i][1],centroids[0][0],centroids[0][1])
        dist_2=dist(X[i][0],X[i][1],centroids[1][0],centroids[1][1])
        dist_3=dist(X[i][0],X[i][1],centroids[2][0],centroids[2][1])
        dist_4=dist(X[i][0],X[i][1],centroids[3][0],centroids[3][1])
        dist_5=dist(X[i][0],X[i][1],centroids[4][0],centroids[4][1])
        
        if(dist_1<dist_2 and dist_1<dist_3 and dist_1<dist_4 and dist_1<dist_5):
            cluster_1.append(dataset['Countries'][i]+", ")
        elif(dist_2<dist_1 and dist_2<dist_3 and dist_1<dist_4 and dist_2<dist_5):
            cluster_2.append(dataset['Countries'][i]+", ")
        elif(dist_3<dist_1 and dist_3<dist_2 and dist_3<dist_4 and dist_3<dist_5):
            cluster_3.append(dataset['Countries'][i]+", ")
        elif(dist_4<dist_1 and dist_4<dist_2 and dist_4<dist_3 and dist_4<dist_5):
            cluster_4.append(dataset['Countries'][i]+", ")
        elif(dist_5<dist_1 and dist_5<dist_2 and dist_5<dist_3 and dist_5<dist_4):
            cluster_5.append(dataset['Countries'][i]+", ")
    


# ====
# Print out the results


if(clus==1):
    print('There are '+str(len(cluster_1))+ ' Countries in Cluster 1.')
    print('Cluster 1 Countries:\n'+'\n'.join(str(x)for x in cluster_1))
    print('The mean life expectancy is:'+ str(centroids[0][1])+' \nMean Birth rate:'+ str(centroids[0][0]))
elif(clus==2):
    print('There are '+str(len(cluster_1))+ ' Countries in Cluster 1.')
    print('Cluster 1 Countries:\n'+'\n'.join(str(x)for x in cluster_1))
    print('The mean life expectancy is:'+ str(centroids[0][1])+' \nMean Birth rate:'+ str(centroids[0][0]))
    print("\n")
    print('There are '+str(len(cluster_2))+ ' Countries in Cluster 2.')
    print('Cluster 2 Countries:\n'+'\n'.join(str(x)for x in cluster_2))
    print('The mean life expectancy is:'+ str(centroids[1][1])+' \nMean Birth rate:'+ str(centroids[1][0]))
    print("\n")
elif(clus==3):
    print('There are '+str(len(cluster_1))+ ' Countries in Cluster 1.')
    print('Cluster 1 Countries:\n'+'\n'.join(str(x)for x in cluster_1))
    print('The mean life expectancy is:'+ str(centroids[0][1])+' \nMean Birth rate:'+ str(centroids[0][0]))
    print("\n")
    print('There are '+str(len(cluster_2))+ ' Countries in Cluster 2.')
    print('Cluster 2 Countries:\n'+'\n'.join(str(x)for x in cluster_2))
    print('The mean life expectancy is:'+ str(centroids[1][1])+' \nMean Birth rate:'+ str(centroids[1][0]))
    print("\n") 
    print('There are '+str(len(cluster_3))+ ' Countries in Cluster 3.')
    print('Cluster 3 Countries:\n'+'\n'.join(str(x)for x in cluster_3))
    print('The mean life expectancy is:'+ str(centroids[2][1])+' \nMean Birth rate:'+ str(centroids[2][0]))
    print("\n")
elif(clus==4):
    print('There are '+str(len(cluster_1))+ ' Countries in Cluster 1.')
    print('Cluster 1 Countries:\n'+'\n'.join(str(x)for x in cluster_1))
    print('The mean life expectancy is:'+ str(centroids[0][1])+' \nMean Birth rate:'+ str(centroids[0][0]))
    print("\n")
    print('There are '+str(len(cluster_2))+ ' Countries in Cluster 2.')
    print('Cluster 2 Countries:\n'+'\n'.join(str(x)for x in cluster_2))
    print('The mean life expectancy is:'+ str(centroids[1][1])+' \nMean Birth rate:'+ str(centroids[1][0]))
    print("\n") 
    print('There are '+str(len(cluster_3))+ ' Countries in Cluster 3.')
    print('Cluster 3 Countries:\n'+'\n'.join(str(x)for x in cluster_3))
    print('The mean life expectancy is:'+ str(centroids[2][1])+' \nMean Birth rate:'+ str(centroids[2][0]))
    print("\n")
    print('There are '+str(len(cluster_4))+ ' Countries in Cluster 4.')
    print('Cluster 4 Countries:\n'+'\n'.join(str(x)for x in cluster_4))
    print('The mean life expectancy is:'+ str(centroids[3][1])+' \nMean Birth rate:'+ str(centroids[3][0]))
    print("\n")
elif(clus==5):
    print('There are '+str(len(cluster_1))+ ' Countries in Cluster 1.')
    print('Cluster 1 Countries:\n'+'\n'.join(str(x)for x in cluster_1))
    print('The mean life expectancy is:'+ str(centroids[0][1])+' \nMean Birth rate:'+ str(centroids[0][0]))
    print("\n")
    print('There are '+str(len(cluster_2))+ ' Countries in Cluster 2.')
    print('Cluster 2 Countries:\n'+'\n'.join(str(x)for x in cluster_2))
    print('The mean life expectancy is:'+ str(centroids[1][1])+' \nMean Birth rate:'+ str(centroids[1][0]))
    print("\n") 
    print('There are '+str(len(cluster_3))+ ' Countries in Cluster 3.')
    print('Cluster 3 Countries:\n'+'\n'.join(str(x)for x in cluster_3))
    print('The mean life expectancy is:'+ str(centroids[2][1])+' \nMean Birth rate:'+ str(centroids[2][0]))
    print("\n")
    print('There are '+str(len(cluster_4))+ ' Countries in Cluster 4.')
    print('Cluster 4 Countries:\n'+'\n'.join(str(x)for x in cluster_4))
    print('The mean life expectancy is:'+ str(centroids[3][1])+' \nMean Birth rate:'+ str(centroids[3][0]))
    print("\n")
    print('There are '+str(len(cluster_5))+ ' Countries in Cluster 5.')
    print('Cluster 5 Countries:\n'+'\n'.join(str(x)for x in cluster_5))
    print('The mean life expectancy is:'+ str(centroids[4][1])+' \nMean Birth rate:'+ str(centroids[4][0]))
    print("\n")

#visualize the clustering algorithm
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=10,c='r')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=10,c='y')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=10,c='g')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=10,c='b')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=10,c='cyan')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,marker='X',color='black')
plt.title('Clusters of data2008')
plt.ylabel('LifeExpectancy')
plt.xlabel('BirthRate(Per1000)')
plt.show()



















