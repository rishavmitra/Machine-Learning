import matplotlib.pyplot as plt
import seaborn as sns

def boxplt(value):
    # (Since Matplotlib doesn't have the ability to ignore missing values we use the seaborn library as it can ignore
    # missing values)
    sns.boxplot(value)
    plt.show()
def hist_gram(value):
    plt.hist(value,bins=10,color='green')
    plt.show()
def main6_graph(value):
    plt.figure(dpi=100)
    plt.ticklabel_format(style='plain')
    k = range(0, len(value))
    plt.scatter(k,value['Sale_Price'].sort_values(), color='red', label='Actual sale price')
    plt.plot(k,value['Mean_Sales'].sort_values(), color='green', label='Mean sale price')
    plt.xlabel('Fitted Points(Ascending)')
    plt.ylabel('Sale Price')
    plt.legend()
    plt.show()
def main6_graph2(value):
    gradewise_list=[]
    for i in range(1,11):
        k = value['Sale_Price'][value['Overall Grade'] == i]
        gradewise_list.append(k)
    classwise_list=[]
    for i in range(1,11):
        k = value['Sale_Price'][value['Overall Grade'] == i]
        classwise_list.append(k)
    plt.figure(dpi = 120, figsize =(15,9))

    ### Plotting Sale Price Grade wise ######
    # z variable is for x -axis
    z = 0
    for i in range(1,11):
        #defining x-axis using z
        points = [k for k in range(z,z+len(classwise_list[i-1]))]
        #plotting
        plt.scatter(points,classwise_list[i-1].sort_values(),label=('houses with overall grade',i),s = 4)
        #plotting gradewise mean
        plt.scatter(points,[classwise_list[i-1].mean() for q in range(len(classwise_list[i-1]))],color='pink',s=6)
        z = max(points)+1

    ##### Plotting Overall mean ###
    plt.scatter([q for q in range(0,z)],value['Mean_Sales'],color='red',label='Overall Mean',s = 6)
    plt.xlabel('Fitted Points(Ascending)')
    plt.ylabel('Sale Price')
    plt.title('Overall Mean')
    plt.legend(loc = 4)
    plt.show()
def main6_residual_graph(values,values2,values3):
    k = range(0,len(values))
    l = [0 for i in range(len(values))]

    plt.scatter(k,values2,color='red',label='Residual',s=2)
    plt.plot(k,l,color='green',label='mean regression',linewidth=2)
    plt.xlabel('Fitted Points')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()

    plt.scatter(k, values3, color='red', label='Residual', s=2)
    plt.plot(k, l, color='green', label='mean regression', linewidth=2)
    plt.xlabel('Fitted Points')
    plt.ylabel('Residuals')
    plt.legend()
    plt.show()




