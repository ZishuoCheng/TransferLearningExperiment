import numpy as np
import matplotlib.pyplot as plt
from pylab import *

def plot_curve(data,save_path=None):
    # print(len(result),len(result[0]))
    font = {
    #'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 30,
    }
    colors = ['r', 'b', 'g', 'm', 'k']
    linestyle = ['-', '--', '-.', ':', ":"]
    markers = ['s', 'o', 'D', '^', '*']
    
    
    rules = [ "DTW", "TW", "TL", "MRL", "Random"]
    # gtypes = ["","-Malicious"]
    
    fig = plt.figure(figsize=(15,6))

    title = save_path
    total_epoch = 600
    width, height = 300, 250
    legend_fontsize = 10
    scale_distance = 48.8
    dpi = 40
    figsize = width / float(dpi), height / float(dpi)
    # fig = plt.figure(figsize=(9,6))
    # plt.subplot(1,2,net_index+1)
    plt.xlim(0, 600)
    plt.ylim(np.min(data), 150)
    interval_y = 0.1
    interval_x = 100
    plt.xticks(np.arange(0, total_epoch + interval_x, interval_x))
    # plt.yticks(np.arange(0, 1 + interval_y, interval_y))
    plt.grid()
    # plt.title(title, font)
    plt.xlabel('Training Epoch', font)
    plt.ylabel(title, font)
    plt.tick_params(labelsize=25)
    for rule_index,value in enumerate(data):
        x_axis = range(len(value))                
        plt.plot(x_axis,value,color=colors[rule_index], linestyle=linestyle[rule_index], lw=1.5) 
        # plt.plot(x_axis,value,color=colors[rule_index], lw=1.5)     
        scatter_x = [i*100 for i in range(6)] + [599]
        scatter_y = [value[i] for i in scatter_x]
        plt.scatter(scatter_x,scatter_y,marker=markers[rule_index],c=colors[rule_index],edgecolors='',s=50)
        plt.plot(range(1),scatter_y[0],linestyle[rule_index]+colors[rule_index]+ markers[rule_index],label=rules[rule_index],lw=1.5,markerfacecolor=colors[rule_index],markersize=6)
        # legend()
    # plt.plot(range(1),scatter_y[0],linestyle[-1]+colors[-1]+ markers[-1],label=rules[-1],lw=1.5,markerfacecolor=colors[-1],markersize=0.001)
    legend(bbox_to_anchor=(0.8, 1.22),ncol=5,fontsize=24)
    if save_path != None:           
        plt.savefig(save_path+'.png', dpi=800, bbox_inches='tight')
        plt.savefig(save_path+'.eps', dpi=800, bbox_inches='tight')
        # plt.savefig("Line_"+net+'.png', dpi=800, bbox_inches='tight')
    plt.show()
    plt.close(fig)

path = ['DTW.txt', 'TW.txt', 'TL.txt', 'RL.txt', 'Random.txt']
results_1 = []
results_2 = []
results_3 = []
for index,p in enumerate(path):
    results = np.loadtxt(p)
    if index != 4:
        r1 = results[:,1]
        results_1.append(r1)

        r2 = results[:,2]
        results_2.append(r2)

        r3 = results[:,3]
        results_3.append(r3)

# plot_curve(results_1, save_path = "Total Hits")
# plot_curve(results_2, save_path = "Average Step Cost")
plot_curve(results_3, save_path = "Average Hits")





