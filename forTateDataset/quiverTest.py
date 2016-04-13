
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import interactive
from scipy.spatial import distance
from matplotlib.pyplot import cm
import matplotlib.colors as colors

n_columns, n_rows = 200, 120

X1 = [43, 51, 31, 5, 66, 22, 194, 66, 20, 45]
Y1 = [76, 54, 35, 3, 69, 16, 100, 46, 53, 101]
X2 = [19, 46, 48, 36, 65, 88, 27, 150, 59, 8]
Y2 = [46, 63, 35, 83, 61, 47, 107, 69, 77, 30]

##import random
##fig, ax = plt.subplots()
##plt.scatter(X1,Y1)
##labels = [str(x) for x in range(len(X1[:10]))]
##for label, x, y in zip(labels, X1, Y1):
##    plt.annotate(label, xy = (x, y), xytext = (-20*random.random(), 20*random.random()), textcoords = 'offset points', ha = 'right', va = 'bottom',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
##        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
##plt.xlim(0,n_columns)
##plt.ylim(0,n_rows) 
##ax.invert_yaxis() 
##plt.title('test 1')
##mng = plt.get_current_fig_manager()
##mng.window.state('zoomed')
##interactive(True)
##plt.show()            
##fig.savefig('./test1.png',bbox_inches='tight')
##plt.close()
##interactive(False)
##
##fig, ax = plt.subplots()
##plt.scatter(X2,Y2)
##labels = [str(x) for x in range(len(X2[:10]))]
##for label, x, y in zip(labels, X2, Y2):
##    plt.annotate(label, xy = (x, y), xytext = (-20*random.random(), 20*random.random()), textcoords = 'offset points', ha = 'right', va = 'bottom',bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
##        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
##plt.xlim(0,n_columns)
##plt.ylim(0,n_rows) 
##ax.invert_yaxis() 
##plt.title('test 2')
##mng = plt.get_current_fig_manager()
##mng.window.state('zoomed')
##interactive(True)
##plt.show()            
##fig.savefig('./test2.png',bbox_inches='tight')
##plt.close()
##interactive(False)

fig, ax = plt.subplots()
a = np.array([2, 3, 5])
b = np.array([1, 1, 0])
c = a+b;
starts = np.zeros((3,3))
ends = np.matrix([a,b,c])
Q = plt.quiver(starts[:,0], starts[:,1], starts[:,2], ends[:,0], ends[:,1], ends[:,2], scale=0)
plt.show()


##fig, ax = plt.subplots()
##X1,Y1 = np.mgrid[-n_columns/2:n_columns/2:1j,-n_columns/2:n_columns/2:1j]#np.array(X1),np.array(Y1))
##Q = plt.streamplot(X1,Y1,np.array(X2),np.array(Y2))
##plt.xlim(0,n_columns)
##plt.ylim(0,n_rows)     
### ax.set_xticks(range(0, len(nodes)))#, minor=False)
### ax.xaxis.tick_top()
### ax.set_xticklabels(nodes, minor=False, fontsize = 7, rotation = 90)
### ax.set_yticklabels(list(reversed(nodes)), minor=False, fontsize = 7) 
##ax.invert_yaxis() 
##plt.title('quiver plot')
##mng = plt.get_current_fig_manager()
##mng.window.state('zoomed')
##interactive(True)
##plt.show()
##fig.savefig('./quivertest.png',bbox_inches='tight')
##plt.close()
##interactive(False)
