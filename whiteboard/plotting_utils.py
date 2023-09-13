import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib.patches as mpatches
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.visualization.scatter import Scatter
import numpy as np



def plot_vanilla(data_list, min_len):

    sns.set_style("whitegrid", {'axes.grid' : True,
                                'axes.edgecolor':'black'
    
                                })
    fig = plt.figure()
    plt.clf()
    ax = fig.gca()
    colors = ["red", "black", "green", "blue", "purple",  "darkcyan", "brown", "darkblue",]
    labels = ["DQN", "DDQN","Maxmin", "EnsembleDQN", "MaxminDQN"]
    color_patch = []
    for color, label, data in zip(colors, labels, data_list):
        sns.tsplot(time=range(min_len), data=data, color=color, ci=95)
        color_patch.append(mpatches.Patch(color=color, label=label))
    print(min_len)
    plt.xlim([0, min_len])
    plt.xlabel('Training Episodes $(\\times10^6)$', fontsize=22)
    plt.ylabel('Average return', fontsize=22)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title('Title', fontsize=14)
    ax = plt.gca()
    ax.set_xticks([10, 20, 30, 40, 50])
    ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])
    
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine()
    plt.tight_layout()
    plt.show()
    


    
def plot_equilibrium_curves_uniform_distr():
    import numpy as np
    import matplotlib.pyplot as plt
    #plt.rcParams["text.usetex"] = True
    # Generate x values; Note: We avoid x = 1 to prevent division by zero.
    x = np.linspace(0.001, 0.999, 4000)
    def fa(u,t):
        if 0.5 < u/t <= 1:
            return ((1+u/t)/2,(1-u/t)*2*t)
        elif u/t > 1:
            return (1,0)
        else:
            return (0.75,t)
    
    def fb(u,t):
        if 0 < 1-(u/(1-t)) <= 0.5:
            return ((1-(u/(1-t)))/2,(1-(u/(1-t)))*2*(1-t))
        elif 1-(u/(1-t)) < 0:
            return (0,0)
        else:
            return (0.25,1-t)
    
    
    # Compute the function values for each x
    u_bar = 0.2
    f_x2 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    g_x = x
    u_bar = 0.3
    f_x3 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    u_bar = 0.1
    f_x1 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    u_bar = 0.4
    f_x4 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    
    a = fa(u_bar,0.3)
    b = fb(u_bar,0.3)
    #for i,val in enumerate(x):
    #    print(val,f_x[i])
    # Plot the functions
    plt.plot(x, f_x1,'deepskyblue', alpha = 0.25,label='  ')
    plt.plot(x, f_x2, 'springgreen', alpha = 0.5, label='  ')
    plt.plot(x, f_x3,'darkorange', alpha = 0.75,label='  ')
    plt.plot(x, f_x4,'r', alpha = 1,label='  ')
    plt.plot(x, g_x, color='black',alpha=0.5,linestyle='dotted',linewidth=1)
    
    #plt.title('Plot of functions in range [0,1]')
    #plt.xlabel('$\theta^{t}$')
    #plt.ylabel('$\theta^{t}\'$')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1) # Adjust y-limits for better visualization
    plt.show()
    
def plot_equilibrium_curves_uniform_distr_sanc():
    import numpy as np
    import matplotlib.pyplot as plt
    #plt.rcParams["text.usetex"] = True
    # Generate x values; Note: We avoid x = 1 to prevent division by zero.
    x = np.linspace(0.001, 0.999, 4000)
    def fa(u,t):
        if 0.5 < u/t <= 1:
            return ((1+u/(t*0.5**(1-t)))/2,(1-u/(t*0.5**(1-t)))*2*t)
        elif u/t > 1:
            return (1,0)
        else:
            return (0.75,t)
    
    def fb(u,t):
        if 0 < 1-(u/(1-(t*0.5**(1-t)))) <= 0.5:
            return ((1-(u/(1-(t*0.5**(1-t)))))/2,(1-(u/(1-(t*0.5**(1-t)))))*2*(1-(t*0.5**(1-t))))
        elif 1-(u/(1-(t*0.5**(1-t)))) < 0:
            return (0,0)
        else:
            return (0.25,1-(t*0.5**(1-t)))
    
    
    # Compute the function values for each x
    u_bar = 0.2
    f_x2 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    g_x = x
    u_bar = 0.3
    f_x3 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    u_bar = 0.1
    f_x1 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    u_bar = 0.4
    f_x4 = [((fa(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fa(u_bar,val)[0]) + ((fb(u_bar,val)[1]/(fa(u_bar,val)[1]+fb(u_bar,val)[1]))*fb(u_bar,val)[0]) for val in x]
    
    a = fa(u_bar,0.3)
    b = fb(u_bar,0.3)
    #for i,val in enumerate(x):
    #    print(val,f_x[i])
    # Plot the functions
    plt.plot(x, f_x1,'deepskyblue', alpha = 0.25,label='  ')
    plt.plot(x, f_x2, 'springgreen', alpha = 0.5, label='  ')
    plt.plot(x, f_x3,'darkorange', alpha = 0.75,label='  ')
    plt.plot(x, f_x4,'r', alpha = 1,label='  ')
    plt.plot(x, g_x, color='black',alpha=0.5,linestyle='dotted',linewidth=1)
    
    #plt.title('Plot of functions in range [0,1]')
    #plt.xlabel('$\theta^{t}$')
    #plt.ylabel('$\theta^{t}\'$')
    plt.grid(True)
    plt.legend()
    plt.ylim(0, 1) # Adjust y-limits for better visualization
    plt.show()
    
plot_equilibrium_curves_uniform_distr_sanc()


    