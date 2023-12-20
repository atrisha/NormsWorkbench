'''
Created on 11 Sept 2023

@author: Atrisha
'''

import matplotlib.pyplot as plt
import numpy as np
import math

def old_plot_with_o_minus_i():
    # Create a grid of x and y values between 0 and 1
    x = np.linspace(0.5, 1, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    n = 0.6
    a,ubar = 0.5,0.15
    # Calculate the function values for each (x, y) pair
    opt_rhet = lambda o : -0.9101*o**2 + 1.9646*o -0.4356
    '''
    Z4 = n * (X * Y**(1 - 0.8) -a*Y)
    Z3 = n * (X * Y**(1 - 0.6)-a*Y)
    Z2 = n * (X * Y**(1 - 0.4)-a*Y)
    Z1 = n *  (X * Y**(1 - 0.2)-a*Y)
    '''
    Z4 = n * (X * Y**(1 - 0.8) -a*Y)
    Z3 = n * (X * Y**(1 - 0.6)-a*Y)
    Z2 = n * (X * Y**(1 - 0.4)-a*Y)
    Z1 = n *  (X * Y**(1 - 0.2)-a*Y)
    
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    # Create a 2D heatmap plot with alpha values
    fig, axes = plt.subplots(1, 4, figsize=(8, 6))
    
    axes[0].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[0].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[0].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[0].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[0].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.8*y,1/0.8),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[0].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.2*y,1/0.2),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[0].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.4*y,1/0.4),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[0].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.6*y,1/0.6),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    n=0.6
    a,ubar=0.7,0.15
    Z4 = n * (X * Y**(1 - 0.8) -a*Y)
    Z3 = n * (X * Y**(1 - 0.6)-a*Y)
    Z2 = n * (X * Y**(1 - 0.4)-a*Y)
    Z1 = n *  (X * Y**(1 - 0.2)-a*Y)
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    
    
    axes[1].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[1].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[1].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[1].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[1].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.8*y,1/0.8),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[1].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.2*y,1/0.2),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[1].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.4*y,1/0.4),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[1].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.6*y,1/0.6),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    n = 0.6# Calculate the function values for each (x, y) pair
    a,ubar = 0.5,0.25
    Z4 = n * (X * Y**(1 - 0.8) -a*Y)
    Z3 = n * (X * Y**(1 - 0.6)-a*Y)
    Z2 = n * (X * Y**(1 - 0.4)-a*Y)
    Z1 = n *  (X * Y**(1 - 0.2)-a*Y)
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    
    
    axes[2].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[2].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[2].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[2].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[2].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.8*y,1/0.8),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[2].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.2*y,1/0.2),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[2].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.4*y,1/0.4),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[2].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.6*y,1/0.6),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    n=0.4
    a,ubar = 0.5,0.15
    
    Z4 = n * (X * Y**(1 - 0.8) -a*Y)
    Z3 = n * (X * Y**(1 - 0.6)-a*Y)
    Z2 = n * (X * Y**(1 - 0.4)-a*Y)
    Z1 = n *  (X * Y**(1 - 0.2)-a*Y)
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    
    
    axes[3].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[3].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[3].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[3].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[3].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.8*y,1/0.8),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[3].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.2*y,1/0.2),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[3].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.4*y,1/0.4),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[3].plot(np.linspace(0.5,1,100),[min(math.pow((1/a)*y-(1/a)*0.6*y,1/0.6),1) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    # Add labels
    plt.tight_layout()
    fig.text(0.5, 0, 'Opinion degree', ha='center')
    fig.text(0.04, 0.5, 'Sanctioning intensity', va='center', rotation='vertical')
    
    
    # Set axis limits to [0, 1]
    plt.xlim(0.5, 1)
    axes[0].set_ylim(0, 1.01)
    axes[1].set_ylim(0, 1.01)
    axes[2].set_ylim(0, 1.01)
    axes[3].set_ylim(0, 1.01)
    plt.tight_layout()
    # Show the plot
    #plt.suptitle('Expression and silence map for opinion degree and sanctioning intensity', x=0.5, y=1,)
    plt.show()

def plot_new_chart_with_phi_minus_i():
    # Create a grid of x and y values between 0 and 1
    x = np.linspace(0.5, 1, 1000)
    y = np.linspace(0, 1, 1000)
    X, Y = np.meshgrid(x, y)
    n = 0.6
    a,ubar = 0.5,0.1
    # Calculate the function values for each (x, y) pair
    # These are estimated based on finding the root of the equation ((o * (1 - x) / 0.5)**(1 / x)) - x and then fitting a second order polynomial on values of o.
    opt_rhet = lambda o : -0.9101*o**2 + 1.9646*o -0.4356
    br = lambda o,x: ((o*(1-x))/0.5)**(1/x)
    '''
    Z4 = n * (X * Y**(1 - 0.8) -a*Y)
    Z3 = n * (X * Y**(1 - 0.6)-a*Y)
    Z2 = n * (X * Y**(1 - 0.4)-a*Y)
    Z1 = n *  (X * Y**(1 - 0.2)-a*Y)
    '''
    Z4 = n * (X * Y**(1 - opt_rhet(0.8)) -a*Y)
    Z3 = n * (X * Y**(1 - opt_rhet(0.6))-a*Y)
    Z2 = n * (X * Y**(1 - opt_rhet(0.4))-a*Y)
    Z1 = n *  (X * Y**(1 - opt_rhet(0.2))-a*Y)
    
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    # Create a 2D heatmap plot with alpha values
    fig, axes = plt.subplots(1, 4, figsize=(8, 6))
    
    axes[0].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[0].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[0].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[0].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[0].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.8))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[0].plot(np.linspace(0.5,1,100),[min(1,br(y,np.clip(opt_rhet(0.2),0,1)))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[0].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.4))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[0].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.6))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    n=0.6
    a,ubar=0.7,0.1
    Z4 = n * (X * Y**(1 - opt_rhet(0.8)) -a*Y)
    Z3 = n * (X * Y**(1 - opt_rhet(0.6))-a*Y)
    Z2 = n * (X * Y**(1 - opt_rhet(0.4))-a*Y)
    Z1 = n *  (X * Y**(1 - opt_rhet(0.2))-a*Y)
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    
    
    axes[1].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[1].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[1].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[1].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[1].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.8))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[1].plot(np.linspace(0.5,1,100),[min(1,br(y,np.clip(opt_rhet(0.2),0,1)))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[1].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.4))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[1].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.6))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    n = 0.6# Calculate the function values for each (x, y) pair
    a,ubar = 0.5,0.2
    Z4 = n * (X * Y**(1 - opt_rhet(0.8)) -a*Y)
    Z3 = n * (X * Y**(1 - opt_rhet(0.6))-a*Y)
    Z2 = n * (X * Y**(1 - opt_rhet(0.4))-a*Y)
    Z1 = n *  (X * Y**(1 - opt_rhet(0.2))-a*Y)
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    
    
    axes[2].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[2].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[2].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[2].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[2].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.8))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[2].plot(np.linspace(0.5,1,100),[min(1,br(y,np.clip(opt_rhet(0.2),0,1)))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[2].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.4))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[2].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.6))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    n=0.4
    a,ubar = 0.5,0.1
    
    Z4 = n * (X * Y**(1 - opt_rhet(0.8)) -a*Y)
    Z3 = n * (X * Y**(1 - opt_rhet(0.6))-a*Y)
    Z2 = n * (X * Y**(1 - opt_rhet(0.4))-a*Y)
    Z1 = n *  (X * Y**(1 - opt_rhet(0.2))-a*Y)
    mask1 = Z1 > ubar
    mask2 = Z2 > ubar
    mask3 = Z3 > ubar
    mask4 = Z4 > ubar
    
    
    axes[3].imshow(mask1, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.8)
    axes[3].imshow(mask2, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.6)
    axes[3].imshow(mask3, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.4)
    axes[3].imshow(mask4, extent=[0.5, 1, 0, 1], cmap='binary', origin='lower', alpha=0.2)
    
    axes[3].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.8))))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.3)
    axes[3].plot(np.linspace(0.5,1,100),[min(1,br(y,np.clip(opt_rhet(0.2),0,1)))  for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.9)
    axes[3].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.4)))) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.7)
    axes[3].plot(np.linspace(0.5,1,100),[min(1,br(y,min(1,opt_rhet(0.6)))) for y in np.linspace(0.5,1,100)],color='red',linestyle='--', alpha=0.5)
    
    # Add labels
    plt.tight_layout()
    fig.text(0.5, 0, 'Opinion degree', ha='center')
    fig.text(0.04, 0.5, 'Sanctioning intensity', va='center', rotation='vertical')
    
    
    # Set axis limits to [0, 1]
    plt.xlim(0.5, 1)
    axes[0].set_ylim(0, 1.01)
    axes[1].set_ylim(0, 1.01)
    axes[2].set_ylim(0, 1.01)
    axes[3].set_ylim(0, 1.01)
    plt.tight_layout()
    # Show the plot
    #plt.suptitle('Expression and silence map for opinion degree and sanctioning intensity', x=0.5, y=1,)
    plt.show()
    
plot_new_chart_with_phi_minus_i()
opt_rhet = lambda o : -0.9101*o**2 + 1.9646*o -0.4356
br = lambda o,x: ((o*(1-x))/0.5)**(1/x)
print(np.clip(opt_rhet(0.2),0,1))
