import numpy as np
import matplotlib.pyplot as plt

def a(n):
    """ returns n**n/n! 
    """
    c = 1.
    for ii in xrange(n-1):
        c *= (n/(ii+1.))
    return c

#def term(n,x):
#    """ returns (-1)**n * x**(2*n)/(2n+1)!
#    """
#    tn = (a(2*n)*(0.5*x/n)**(2*n)*(-1)**n)/(2*n+1)
#    return tn

def term(n,x):
    """ returns (-1)**n * x**(2n)/(2n+1)!
    """
    tn = ((-1)**n)
    for ii in xrange(2*n):
      tn *= (x/(ii+1.))
    tn *= 1./(2*n+1)
    return tn

def mysinc(x):
    return np.sin(x)/x

def plot():
    fig,ax = plt.subplots(1)
    x = np.linspace(0,60,3001)
    ax.plot(x,(mysinc(x)),'y--',lw=4,label='actual function')
    ax.plot(x,np.zeros(len(x)),'y:')
    colors = ['b','g','r','k','c','m','y']
    N = np.arange(30,60,4)[:7] #increasing sequence
    n_old = -1
    y = 0
    for n in N:
      for nn in np.arange(n_old,n,1):
        y += term(nn+1,x)
      n_old = n
      ax.plot(x,(y),'--',color=colors.pop(),label='n='+str(n))
    ax.legend().draggable()
    ax.set_ylim(-0.5,1.1)
    #ax.semilogy()
    plt.show()

if __name__=='__main__':
    plot()


    
