import matplotlib 
matplotlib.use("Agg")
import multiprocessing
from matplotlib import pyplot as plt 
plt.rcParams['text.usetex']=True
import scipy 
import numpy as np 
import itertools
import tqdm
import random
#@np.vectorize
def v0sinalpha(t,delta , a,b,v0,theta=0):
    return v0*np.sin(theta+np.arctan2(a*np.tan(delta(t)),b))

def ydot(t,y,delta,a,b,v0):
    #return v0*np.sin(np.arctan2(a*np.tan(delta(t)),b)+y[1])
    return  v0sinalpha(t,delta,a,b,v0,theta=y[1])

def thetadot(t,y,delta,a,b,v0):
    return v0sinalpha(t,delta,a,b,v0)/a

#@np.vectorize
def delta(t):
    return -0.5* np.pi * np.sin(2*np.pi *0.1*t)

def fdot(t,y,delta,a,b,v0):
    return [ydot(t,y,delta,v0,a,b),thetadot(t,y,delta,a,b,v0)] 

def fdotprod(t,y,delta,a,b,v0):
    return ydot(t,y,delta,v0,a,b)*thetadot(t,y,delta,a,b,v0) 

def fdot_linear(t,y,delta,a,b,v0):
    return [v0*(a*delta(t)/b+y[1]),v0*delta(t)/b]

def simulate(t=[0,100],v0=10,a=1.1,b=3.3,y0=2,theta0=0,delta=delta,event=(ydot,thetadot),fdot=fdot):
    res=scipy.integrate.solve_ivp(fdot,t,[y0,theta0],method='DOP853',t_eval=np.arange(1,101),vectorized=False,args=(delta,a,b,v0),rtol=1e-13,events=event)
    return res

def data():
    #Q3
    res=simulate(event=None)
    np.savez("data/Q3_t.npz",res.t)
    np.savez("data/Q3_y0.npz",res.y[0])
    np.savez("data/Q3_y1.npz",res.y[1])

#data()
def plot3():
    #t,theta,y=np.load("data/Q3_t.npz"),np.load("data/Q3_y0.npz"),np.load("data/Q3_y1.npz")
    res=simulate(delta=lambda x:0,event=None)
    t=res.t
    theta=res.y[1]
    y=res.y[0] 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Premier subplot pour x
    ax1.plot(t, theta, label=r'$\theta(t)$', color='blue')
    ax1.set_title(r'Graphique de $\theta$ en fonction du temps')
    ax1.set_xlabel('Temps (t)')
    ax1.set_ylabel(r'$\theta$')
    ax1.legend()
    # Deuxième subplot pour y
    ax2.plot(t, y, label='y(t)', color='red')
    ax2.set_title('Graphique de y en fonction du temps')
    ax2.set_xlabel('Temps (t)')
    ax2.set_ylabel('y')
    ax2.legend()
    fig.savefig("pyplot/Q3.png")

def plot7():
    for urlu in [0,3,15]:
        res=simulate(y0=0,theta0=0,event=None,delta=lambda x:urlu)
        res2=simulate(fdot=fdot_linear,y0=0,theta0=0,event=None,delta=lambda x:urlu)
        plt.plot(res.t,res.y[0],label=f'systeme normal pour {r'$\delta(t)$'}={urlu}')
        plt.plot(res2.t,res2.y[0],label=f'systeme linéarisé pour{r'$\delta(t)$'}={urlu}')
        plt.xlabel('Temps (t)')
        plt.ylabel('y(t)')
    plt.legend()
    plt.savefig("pyplot/Q7.png")
plot7()
def plot():
    #question=["data/Q3.npz","data/Q3.2.npz","data/Q8.npz"]
    res = simulate()
    t=res.t
    y=res.y[0]
    theta= res.y[1]
    ty=res.t_events[0]
    yy=res.y_events[0][:,0]
    ttheta=res.t_events[1]
    ytheta=res.y_events[1][:,1]
    plt.plot(t,y)
    plt.scatter(ty,yy)
    plt.scatter(ttheta,ytheta)
    plt.plot(t,theta)

    print("difftime theta ")
    print(np.mean(np.diff(ttheta)))
    print("difftime ydot ")
    print(np.mean(np.diff(ty[0::2])))
    print(np.mean(np.diff(ty[1::2])))
    print("difference time t")
    print(np.mean(np.diff(ty)[0::2]))
    print(np.mean(np.diff(ty)[1::2]))
    plt.show()

    res = simulate(event=fdotprod)
    t=res.t
    y=res.y[0]
    theta= res.y[1]
    ty=res.t_events[0]
    yy=res.y_events[0][:,0]
    plt.plot(t,y)
    plt.scatter(ty,yy)
    plt.scatter(ttheta,ytheta)
    plt.plot(t,theta)

    print("difftime theta ")
    print(np.mean(np.diff(ttheta)))
    print("difftime ydot ")
    print(np.mean(np.diff(ty[0::2])))
    print(np.mean(np.diff(ty[1::2])))
    print("difference time t")
    print(np.mean(np.diff(ty)[0::2]))
    print(np.mean(np.diff(ty)[1::2]))
    plt.show()



@np.vectorize
def gaussian(t,a,u,sigma2):
    sigma2=np.abs(sigma2)
    return a*np.exp(-((t-u)**2)/(2*sigma2))/np.sqrt(2*np.pi*sigma2)

def minimizer(guess):
    def dez(t):
        val=0
        for i in range(0,len(guess)-2,3):
            a,u,sigma=guess[i:i+3]
            val+=gaussian(t,a,u,sigma)
        return val
    res=simulate(y0=0,delta=dez,event=None)
    if(len(res.t)==0):
        return np.inf
    real_y=gaussian(res.t,25,50,25)
    mse=np.mean((res.y[0]-real_y)**2)
    return mse
#print(minimizer([1,49,1]))

def work(guess):
    options={'maxiter':10}
    res=scipy.optimize.minimize(minimizer,guess,tol=1,options=options)
    print(f'{ guess} : { res.x }-> {res.fun}')
    if(res.success==True):
        print(res)
""" code utilisé pour tester plusieurs minimization en paralèlle
#print(scipy.optimize.minimize(minimizer,[1,1,1]))
size = [3*x for x in reversed(range (3,4))]
size=[9]
for length in size:
    testvalue=[random.randint(1,100) for x in range ( 10)]
    echantillon=list(itertools.combinations_with_replacement(testvalue,length))
    num_cores = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cores) as pool:
        pool.map(work,echantillon)
"""
def plotter():
    guess=[23.36933557,23.06962894,23.198646 , 23.36933557,23.06962894,23.198646 , 3.52108979, 3.4250309 , 3.18470741, 3.38102216 ,3.11939477,52.11371241 ] 
    def dez(t):
        val=0
        for i in range(0,len(guess)-2,3):
            a,u,sigma=guess[i:i+3]
            val+=gaussian(t,a,u,sigma)
        return val
    res=simulate(y0=0,delta=dez,event=None)
    real_y=gaussian(res.t,25,50,25)
    plt.plot(res.t,res.y[0],color='blue')
    plt.plot(res.t,real_y,color='red')
    plt.show()
