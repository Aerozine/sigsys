import numpy as np 
from matplotlib import pyplot as plt 
import scipy 

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
    res=scipy.integrate.solve_ivp(fdot,t,[y0,theta0],method='DOP853',vectorized=False,args=(delta,a,b,v0),rtol=1e-13,events=event)
    return res

def data():
    #Q3
    res=simulate(event=None)
    np.savez("data/Q3_t.npz",res.t)
    np.savez("data/Q3_y0.npz",res.y[0])
    np.savez("data/Q3_y1.npz",res.y[1])

data()
def plot3():
    t,theta,y=np.load("data/Q3_t.npz"),np.load("data/Q3_y0.npz"),np.load("data/Q3_y1.npz")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    # Premier subplot pour x
    ax1.plot(t, theta, label='x(t)', color='blue')
    ax1.set_title('Graphique de x en fonction du temps')
    ax1.set_xlabel('Temps (t)')
    ax1.set_ylabel('x')
    ax1.legend()
    # Deuxième subplot pour y
    ax2.plot(t, y, label='y(t)', color='red')
    ax2.set_title('Graphique de y en fonction du temps')
    ax2.set_xlabel('Temps (t)')
    ax2.set_ylabel('y')
    ax2.legend()
    fig.savefig("pyplot/Q3.png")
plot3()
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
#plot()




@np.vectorize
def gaussian(t,a,u,sigma2):
    return a*np.exp(-((t-u)**2)/(2*sigma2))/np.sqrt(2*np.pi*sigma2)

def minimizer(guess):
    a,u,sigma=guess
    def dez(t):
        return gaussian(t,a,u,sigma)
    res=simulate(y0=0,delta=dez,event=None)
    real_y=gaussian(res.t,a,u,sigma)
    mse=np.mean((res.y-real_y)**2)
    return mse

testval= [x**2 for x in range(1, 11)]
for i in testval:
    for j in testval:
        for k in testval:
            res=scipy.optimize.minimize(minimizer,[i,j,k])
            if(res.succes=True)
                print(res)
