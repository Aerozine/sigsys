import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Définition des paramètres

v0 = 10  # (m/s)
t_initial = 0 # (s)
t_final = 100  # (s)
y0 = 2  
theta0 = 0  
a = 1.1 # (m)
b = 3.3 # (m)

t_eval = np.linspace(0, t_final, 1000)  # Points de temps


# Définition des entrées
def delta1(t):
    return 0  

def delta2(t):
    return - (np.pi / 2) * np.sin(2 * np.pi * 0.1 * t)

# Fonction pour résoudre le système
def system(t, X, delta_func):
    y, theta = X
    delta_t = delta_func(t)
    alpha = np.arctan(a * np.tan(delta_t)/b)
    
    dy_dt = v0 * np.sin(alpha + theta)
    dtheta_dt = v0 * np.sin(alpha)/a
    
    return [dy_dt, dtheta_dt]

# Simulation 
sol1 = solve_ivp(system, (t_initial, t_final), [y0, theta0], t_eval=t_eval, args=(delta1,))
sol2 = solve_ivp(system, (t_initial, t_final), [y0, theta0], t_eval=t_eval, args=(delta2,))

# Extraction des résultats
t = sol1.t
y1, theta1 = sol1.y
y2, theta2 = sol2.y  
delta1 = np.zeros_like(t)
delta2 = - (np.pi / 2) * np.sin(2 * np.pi * 0.1 * t)

# Affichage des résultats
plt.figure(figsize=(12, 8))

# Tracé de δ(t) pour le second cas
plt.subplot(3, 1, 1)
plt.plot(t, delta1, label="δ(t) = 0", linestyle="dashed", color='blue')
plt.plot(t, delta2, label=r"$\delta(t) = -\frac{\pi}{2} \sin(2\pi \times 0.1 \times t)$", color='purple')
plt.yticks([-1.5, 0, 1.5])
plt.xlabel("Temps [s]")
plt.ylabel(r"$\delta(t)$")
plt.legend()
plt.grid()

# Tracé de y(t)
plt.subplot(3, 1, 2)
plt.plot(t, y1, label="y(t) pour δ(t) = 0", linestyle="dashed", color='blue')
plt.plot(t, y2, label=r"y(t) pour $\delta(t) = -\frac{\pi}{2} \sin(2\pi \times 0.1 \times t)$", color='red')
plt.yticks([-30, -20, -10, 0, 2])
plt.xlabel("Temps [s]")
plt.ylabel("y(t)")
plt.legend()
plt.grid()

# Tracé de θ(t)
plt.subplot(3, 1, 3)
plt.plot(t, theta1, label=r"$\theta(t)$ pour $\delta(t) = 0$", linestyle="dashed", color='blue')
plt.plot(t, theta2, label=r"$\theta(t)$ pour $\delta(t) = -\frac{\pi}{2} \sin(2\pi \times 0.1 \times t)$", color='red')
plt.yticks([0, -25])
plt.xlabel("Temps [s]")
plt.ylabel(r"$\theta(t)$")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()