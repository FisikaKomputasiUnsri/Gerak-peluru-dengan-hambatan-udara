# Gerak-peluru-dengan-hambatan-udara
Gerak peluru dengan mempertimbangkan hambatan udara

import math
import numpy as np
import matplotlib.pyplot as plt

# ==============================
# PARAMETER FISIK
# ==============================
g = 9.81            # m/s^2
m = 0.02            # kg (massa peluru)
Cd = 0.47           # koefisien drag (bentuk bulat)
rho = 1.225         # kg/m3 (massa jenis udara)
r = 0.009           # m (jari-jari peluru ~9 mm)
A = math.pi * r**2  # luas penampang
c = 0.5 * Cd * rho * A

# ==============================
# KONDISI AWAL
# ==============================
x0, y0 = 0.0, 0.0
speed = 300.0                 # kecepatan awal (m/s)
angle = math.radians(30)      # sudut peluncuran
vx0 = speed * math.cos(angle)
vy0 = speed * math.sin(angle)

# ==============================
# SISTEM PERSAMAAN DIFERENSIAL
# ==============================
def deriv(state):
    x, y, vx, vy = state
    v = math.hypot(vx, vy)
    ax = -(c/m) * v * vx
    ay = -g - (c/m) * v * vy
    return (vx, vy, ax, ay)

def rk4_step(state, dt):
    k1 = deriv(state)
    s2 = tuple(state[i] + 0.5*dt*k1[i] for i in range(4))
    k2 = deriv(s2)
    s3 = tuple(state[i] + 0.5*dt*k2[i] for i in range(4))
    k3 = deriv(s3)
    s4 = tuple(state[i] + dt*k3[i] for i in range(4))
    k4 = deriv(s4)
    new = tuple(state[i] + dt*(k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0 for i in range(4))
    return new

# ==============================
# SIMULASI DENGAN DRAG (NUMERIK)
# ==============================
dt = 0.001
state = (x0, y0, vx0, vy0)
trajectory_drag = [(0.0, x0, y0, vx0, vy0)]
t = 0.0

while True:
    state_next = rk4_step(state, dt)
    t += dt
    trajectory_drag.append((t,)+state_next)

    # cek jika peluru sudah jatuh (y <= 0)
    if state[1] > 0 and state_next[1] <= 0:
        # interpolasi linear untuk titik jatuh
        t1, x1, y1, vx1, vy1 = trajectory_drag[-2]
        t2, x2, y2, vx2, vy2 = trajectory_drag[-1]
        frac = y1 / (y1 - y2)
        t_land = t1 + frac * (t2 - t1)
        x_land = x1 + frac * (x2 - x1)
        vx_land = vx1 + frac * (vx2 - vx1)
        vy_land = vy1 + frac * (vy2 - vy1)
        trajectory_drag[-1] = (t_land, x_land, 0.0, vx_land, vy_land)
        break
    state = state_next

times_drag = [p[0] for p in trajectory_drag]
xs_drag = [p[1] for p in trajectory_drag]
ys_drag = [p[2] for p in trajectory_drag]
vs_drag = [math.hypot(p[3], p[4]) for p in trajectory_drag]

Ek_drag = [0.5 * m * v**2 for v in vs_drag]
Ep_drag = [m * g * y for y in ys_drag]
Em_drag = [Ek_drag[i] + Ep_drag[i] for i in range(len(Ek_drag))]

# ==============================
# SOLUSI ANALITIK TANPA DRAG
# ==============================
T_flight = 2 * vy0 / g
t_vals = np.linspace(0, T_flight, 500)
x_analytical = vx0 * t_vals
y_analytical = vy0 * t_vals - 0.5 * g * t_vals**2
v_analytical = np.sqrt(vx0**2 + (vy0 - g*t_vals)**2)

Ek_analytical = 0.5 * m * v_analytical**2
Ep_analytical = m * g * y_analytical
Em_analytical = Ek_analytical + Ep_analytical

# ==============================
# PLOT LINTASAN
# ==============================
plt.figure(figsize=(10,5))
plt.plot(xs_drag, ys_drag, label="Dengan drag (numerik RK4)")
plt.plot(x_analytical, y_analytical, '--', label="Tanpa drag (analitik)")
plt.xlabel("Jarak horizontal (m)")
plt.ylabel("Ketinggian (m)")
plt.title("Lintasan Gerak Peluru")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# PLOT KECEPATAN VS WAKTU
# ==============================
plt.figure(figsize=(10,5))
plt.plot(times_drag, vs_drag, label="Dengan drag (numerik RK4)")
plt.plot(t_vals, v_analytical, '--', label="Tanpa drag (analitik)")
plt.xlabel("Waktu (s)")
plt.ylabel("Kecepatan (m/s)")
plt.title("Perbandingan Kecepatan vs Waktu")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# PLOT ENERGI KINETIK VS WAKTU
# ==============================
plt.figure(figsize=(10,5))
plt.plot(times_drag, Ek_drag, label="Dengan drag (numerik RK4)")
plt.plot(t_vals, Ek_analytical, '--', label="Tanpa drag (analitik)")
plt.xlabel("Waktu (s)")
plt.ylabel("Energi Kinetik (Joule)")
plt.title("Energi Kinetik vs Waktu")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# PLOT ENERGI MEKANIK TOTAL VS WAKTU
# ==============================
plt.figure(figsize=(10,5))
plt.plot(times_drag, Em_drag, label="Dengan drag (numerik RK4)")
plt.plot(t_vals, Em_analytical, '--', label="Tanpa drag (analitik)")
plt.xlabel("Waktu (s)")
plt.ylabel("Energi Mekanik Total (Joule)")
plt.title("Energi Mekanik Total vs Waktu")
plt.legend()
plt.grid(True)
plt.show()

# ==============================
# HASIL NUMERIK
# ==============================
print("=== HASIL SIMULASI ===")
print(f"Dengan drag : Jarak = {xs_drag[-1]:.2f} m, Waktu = {times_drag[-1]:.2f} s")
print(f"Tanpa drag  : Jarak = {x_analytical[-1]:.2f} m, Waktu = {T_flight:.2f} s")
