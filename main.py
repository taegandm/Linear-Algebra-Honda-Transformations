import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import show

D = np.array([[-6.5, -6.5, -6.5, -6.5, -2.5, -2.5, -0.75, -0.75, 3.25, 3.25, 4.5, 4.5, 6.5, 6.5, 6.5, 6.5],
              [-2, -2, 0.5, 0.5, 0.5, 0.5, 2, 2, 2, 2, 0.5, 0.5, 0.5, 0.5, -2, -2],
              [-2.5, 2.5, 2.5, -2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, -2.5, 2.5, 2.5, -2.5],
              [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

C = np.array([[0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
              [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
              [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
              [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]])

# Center (-5, 10, 10)
P1 = np.array([[1, 0, 0.5, 0],
               [0, 1, -1, 0],
               [0, 0, 0, 0],
               [0, 0, -0.1, 1]])

PD1 = np.matmul(P1, D)
for j in range(16):
    PD1[:, j] = PD1[:, j]/PD1[3, j]

PD1[0, :] = PD1[0, :]/PD1[3, :]
PD1[1, :] = PD1[1, :]/PD1[3, :]

f, ax1 = plt.subplots(1)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
ax1.plot(PD1[0, :], PD1[1, :], 'b.')

for i in range(16):
    for j in range(i):
        if C[i, j] == 1:
            ax1.plot([PD1[0, i], PD1[0, j]], [PD1[1, i], PD1[1, j]], 'r-')
ax1.set_title('Center at (-5,10,10)')
ax1.axis('off')

# Center (0, 10, 25)
P2 = np.array([[1, 0, 0, 0],
               [0, 1, -10/25, 0],
               [0, 0, 0, 0],
               [0, 0, -1/25, 1]])
PD2 = np.matmul(P2, D)

for j in range(16):
    PD2[:, j] = PD2[:, j]/PD2[3, j]
PD2[0, :] = PD2[0, :]/PD2[3, :]
PD2[1, :] = PD2[1, :]/PD2[3, :]

f, ax2 = plt.subplots(1)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
ax2.plot(PD2[0, :], PD2[1, :], 'b.')

for i in range(16):
    for j in range(i):
        if C[i, j] == 1:
            ax2.plot([PD2[0, i], PD2[0, j]], [PD2[1, i], PD2[1, j]], 'r-')
ax2.set_title('Center at (0,10,25)')
ax2.axis('off')

# Rotate 30* about y, center (0, 10, 25)
R1 = np.array([[math.cos(math.pi/6), 0, (-1)*math.sin(math.pi/6), 0],
               [0, 1, 0, 0],
               [math.sin(math.pi/6), 0, math.cos(math.pi/6), 0],
               [0, 0, 0, 1]])
RD1 = np.matmul(R1, D)  # rotate 30* about y
PRD1 = np.matmul(P2, RD1)  # perform perspective projection (0, 10, 25)

for j in range(16):
    PRD1[:, j] = PRD1[:, j]/PRD1[3, j]
PRD1[0, :] = PRD1[0, :]/PRD1[3, :]
PRD1[1, :] = PRD1[1, :]/PRD1[3, :]

f, ax3 = plt.subplots(1)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
ax3.plot(PRD1[0, :], PRD1[1, :], 'b.')
for i in range(16):
    for j in range(i):
        if C[i, j] == 1:
            ax3.plot([PRD1[0, i], PRD1[0, j]], [PRD1[1, i], PRD1[1, j]], 'r-')
ax3.set_title('Rotate 30* around y, center at (0,10,25)')
ax3.axis('off')

# Rotate 45* about z, center (0, 10, 25)
R2 = np.array([[math.cos(math.pi/4), (-1)*math.sin(math.pi/4), 0, 0],
               [math.sin(math.pi/4), math.cos(math.pi/4), 0, 0],
               [0, 0, 1, 0],
               [0, 0, 0, 1]])
RD2 = np.matmul(R2, D)  # rotate 45* about z
PRD2 = np.matmul(P2, RD2)  # perform perspective projection (0, 10, 25)

for j in range(16):
    PRD2[:, j] = PRD2[:, j]/PRD2[3, j]
PRD2[0, :] = PRD2[0, :]/PRD2[3, :]
PRD2[1, :] = PRD2[1, :]/PRD2[3, :]

f, ax4 = plt.subplots(1)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
ax4.plot(PRD2[0, :], PRD2[1, :], 'b.')

for i in range(16):
    for j in range(i):
        if C[i, j] == 1:
            ax4.plot([PRD2[0, i], PRD2[0, j]], [PRD2[1, i], PRD2[1, j]], 'r-')
ax4.set_title('Rotate 45* around z, center at (0,10,25)')
ax4.axis('off')

# Zoom 150%, center (0, 10, 25)
Z = np.array([[1.5, 0, 0, 0],
              [0, 1.5, 0, 0],
              [0, 0, 1.5, 0],
              [0, 0, 0, 1]])
ZD = np.matmul(Z, D)
PDZ = np.matmul(P2, ZD)

for j in range(16):
    PDZ[:, j] = PDZ[:, j]/PDZ[3, j]
PDZ[0, :] = PDZ[0, :]/PDZ[3, :]
PDZ[1, :] = PDZ[1, :]/PDZ[3, :]

f, ax5 = plt.subplots(1)
plt.xlim(-12, 12)
plt.ylim(-12, 12)
ax5.plot(PDZ[0, :], PDZ[1, :], 'b.')

for i in range(16):
    for j in range(i):
        if C[i, j] == 1:
            ax5.plot([PDZ[0, i], PDZ[0, j]], [PDZ[1, i], PDZ[1, j]], 'r-')
ax5.set_title('Zoom 150%, center at (0,10,25)')
ax5.axis('off')
show()
