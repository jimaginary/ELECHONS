import elechons.models.linear_regression as lr
import elechons.regress_temp as r
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm, colors
from elechons.processing import edges as ec
from elechons.data import station_handler as sh
import elechons.config as config
import torch

r.init('mean')

filt = r.lat != np.nan
print(sum(filt))

C = torch.tensor(np.corrcoef(r.regression_error[filt]).flatten())
N = np.prod(C.shape)

d_lat = (r.lat[filt, np.newaxis] - r.lat[np.newaxis, filt]).flatten() * 2 * np.pi * config.EARTH_RADIUS_KM / 360
d_long = (r.long[filt, np.newaxis] - r.long[np.newaxis, filt]).flatten() * 2 * np.pi * config.EARTH_RADIUS_KM / 360

x = torch.tensor([d_lat, d_long], requires_grad=True, dtype=torch.float32).unsqueeze(0)
xT = x.transpose(1, 2)

n = 2
kA = torch.tensor(np.linspace(0.001,n*0.001,n), dtype=torch.float32).view(n,1,1)
A = (kA * torch.eye(2)).clone().detach().requires_grad_(True)

kB = torch.tensor(np.linspace(0.001,n*0.001,n), dtype=torch.float32).view(n,1,1)
B = (kB * torch.eye(2)).clone().detach().requires_grad_(True)

kD = torch.tensor(n*[1/n], dtype=torch.float32).view(n,1)
D = kD.clone().detach().requires_grad_(True)

optimiser = torch.optim.Adam([A, B, D], lr=1e-4)

for i in range(1000):
    optimiser.zero_grad()
    xAAx = torch.sqrt((torch.matmul(A, x) * torch.matmul(A, x)).sum(dim=1).clamp(min=1e-12))
    xBBx = torch.sqrt((torch.matmul(B, x) * torch.matmul(B, x)).sum(dim=1).clamp(min=1e-12))
    f = ((C - torch.matmul(D.T / D.sum(), torch.exp(-xAAx) * torch.cos(xBBx)).sum(dim=0))**2).sum(dim=0)
    f.backward()
    optimiser.step()

def fit(a, phi):
    v = torch.tensor([a*np.cos(phi),-a*np.sin(phi)], dtype=torch.float32)
    vAAv = torch.sqrt((torch.matmul(A, v) * torch.matmul(A, v)).sum(dim=1).clamp(min=1e-12))
    vBBv = torch.sqrt((torch.matmul(B, v) * torch.matmul(B, v)).sum(dim=1).clamp(min=1e-12))
    return torch.matmul(D.T / D.sum(), torch.exp(-vAAv) * torch.cos(vBBv)).sum(dim=0)

a = np.linspace(0,4000,200)
phi = np.linspace(-np.pi/2,np.pi/2,200)
Am,phim = np.meshgrid(a,phi)
Z = np.zeros_like(Am)
for i in range(Am.shape[0]):
    for j in range(Am.shape[1]):
        Z[i,j] = fit(Am[i,j],phim[i,j])
plt.imshow(Z,cmap='rainbow',vmax=1,vmin=-0.2, extent=[np.min(a),np.max(a),np.min(phi),np.max(phi)], aspect='auto')
plt.colorbar(label = 'correlation')
# plt.title('anisotropic correlation fit for regression error')
plt.xlabel('distance (km)')
plt.ylabel('angle (rad)')
plt.savefig('plts/anisotropy_fit.png', dpi=300, bbox_inches='tight')
plt.close()

ATA = (A.transpose(-1,-2) @ A).detach().numpy()
BTB = (B.transpose(-1,-2) @ B).detach().numpy()
Dn = (D/D.sum()).detach().numpy()
print(f'f = {np.sqrt(f.item()/N)}, ATA = \n{ATA}, \nBTB = \n{BTB}, \nD = {Dn}')

for i in range(n):
    print(i)
    ATAl, ATAv = np.linalg.eigh(ATA[i])
    BTBl, BTBv = np.linalg.eigh(BTB[i])
    print(f'sqrtATAl = {np.sqrt(ATAl)}, ATAv = \n{ATAv}')
    print(f'sqrtBTBl = {np.sqrt(np.abs(BTBl))}, BTBv = \n{BTBv}')

# Z = torch.matmul(D.T, torch.exp(-xAAx) * torch.sinc(xBBx)).sum(dim=0).detach().numpy()
# plt.scatter(d_long, d_lat, c=Z, cmap='rainbow')
# plt.colorbar()
# plt.show()

# # now look for heteroscedasticity
# xAAx = torch.sqrt((torch.matmul(A, x) * torch.matmul(A, x)).sum(dim=1).clamp(min=1e-12))
# xBBx = torch.sqrt((torch.matmul(B, x) * torch.matmul(B, x)).sum(dim=1).clamp(min=1e-12))
# plt.scatter(C.numpy(), torch.matmul(D.T / D.sum(), torch.exp(-xAAx) * torch.sinc(xBBx)).sum(dim=0).detach().numpy())
# plt.savefig('plts/anisotropy_heteroscedasticity.png', dpi=300, bbox_inches='tight')

# for i in range(1000):
#     optimiser.zero_grad()
#     xAAx = torch.sqrt((torch.matmul(A, x) * torch.matmul(A, x)).sum(dim=1).clamp(min=1e-12))
#     f = ((C - torch.matmul(D.T, torch.exp(-xAAx)).sum(dim=0))**2).sum(dim=0)
#     f.backward()
#     optimiser.step()

# print(f'f = {np.sqrt(f.item()/N)}, ATA = {A.transpose(-1,-2) @ A}, D = {D/D.sum()}')