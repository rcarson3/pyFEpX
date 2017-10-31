import numpy as np
#import numpy.matlib as npm
#import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches

rc('mathtext', default='regular')

font = {'size'   : 14}

rc('font', **font)

fileLoc = ['/Users/robertcarson/OneDrive/n500_pois_iso_A/post.force2', '/Users/robertcarson/OneDrive/n500_pois_dg_0_all14_A/hires/post.force2', '/Users/robertcarson/OneDrive/n500_pois_dg_1_all_1_A/post.force2'] #, '/Volumes/My Passport for Mac/Simulations/midres/aniStudy/A/n500_pois_dg_1_all0_A/post.force2']

ii = 0

clrs = ['red', 'blue', 'green', 'black']
mrks = ['-.', ':', '--', 'solid']

fig, ax = plt.subplots(1)

s = ['A','E','F','H','I']

MSize2 = 3

for fLoc in fileLoc:

    data = np.loadtxt(fLoc, comments='%')
    
    # simname='LoadControl'
    simname = 'DispControl'
    l0 = 1
    epsdot = 1e-3
    MSize = 2
    
    nincr = data.shape[0]
    nind = np.arange(0,nincr+1)
    istep = np.concatenate((np.array([1]), data[:, 0]))
    sig = np.concatenate((np.array([0]), data[:, 4])) / (l0 ** 2)
    time = np.concatenate((np.array([0]), data[:, 6]))
    eps = np.zeros((nincr + 1))
    
    ind = np.squeeze(np.asarray([nind[time==3.0], nind[time==6.0], nind[time==9.0], nind[time==12.0], nind[time==15.0]]))
    
    for i in range(1, nincr + 1):
        dtime = time[i] - time[i - 1]
        if sig[i] - sig[i - 1] > 0:
            eps[i] = eps[i - 1] + epsdot * dtime
        else:
            eps[i] = eps[i - 1] - epsdot * dtime
    
    if simname == 'LoadControl':
#        fig = plt.figure()
        # 	ax=plt.axis([0,0.10,0,200])
        ax.plot(eps, sig, color=clrs[ii], marker='*', markersize=MSize)
    elif simname == 'DispControl':
#        fig = plt.figure()
        # 	plt.axis([0,0.10,0,200])
        if ii == 0:
            ax.plot(eps, sig, color=clrs[ii], linestyle=mrks[ii], linewidth=MSize2)
        else:
            ax.plot(eps, sig, color=clrs[ii], linestyle=mrks[ii], linewidth=MSize)
#        if (ii==0):
#            for i, txt in enumerate(s):
#                if (i==0) or (i==4):
#                    ax.annotate(txt,(np.squeeze(eps[ind[i]]+eps[ind[i]]*0.07), np.squeeze(sig[ind[i]]-sig[ind[i]]*0.07)))
#                else:
#                    ax.annotate(txt,(np.squeeze(eps[ind[i]]+eps[ind[i]]*0.05), np.squeeze(sig[ind[i]]+sig[ind[i]]*0.2)))
#    [print(j) for j in eps]
    ii += 1
    
fLoc = '/Users/robertcarson/OneDrive/n500_pois_iso_A/post.force'
data = np.loadtxt(fLoc, comments='%')
ax.plot(data[:,0], data[:,1], color=clrs[ii], linestyle=mrks[ii], linewidth=MSize)
ax.grid()

box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height * 0,
                 box.width, box.height * 1])
                 
ax.axis([-0.0035, 0.0035, -300, 300])

ax.set_ylabel('Macroscopic engineering stress [MPa]')
ax.set_xlabel('Macroscopic engineering strain [-]')
#plt.title('Macroscopic Stress-Strain Curve')

red_patch = mpatches.Patch(color='red', label='Isotropic hardening')
blue_patch = mpatches.Patch(color='blue', label='Latent hardening with direct hardening off')
green_patch = mpatches.Patch(color='green', label='Latent hardening with direct hardening on')
black_patch = mpatches.Patch(color='black', label='Experimental OMC copper')

ax.legend(handles=[red_patch, blue_patch, green_patch, black_patch], loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, ncol=1)

fig.show()
plt.show()



picLoc = 'SS_strain_hires_exp.png'
fig.savefig(picLoc, dpi = 300, bbox_inches='tight')
