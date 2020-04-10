import numpy as np
import os

base = '/users/students/r0797363/spchdisk/src/FHVAE_TF2/exp'
split = 'test'
feat = 'mu1'

print('exp\t#\tglobal_var\tmean_spk_var')
for exp in os.listdir(base):
    if os.path.exists(os.path.join(base, exp, split, feat)):
        mu2_by_spk = dict()
        all_x = []
        for seq in os.listdir(os.path.join(base, exp, split, feat)):
            x = np.load(os.path.join(base, exp, split, feat, seq))
            spk = seq.split('_')[0]
            if spk in mu2_by_spk:
                mu2_by_spk[spk].append(x.squeeze())
            else:
                mu2_by_spk[spk] = [x.squeeze()]
            all_x.append(x.squeeze())

        all_x = np.array(all_x)

        variances = []
        for spk in mu2_by_spk: 
            x = np.array(mu2_by_spk[spk])
            var = np.square(x - x.mean(axis=0)).sum(axis=1).mean()
            variances.append(var)

        mean_var = np.mean(variances)
        global_var = np.square(all_x - all_x.mean(axis=0)).sum(axis=1).mean()
        count = all_x.shape[0]
        print('%s\t%d\t%f\t%f' % (exp, count, global_var, mean_var))
