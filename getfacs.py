import os

seqs = []
with open("/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank/train/wav.scp", "r") as pid:
	line = pid.readline()
	while line:
		seq = line.split(" ")[0]
		seqs.append(seq)
		line = pid.readline()

facs = dict()
with open("/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank/fac/all_facs.scp", "r") as fid:
	line = fid.readline()
	while line:
		seq = line.split(" ")[0]
		gender = line.split(" ")[1]
		region = line.split(" ")[2]
		facs[seq] = [gender, region]
		line = fid.readline()

with open("/esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank/fac/test_on_trainset_facs.scp", "w") as gid:
	for seq in seqs:
		fac = facs[seq]
		gender = fac[0]
		region = fac[1]
		gid.write(seq+" "+str(gender)+" "+str(region))


			
