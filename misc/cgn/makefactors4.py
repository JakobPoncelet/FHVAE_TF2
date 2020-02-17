import os

# make .scp files with factors per speaker.
# Source is the time-aligned labels file (talab)

dir = "/users/spraak/hvanhamm/repos/ScalableFHVAE/"
outdir = dir + "misc/"
with open(os.path.join(outdir,"speakers_regions.csv")) as f:
    speaker=[l.rstrip('\n').split(';')[0].lower() for l in f]
    f.seek(0)
    gender=[l.rstrip('\n').split(';')[1] for l in f]
    f.seek(0)
    reg1=[l.rstrip('\n').split(';')[2] for l in f]
    f.seek(0)
    reg2=[l.rstrip('\n').split(';')[3] for l in f]
    f.seek(0)
    reg3=[l.rstrip('\n').split(';')[4] for l in f]
    f.seek(0)
    size=[l.rstrip('\n').rstrip('\r').split(';')[5] for l in f]
    f.close()

f_spk=open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab_spk.scp"),"w")
f_comp=open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab_comp.scp"),"w")
f_reg1=open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab_reg1.scp"),"w")
f_reg2=open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab_reg2.scp"),"w")
f_reg3=open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab_reg3.scp"),"w")
f_gender=open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab_gender.scp"),"w")

with open(os.path.join(outdir,"cgn_per_spk_afgklno_all_talab.pho")) as f:
    for l in f:
        line=l.split()
        if len(line)==1:
            (spk,nr,comp)=line[0].split('_')
            k = speaker.index(spk.lower())
            f_spk.write('%s %s\n' % (line[0],spk))
            f_comp.write('%s %s\n' % (line[0], comp))
            f_reg1.write('%s %s\n' % (line[0], reg1[k]))
            f_reg2.write('%s %s\n' % (line[0], reg2[k]))
            f_reg3.write('%s %s\n' % (line[0], reg3[k]))
            f_gender.write('%s %s\n' % (line[0], gender[k]))

f_spk.close()
f_comp.close()
f_reg1.close()
f_reg2.close()
f_reg3.close()
f_gender.close()

