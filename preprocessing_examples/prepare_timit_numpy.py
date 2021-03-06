from __future__ import division
import os
import wave
import argparse
import subprocess
from sphfile import SPHFile

"""
prepare TIMIT data for FHVAE

## COMMAND PARAMETERS:
/users/spraak/spchdata/timit/CDdata /users/spraak/spchdata/timit/CDdata/timit/doc --ftype fbank --out_dir /esat/spchdisk/scratch/jponcele/fhvae_jakob/datasets/timit_np_fbank
"""


def maybe_makedir(d):
    try:
        os.makedirs(d)
    except OSError:
        pass

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("timit_wav_dir", type=str,
        help="TIMIT directory containing wavs, e.g. timit/CDdata")
parser.add_argument("timit_doc_dir", type=str,
        help="TIMIT directory containing spkrinfo.txt, e.g. timit/CDdata/timit/doc")
parser.add_argument("--ftype", type=str, default="fbank", choices=["fbank", "spec"],
        help="feature type")
parser.add_argument("--out_dir", type=str, default="./datasets/timit_np_fbank",
        help="output data directory")
parser.add_argument("--dev_spk", type=str, default="./misc/timit/timit_dev_spk.list",
        help="path to list of dev set speakers")
parser.add_argument("--test_spk", type=str, default="./misc/timit/timit_test_spk.list",
        help="path to list of test set speakers")
args = parser.parse_args()
print(args)

# Samples per frame to convert timealigned labels (timit database uses #samples for start and stop times)
SampleRate = 16000
SecondsPerFrame = 0.01

sam_frame = SampleRate * SecondsPerFrame


# retrieve partition
with open(args.dev_spk) as f:
    dt_spks = [line.rstrip().lower() for line in f]
with open(args.test_spk) as f:
    tt_spks = [line.rstrip().lower() for line in f]

# convert sph to wav and dump scp
wav_dir = os.path.abspath("%s/wav" % args.out_dir)
tr_scp = "%s/train/wav.scp" % args.out_dir
dt_scp = "%s/dev/wav.scp" % args.out_dir
tt_scp = "%s/test/wav.scp" % args.out_dir

maybe_makedir(wav_dir)
maybe_makedir(os.path.dirname(tr_scp))
maybe_makedir(os.path.dirname(dt_scp))
maybe_makedir(os.path.dirname(tt_scp))

tr_f = open(tr_scp, "w")
dt_f = open(dt_scp, "w")
tt_f = open(tt_scp, "w")

paths = []
utts_per_spk = {}
for root, _, fnames in sorted(os.walk(args.timit_wav_dir)):
    spk = root.split("/")[-1].lower()
    if spk in dt_spks:
        f = dt_f
    elif spk in tt_spks:
        f = tt_f
    else:
        f = tr_f

    uttids = []
    for fname in fnames:
        if fname.endswith(".wav") or fname.endswith(".WAV"):
            sph_path = "%s/%s" % (root, fname)
            path = "%s/%s_%s" % (wav_dir, spk, fname)
            uttid = "%s_%s" % (spk, os.path.splitext(fname)[0])
            f.write("%s %s\n" % (uttid, path))
            sph = SPHFile(sph_path)
            sph.write_wav(path)
            uttids.append(uttid)

    utts_per_spk[spk] = uttids

tr_f.close()
dt_f.close()
tt_f.close()

print("converted to wav and dumped scp files")

# compute feature
feat_dir = os.path.abspath("%s/%s" % (args.out_dir, args.ftype))
maybe_makedir(feat_dir)

def compute_feature(name):
    cmd = ["python", "./scripts/preprocess/prepare_numpy_data.py", "--ftype=%s" % args.ftype]
    cmd += ["%s/%s/wav.scp" % (args.out_dir, name), feat_dir]
    cmd += ["%s/%s/feats.scp" % (args.out_dir, name)]
    cmd += ["%s/%s/len.scp" % (args.out_dir, name)]

    p = subprocess.Popen(cmd)
    if p.wait() != 0:
        raise RuntimeError("Non-zero (%d) return code for `%s`" % (p.returncode, " ".join(cmd)))

for name in ["train", "dev", "test"]:
    compute_feature(name)

print("computed feature")

# put the regularizing factors in a file
facs = "%s/fac/all_facs.scp" % args.out_dir
facs_gender = "%s/fac/all_facs_gender.scp" % args.out_dir
facs_reg = "%s/fac/all_facs_region.scp" % args.out_dir
facs_spk = "%s/fac/all_facs_spk.scp" % args.out_dir
tr_facs = "%s/fac/train_facs.scp" % args.out_dir
dt_facs = "%s/fac/dev_facs.scp" % args.out_dir
tt_facs = "%s/fac/test_facs.scp" % args.out_dir
maybe_makedir(os.path.dirname(facs))
maybe_makedir(os.path.dirname(facs_gender))
maybe_makedir(os.path.dirname(facs_reg))
maybe_makedir(os.path.dirname(facs_spk))
maybe_makedir(os.path.dirname(tr_facs))
maybe_makedir(os.path.dirname(dt_facs))
maybe_makedir(os.path.dirname(tt_facs))

facs_f = open(facs, "w")
facs_gender_f = open(facs_gender, "w")
facs_reg_f = open(facs_reg, "w")
facs_spk_f = open(facs_spk, "w")
tr_facs_f = open(tr_facs, "w")
dt_facs_f = open(dt_facs, "w")
tt_facs_f = open(tt_facs, "w")

tr_facs_f.write("#seq gender region\n")
dt_facs_f.write("#seq gender region\n")
tt_facs_f.write("#seq gender region\n")

info_f = open(os.path.join(args.timit_doc_dir, 'spkrinfo.txt'), "r")

for x in range(0, 40, 1):  # the first 40 lines are header lines in my spkrinfo.txt
    line = info_f.readline()

while line:
    spk = (line.split("  ")[0]).lower()
    gender = (line.split("  ")[1]).lower()
    reg = line.split("  ")[2]
    spkid = gender+spk

    if spkid in dt_spks:
        fff = dt_facs_f
    elif spkid in tt_spks:
        fff = tt_facs_f
    else:
        fff = tr_facs_f

    for uttid in utts_per_spk[spkid]:
        facs_f.write(uttid+" "+gender+" "+reg+"\n")
        fff.write(uttid+" "+gender+" "+reg+"\n")
        facs_gender_f.write(uttid+" "+gender+"\n")
        facs_reg_f.write(uttid+" "+reg+"\n")
        facs_spk_f.write(uttid+" "+spkid+"\n")

    line = info_f.readline()

info_f.close()

facs_f.close()
facs_reg_f.close()
facs_gender_f.close()
facs_spk_f.close()
tr_facs_f.close()
dt_facs_f.close()
tt_facs_f.close()

print("stored all regularizing factors")

# put the phones with start/end times in a file
talabs = "%s/fac/all_facs_phones.scp" % args.out_dir
class1 = "%s/fac/all_facs_class1.scp" % args.out_dir
class2 = "%s/fac/all_facs_class2.scp" % args.out_dir

talabs_tt = "%s/fac/test_talabs.scp" % args.out_dir
class1_tt = "%s/fac/test_class1.scp" % args.out_dir
class2_tt = "%s/fac/test_class2.scp" % args.out_dir

maybe_makedir(os.path.dirname(talabs))
maybe_makedir(os.path.dirname(class1))
maybe_makedir(os.path.dirname(class2))
maybe_makedir(os.path.dirname(talabs_tt))
maybe_makedir(os.path.dirname(class1_tt))
maybe_makedir(os.path.dirname(class2_tt))

# key=phone, value=phoneid (number)
phone_ids = {}
# key=phone, value=phoneclass
phoneclass1 = {}
phoneclass2 = {}

with open('./misc/timit/timit_phoneclasses.txt', "r") as pid:
    line = pid.readline()  #skip header
    line = pid.readline()
    while line:
        c1 = line.split("\t")[0]
        c2 = line.split("\t")[1]
        phones = line.split("\t")[2]
        for phone in phones.rstrip().split(" "):
            phoneclass1[phone] = c1
            phoneclass2[phone] = c2
        line = pid.readline()

talabs_f = open(talabs, "w")
class1_f = open(class1, "w")
class2_f = open(class2, "w")
talabs_t = open(talabs_tt, "w")
class1_t = open(class1_tt, "w")
class2_t = open(class2_tt, "w")

cnt = 0
for root, _, fnames in sorted(os.walk(args.timit_wav_dir)):
    spk = root.split("/")[-1].lower()

    for fname in fnames:
        if fname.endswith(".phn"):
            uttid = "%s_%s" % (spk, os.path.splitext(fname)[0])
            talabs_f.write(str(uttid)+"\n")
            class1_f.write(str(uttid)+"\n")
            class2_f.write(str(uttid) + "\n")

            if spk in tt_spks:
                talabs_t.write(str(uttid) + "\n")
                class1_t.write(str(uttid) + "\n")
                class2_t.write(str(uttid) + "\n")

            with open(os.path.join(root, fname), "r") as pid:
                line = pid.readline()
                while line:
                    start = line.split(" ")[0]
                    start = float(start)/sam_frame
                    start = int(start)
                    end = line.split(" ")[1]
                    end = float(end)/sam_frame
                    end = int(end)
                    phone = line.split(" ")[2].rstrip()

                    if phone in phone_ids:
                        phone_id = phone_ids[phone]
                    else:
                        phone_id = cnt
                        phone_ids[phone] = phone_id
                        cnt += 1

                    talabs_f.write(str(start)+" "+str(end)+" "+str(phone_id)+"\n")
                    class1_f.write(str(start)+" "+str(end)+" "+str(phoneclass1[phone])+"\n")
                    class2_f.write(str(start)+" "+str(end)+" "+str(phoneclass2[phone])+"\n")

                    if spk in tt_spks:
                        talabs_t.write(str(start) + " " + str(end) + " " + str(phone_id) + "\n")
                        class1_t.write(str(start) + " " + str(end) + " " + str(phoneclass1[phone]) + "\n")
                        class2_t.write(str(start) + " " + str(end) + " " + str(phoneclass2[phone]) + "\n")

                    line = pid.readline()

talabs_f.close()
class1_f.close()
class2_f.close()
talabs_t.close()
class1_t.close()
class2_t.close()

phonetable = "%s/fac/phonetable.txt" % args.out_dir
with open(phonetable, "w") as lid:
    for key in sorted(phone_ids, key=phone_ids.get):
        lid.write(str(phone_ids[key])+" "+str(key)+"\n")

print("stored all talabs")