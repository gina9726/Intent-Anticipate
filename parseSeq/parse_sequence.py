#-*- coding: utf-8 -*-
import numpy as np
import pdb, os

seqid2int = [] 
vocab2idx = dict()
count = 1 
intentionNum = 0
with open('seq.csv') as f:
    for lines in f:
        line = lines.split(',')
        seq = line[0].rstrip('\r\n')
        if seq != '':
            # build action vocabulary
            for action in line[3:]:
                if action.rstrip('\r\n') == '':
                    break
                else:
                    if action.rstrip('\r\n') not in vocab2idx:
                        vocab2idx[action.rstrip('\r\n')] = count
                        count += 1
        else:
            intentionNum += 1
    seqid2seq = []
    intentionID = len(vocab2idx)
    totalID = 115 # because some intention appears in atomic action
    f.seek(0)
    tmp_int = False
    for lines in f:
        line = lines.split(',')
        seq = line[0].rstrip('\r\n')
        if seq == '':
            intention = line[1].rstrip('\r\n')
            if intention not in vocab2idx:
                if tmp_int:
                    intentionID = tmp_int + 1
                    tmp_int = False
                else:
                    intentionID += 1
                vocab2idx[intention] = intentionID # len(vocab2idx) 
            else:
                tmp_int = intentionID
                intentionID = vocab2idx[intention]
        else:
            try:
                seq = int(seq)
                seqid2int.append(intentionID)
                tmp_seq = [] # max sequence length 7, add intetion and EOS to the end
                for action in line[3:]:
                    if action.rstrip('\r\n') in vocab2idx:
                        tmp_seq.append(vocab2idx[action.rstrip('\r\n')])
                tmp_seq.append(totalID) # add EOS at the end
                seqid2seq.append(np.array(tmp_seq))
            except:
                pass
    vocab2idx['EOS'] = totalID # (intentionID + 1) 
    vocab2idx['None'] = 0
    idx2vocab = {v: k for k, v in vocab2idx.items()}
    for k, v in vocab2idx.items():
        print k, vocab2idx[k]

seqid2seq = np.array(seqid2seq)
seqid2int = np.array(seqid2int)
seqid2int = np.expand_dims(seqid2int, 1)
seqrecord={}
for i in xrange(len(seqid2int)):
    if seqid2int[i] not in seqrecord.keys():
        seqrecord[int(seqid2int[i])] = 1
    else:
        seqrecord[int(seqid2int[i])] +=1
for i in seqrecord.keys():
    print i, seqrecord[i]

def parse_order(ratio):
    filename = os.path.join("parseSeq/ReduceTable",str(ratio)+".npy")
    if os.path.isfile(filename):
        print "reading from table"
        Table = np.load(filename)
    else:
        print "Create %f Table"%(ratio)
        filename = os.path.join("parseSeq/ReduceTable",str(ratio))
        count={}
        for i in seqrecord.keys():
            count[i] = np.ceil(seqrecord[i]*ratio)
        Table = np.int8(np.zeros(len(seqid2int)))
        idx=0
        for i in xrange(len(seqrecord.keys())):
            ID = int(seqid2int[idx])
            selected = np.random.choice(seqrecord[ID],int(count[ID]),replace=False).tolist()
            print ID,' select ',selected
            for j in xrange(seqrecord[ID]):
                if j in selected:
                    Table[idx+j]=1
                    print i,'-',j
                else:
                    Table[idx+j]=0
            idx+=seqrecord[ID]
        np.save(filename,Table)

    return Table

def parse_sensor(filePath):
    with open(filePath) as f:
        acc_data = np.zeros((0,3), dtype=float)
        gyro_data = np.zeros((0,3), dtype=float)
        for line in f:
            [timestamp, x_gyro, y_gyro, z_gyro, x_acc, y_acc, z_acc] = line.rstrip('\n').split(',')
            if timestamp != 'Time':
                acc_tmp = np.asarray([[float(x_acc), float(y_acc), float(z_acc)]])
                gyro_tmp = np.asarray([[float(x_gyro), float(y_gyro), float(z_gyro)]])
                acc_data = np.concatenate([acc_data, acc_tmp], axis=0)
                gyro_data = np.concatenate([gyro_data, gyro_tmp], axis=0)
        acc_data = np.asarray(acc_data)
        gyro_data = np.asarray(gyro_data)
        
        return acc_data
