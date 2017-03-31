user='C'
model='model/CcombFT_oR10/model-500'
save_dir='model/C_RL-RNN'
logfile='logfile/trainC.log'
python main.py --task train --user ${user} --net ${model} --save_dir ${save_dir} \
&> ${logfile}
