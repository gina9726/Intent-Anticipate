user='C'
model='model/C_RL-RNN/RL-RNN-500'
for percent in 0.25 1
do
python main.py --task test --user ${user} --net ${model} --percent ${percent} \
&> logfile/testC_p${percent}.log
done
