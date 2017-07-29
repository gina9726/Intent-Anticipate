for user in A B C
do
model=model/${user}_nopre_comb_new_RL-RNN_100_-100_ro1_lr3_4_r1_zp_s2/RL-RNN-500
python main_vis.py --task test --user ${user} --net ${model} --percent 1 --trainR 0.99
done
