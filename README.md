# Intent-Anticipate

Here are some codes used in the paper "Anticipating Daily Intention using On-Wrist Motion Triggered Sensing" \[1\].
The dataset and more details can be found in this [website](http://aliensunmin.github.io/project/intent-anticipate/).

\[1\] Anticipating Daily Intention using On-Wrist Motion Triggered Sensing, by Tz-Ying Wu*, Ting-An Chien*, Cheng-Sheng Chan, Chan-Wei Hu, Min Sun, ICCV 2017.

## Usage
Training:
```
python main.py --task train --user ${user} --net ${model} --save_dir ${save_dir} &> ${logfile}
```
Testing:
```
python main.py --task test --user ${user} --net ${model} --percent ${percent} &> ${logfile}
```
