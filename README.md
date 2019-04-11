## Author:
Tianzuo Zhang(SID:470085460), Dehong Liang(SID:470188761) and Shanshan Luo(SID:470349580)

## Tool environment
* python 3.7.0
* numpy 1.15.0
* sklearn 0.0

##File structure
We have total 2 files : classification.py , mnist_reader.py
and an empty folder "data/fashion" storing data set.

## How to use
use the following command to download the data set from github 
>git clone git@github.com:zalandoresearch/fashion-mnist.git

then,move the totally 4 training and testing files to the "data/fashion" from where 
our program will read these files.
The mnist_reader is responsible to read and parse .gz files from that destination

finally,run our main classification file
>python3 classification.py

##Result
The result predicts the 10000 testing data
