#!/usr/bin/env bash
set -xe

date;
MACHINE="martin@tester-desktop.smn.cs.brown.edu";
REMOTE_DIR="/home/martin/projects/pending";
REMOTE_PY="/home/martin/anaconda3/bin/python3";


#rsync -e "ssh -v" -var "adobe-5k/dataset/" "$MACHINE:$REMOTE_DATASET_DIR"; # slash on source copy content instead of dir
scp *.py "$MACHINE:$REMOTE_DIR";
#ssh "$MACHINE" "cd $REMOTE_DIR; nohup sh -c \"$REMOTE_PY build.py 2>&1 | tee out.log \" & ";
ssh "$MACHINE" "cd $REMOTE_DIR; nohup sh -c \"$REMOTE_PY visualize.py 2>&1 | tee out.log \" & ";
scp "$MACHINE:$REMOTE_DIR/hist.png" .
scp "$MACHINE:$REMOTE_DIR/records.npy" .




