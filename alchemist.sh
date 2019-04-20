#!/usr/bin/env bash
set -xe

date;
MACHINE="martin@tester-desktop.smn.cs.brown.edu";
REMOTE_DIR="/home/martin/projects/pending";
REMOTE_PY="/home/martin/anaconda3/bin/python3";


#rsync -e "ssh -v" -var "adobe-5k/dataset/" "$MACHINE:$REMOTE_DATASET_DIR"; # slash on source copy content instead of dir
scp -r *.py "$MACHINE:$REMOTE_DIR";

#ssh "$MACHINE" "cd $REMOTE_DIR; nohup sh -c \"$REMOTE_PY build.py 2>&1 | tee out.log \" & ";
#ssh "$MACHINE" "cd $REMOTE_DIR; nohup sh -c \"$REMOTE_PY visualize.py 2>&1 | tee out.log \" & ";
#ssh "$MACHINE" "cd $REMOTE_DIR; nohup $REMOTE_PY gas_station.py txpool 2>&1 > gas_station.log &";
ssh "$MACHINE" "cd $REMOTE_DIR; nohup $REMOTE_PY gas_station.py predtable 2>&1 > predtable.log &";
#ssh "$MACHINE" "cd $REMOTE_DIR; nohup sh -c \"$REMOTE_PY features.py 2>&1 | tee features.log \"& ";
#scp "$MACHINE:$REMOTE_DIR/hist.png" .
#scp "$MACHINE:$REMOTE_DIR/records.npy" .




