#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export PCCL_LOG_LEVEL=DEBUG

python $SCRIPT_DIR/master.py &
MASTER_PID=$!
echo "Started master.py with PID $MASTER_PID"

sleep 5

python $SCRIPT_DIR/peer.py &
PEER1_PID=$!
echo "Started first peer.py with PID $PEER1_PID"

python $SCRIPT_DIR/peer.py &
PEER2_PID=$!
echo "Started second peer.py with PID $PEER2_PID"

cleanup() {
    echo "Terminating master.py with PID $MASTER_PID"
    kill $MASTER_PID
    exit
}

trap cleanup SIGINT SIGTERM

wait $PEER1_PID
echo "First peer.py (PID $PEER1_PID) has exited."

wait $PEER2_PID
echo "Second peer.py (PID $PEER2_PID) has exited."

echo "Both peers have exited. Terminating master.py."
kill $MASTER_PID

wait $MASTER_PID
echo "master.py has been terminated."
