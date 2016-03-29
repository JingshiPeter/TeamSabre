#! /opt/local/bin/expect -f
spawn ssh your_eid@me-dimitrovresearch.engr.utexas.edu
expect -re "assword: *$"
sleep 1
send "your_password\r"
interact
