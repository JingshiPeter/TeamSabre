#! /usr/bin/expect -f
spawn ssh js79735@me-dimitrovresearch.engr.utexas.edu
expect -re "assword: *$"
sleep 1
send "Y2MwOWViZjc0Yzg2OTU3Y2QzODBlNWQx\r"
interact
