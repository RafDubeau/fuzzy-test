#!/bin/bash

directory="./local_dependencies"

# Create the directory if it does not exist
if [[ ! -d "$directory" ]]; then
    echo "Directory does not exist. Creating now"
    mkdir $directory
else
    echo "Directory exists. Emptying now"
    rm -rf $directory/*
fi

# Install FuzzyTypes into the local directory
pip install -t $directory git+https://github.com/bright-blue-im/FuzzyTypes.git

