#!/bin/bash

# Check local IP address
local_ip="ipconfig getifaddr en0"
localip=$(eval "$local_ip")
echo "The output is: $localip"

# Check connection to a site
wget -q --spider http://google.com
if [ $? -ne 0 ]; then
    echo "connection to site failed" 
fi

# Download a document
url="https://dmtn-226.lsst.io/DMTN-226.pdf"
wget "$url"
if [ $? -ne 0 ]; then
    echo "Command failed with exit status $?"
fi

# Check file exists locally
file_path=$(basename $url)
if [ ! -f "$file_path" ]; then
    echo "File does not exist: $file_path"
fi
rm $file_path