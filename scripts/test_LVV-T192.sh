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

# Run a speed test 
VENV_NAME="LVV-T192_env"

echo "Creating virtual environment '$VENV_NAME'..."
python3 -m venv "$VENV_NAME"

source "$VENV_NAME/bin/activate"
if [ $? -eq 0 ]; then
    echo "Virtual environment '$VENV_NAME' activated."

    # Install speedtest package in venv
    package="speedtest-cli"
    echo "Installing $package..."
    pip install "$package"

    # Check if the installation was successful
    if [ $? -eq 0 ]; then
        echo "$PACKAGE installed successfully."
    else
        echo "Failed to install $PACKAGE."
        exit 1
    fi

    # Run speed test
    echo "Checking internet speed..."
    output=$(speedtest-cli --simple)

    download_speed=$(echo "$output" | grep 'Download:' | awk '{print $2, $3}')
    upload_speed=$(echo "$output" | grep 'Upload:' | awk '{print $2, $3}')
    ping_time=$(echo "$output" | grep 'Ping:' | awk '{print $2, $3}')

    echo "Parsed Speed Test Results:"
    echo "Download Speed: $download_speed"
    echo "Upload Speed: $upload_speed"
    echo "Ping Time: $ping_time"
fi

# Deactivate and remove the venv
echo "Deactivating the virtual environment '$VENV_NAME'..."
deactivate
echo "Removing the virtual environment directory '$VENV_NAME'..."
rm -rf "$VENV_NAME"
