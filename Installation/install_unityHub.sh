#!/bin/bash
echo "##########################"
echo "# Unity Hub installation #"
echo "##########################"
echo ""
echo ""


echo "Adding Unity Hub repository"

sudo sh -c 'echo "deb https://hub.unity3d.com/linux/repos/deb stable main" > /etc/apt/sources.list.d/unityhub.list'
wget -qO - https://hub.unity3d.com/linux/keys/public | sudo apt-key add -

echo "Update and install Unity Hub"
sudo apt update
sudo apt-get install unityhub