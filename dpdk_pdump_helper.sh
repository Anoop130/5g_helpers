#!/bin/bash

if [ $EUID -eq 0 ]; then
	echo "This script should not be run as root"
	exit 1
fi

SCRIPT_DIR=$(realpath .)

if [ -e build ]; then
	echo "build exists exiting..."
	exit 0
fi

if [ -z DPDK_Makefile ]; then
	echo "No makefile exiting..."
	exit 1
fi

PS4="[CUFH attacker build] "
set -x

# Get the correct DPDK version 19.11.14
wget https://fast.dpdk.org/rel/dpdk-19.11.14.tar.xz

# Extract source code
tar -xvf ./dpdk-19.11.14.tar.xz
rm -f ./dpdk-19.11.14.tar.xz

cd dpdk-stable-19.11.14/

make defconfig O=x86_64-native-linuxapp-gcc

echo "CONFIG_RTE_LIBRTE_PDUMP=y" >> x86_64-native-linuxapp-gcc/.config
echo "CONFIG_RTE_LIBRTE_PMD_PCAP=y" >> x86_64-native-linuxapp-gcc/.config

make -j$(nproc) O=x86_64-native-linuxapp-gcc/

cd $SCRIPT_DIR

export RTE_SDK="$SCRIPT_DIR/dpdk-stable-19.11.14/"
export RTE_TARGET="$SCRIPT_DIR/x86_64-native-linuxapp-gcc"

# make -f DPDK_Makefile