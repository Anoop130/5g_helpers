#!/bin/bash

# === CONFIG ===
ATTACK_LIST=("cp_dl_long_10mb_DR_1000.pcap")
INJECTION_POINTS=("DU" "RU")
CRASH_LOG="crash_logs.txt"
RESULTS_DIR="$PWD/attack_results"

RU_DIR="$HOME/srsRAN_Project/build/apps/examples/ofh"
RU_CMD="ru_emulator -c emu_dpdk.yaml"

GNB_DIR="$HOME/srsRAN_Project/build/apps/gnb"
GNB_CMD="./gnb -c gnb_dpdk.yaml -c $HOME/srsRAN_Project/configs/testmode.yml"

ATTACKER_DIR="$HOME/Dos-attacker"
INTERFACE_DU="enp5s0f1"
INTERFACE_RU="enp8s0f0np0"

# === CLEANUP PREVIOUS RESULTS ===
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
echo -n > "$CRASH_LOG"

for ATTACK_FILE in "${ATTACK_LIST[@]}"; do
  for INJ in "${INJECTION_POINTS[@]}"; do
    ATTACK_NAME="${ATTACK_FILE%.pcap}"
    ATTACK_DIR="$RESULTS_DIR/${ATTACK_NAME}_INJ_${INJ}"
    mkdir -p "$ATTACK_DIR"

    echo "=== Starting attack: $ATTACK_FILE | Injection Point: $INJ ==="

# ==============================gNB=========================================

    echo "[DEBUG] Launching gNB via expect..."
    cd "$GNB_DIR"
    gnb_log="$ATTACK_DIR/gnb_${ATTACK_NAME}.log"

    "$ATTACKER_DIR/start_gnb.expect" $GNB_CMD > "$gnb_log" 2>&1 &
    gnb_pid=$!
    echo "[DEBUG] gNB PID: $gnb_pid"

    echo "[DEBUG] Waiting 10s for gNB to stabilize..."
    sleep 10


# ======================== ru ========================================

    # Launch RU
    echo "[DEBUG] Launching ru_emulator..."
    cd "$RU_DIR"
    sudo $RU_CMD > "$ATTACK_DIR/ru_${ATTACK_NAME}_${INJ}.log" 2>&1 &
    ru_pid=$!
    echo "[DEBUG] RU PID: $ru_pid"

    # Wait 5s for system to stabilize
    echo "[DEBUG] Waiting 5s pre-attack to stabilize"
    sleep 5

    # Set interface
    if [ "$INJ" == "DU" ]; then
      INTF="$INTERFACE_DU"
    else
      INTF="$INTERFACE_RU"
    fi

# ======================= attacker ========================================

    # Launch attacker
    echo "[DEBUG] Launching attacker on $INTF..."
    echo "[DEBUG] Start time: [$(date '+%Y-%m-%d %H:%M:%S')]"
    cd "$ATTACKER_DIR"
    sudo tcpreplay --intf1=$INTF --loop=0 --topspeed "Traffic/custom/$ATTACK_FILE" &
    attacker_pid=$!
    echo "[DEBUG] Attacker PID: $attacker_pid"

    echo "[DEBUG] Attacking for 15 seconds"

    # === Monitor for 15 seconds ===
    crashed=0
    for ((i=0; i<15; i++)); do
      sleep 1
      if ! kill -0 $ru_pid 2>/dev/null; then
        echo "[CRASH] ru_emulator crashed during $ATTACK_FILE injection $INJ" | tee -a "$CRASH_LOG"
        crashed=1
        break
      fi
      if ! kill -0 $gnb_pid 2>/dev/null; then
        echo "[CRASH] gNB crashed during $ATTACK_FILE injection $INJ" | tee -a "$CRASH_LOG"
        crashed=1
        break
      fi
    done

    # Stop attacker after 15s
    echo "[DEBUG] Stopping attacker..."
    sudo kill $attacker_pid 2>/dev/null

    # If crashed, stop everything immediately
    if [ $crashed -eq 1 ]; then
      echo "[DEBUG] Terminating RU and gNB due to crash..."
      sudo kill $ru_pid 2>/dev/null
      sudo kill $gnb_pid 2>/dev/null
      echo "Crash occurred. Skipping post-attack delay."
      echo "----------------------------------------"
      continue
    fi

    # Wait extra 5s after attacker completes
    echo "[DEBUG] Waiting extra 5s post-attack..."
    sleep 5

    # Stop RU and gNB
    echo "[DEBUG] Stopping ru_emulator..."
    sudo kill $ru_pid 2>/dev/null
    echo "[DEBUG] Stopping gNB..."
    sudo kill $gnb_pid 2>/dev/null

    echo "✅ Completed $ATTACK_FILE | Injection $INJ"
    echo "----------------------------------------"
  done
done

echo "[DEBUG] Cleaning up any stray processes..."
sudo pkill -f ru_emulator
sudo pkill -f gnb
sudo pkill -f tcpreplay

echo "All attacks completed. See logs in: $RESULTS_DIR"