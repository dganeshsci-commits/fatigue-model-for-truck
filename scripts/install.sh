#!/usr/bin/env bash
# ==============================================================================
#  Raspberry Pi 5 Auto-Installer for Truck Fatigue Detection Model
#  Compatible with: Raspberry Pi OS (64-bit Bookworm / Bullseye), Ubuntu 24.04 ARM64
# ==============================================================================

set -e

echo "======================================================================"
echo " FleetLogix: Installing Truck Fatigue Detection on Raspberry Pi 5"
echo "======================================================================"

# 1. Check Root / Sudo
if [ "$EUID" -eq 0 ]; then
    echo "[Notice] Running as root. System packages will be installed."
else
    echo "[Notice] Running as non-root user. Sudo will be used for apt packages."
fi

SUDO_CMD=""
if [ "$EUID" -ne 0 ]; then
    SUDO_CMD="sudo"
fi

# 2. Install System Dependencies
echo "[1/5] Installing system packages & camera dependencies..."
$SUDO_CMD apt-get update -y
$SUDO_CMD apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    v4l-utils \
    libcamera-tools \
    git \
    curl

# 3. Create Python Virtual Environment
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$INSTALL_DIR"

echo "[2/5] Setting up Python virtual environment in $INSTALL_DIR/venv ..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# 4. Install Python Dependencies
echo "[3/5] Installing Python packages (ONNX Runtime ARM64, OpenCV, etc.)..."
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 5. Run Self-Test
echo "[4/5] Running diagnostic model validation test..."
python scripts/test_inference.py

# 6. Service Setup Prompt
echo "======================================================================"
echo " Installation Complete!"
echo "======================================================================"
echo "To run the standalone edge monitor with live camera:"
echo "  source venv/bin/activate"
echo "  python edge_runner.py"
echo ""
echo "To run the REST API and MJPEG server:"
echo "  source venv/bin/activate"
echo "  python app.py"
echo ""
echo "To install the systemd autostart service, run:"
echo "  sudo cp systemd/truck-fatigue.service /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now truck-fatigue.service"
echo "======================================================================"
