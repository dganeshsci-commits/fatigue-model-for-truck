# Raspberry Pi 5 Systemd Service Setup

To enable the Truck Fatigue Detection service to automatically start when the Raspberry Pi 5 boots in the truck:

## 1. Edit Paths (if your username or path differs)
Open `truck-fatigue.service` and verify `User=pi` and `WorkingDirectory=/home/pi/fatigue-model-for-truck`.

## 2. Copy and Enable Service
```bash
# Copy unit file to systemd directory
sudo cp systemd/truck-fatigue.service /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable service to start on boot
sudo systemctl enable truck-fatigue.service

# Start service immediately
sudo systemctl start truck-fatigue.service
```

## 3. Check Service Status & Logs
```bash
# View live status
sudo systemctl status truck-fatigue.service

# Follow real-time logs
journalctl -u truck-fatigue.service -f
```

## 4. Stopping or Disabling
```bash
sudo systemctl stop truck-fatigue.service
sudo systemctl disable truck-fatigue.service
```
