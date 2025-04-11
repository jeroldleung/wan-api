# wan-api

Providing APIs for accessing self-hosting Wan model.

## Download model

```bash
git lfs install
git clone https://www.modelscope.cn/Wan-AI/Wan2.1-T2V-1.3B-Diffusers.git pretrained_models/Wan2.1-T2V-1.3B-Diffusers
```

## Run as system service

Copy the following system service configuration as `wan-api.service` into `/etc/systemd/system/`

```bash
[Unit]
Description=WAN API Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=YOUR_WORKING_DIRECTORY
ExecStart=YOUR_WORKING_DIRECTORY/.venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Start the system service

```bash
sudo systemctl daemon-reload # Reload systemd to apply changes
sudo systemctl start wan-api # Start the service
sudo systemctl enable wan-api # Optional: enable it to run on boot
```