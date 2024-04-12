# Useful System Commands

This document contains a collection of useful system management commands for developers working with machine learning and data science projects.

## GPU Management
Monitor GPU usage and performance in real-time:
```bash
watch -n 0.5 nvidia-smi
```

Second method:
```bash
nvidia-smi dmon
```

## Memory and Process Management
Check memory usage:
```bash
watch -n 0.5 free -h
```

## Network
Check all open ports:
```bash
netstat -tuln
```

## CPU Temperature Monitoring
Check CPU temperature to ensure your system is not overheating during intensive computing tasks:

On Linux, you can use `sensors` from the `lm-sensors` package:
* Installation
```bash
sudo apt-get install lm-sensors
sudo sensors-detect
```
* Usage
```bash
watch -n 0.5 sensors
```
