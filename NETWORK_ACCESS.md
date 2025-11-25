# Network Access Guide

This guide explains how to make the Whisper transcription app accessible to others on your local network.

## Quick Setup

The app is already configured to accept network connections. When you run `python app.py`, it will:

1. Start the server on `0.0.0.0:7860` (accessible from network)
2. Display your local IP address
3. Show access URLs for both local and network access

## Access URLs

When you start the app, you'll see:

```
Local access:    http://127.0.0.1:7860
Network access:  http://192.168.0.230:7860
```

### For You (on the same computer):
- Use: `http://127.0.0.1:7860` or `http://localhost:7860`

### For Others (on the same network):
- Use: `http://YOUR_IP_ADDRESS:7860` (e.g., `http://192.168.0.230:7860`)

## Finding Your IP Address

### Windows:
```powershell
ipconfig | findstr /i "IPv4"
```

### Alternative Methods:
1. Open Command Prompt
2. Type: `ipconfig`
3. Look for "IPv4 Address" under your active network adapter

## Firewall Configuration

### Windows Firewall

You may need to allow connections through Windows Firewall:

1. **Open Windows Defender Firewall:**
   - Press `Win + R`
   - Type: `wf.msc`
   - Press Enter

2. **Create Inbound Rule:**
   - Click "Inbound Rules" → "New Rule"
   - Select "Port" → Next
   - Select "TCP"
   - Enter port: `7860`
   - Select "Allow the connection"
   - Apply to all profiles (Domain, Private, Public)
   - Name it: "Whisper Transcription App"
   - Click Finish

### Quick Firewall Command (Run as Administrator):
```powershell
netsh advfirewall firewall add rule name="Whisper App" dir=in action=allow protocol=TCP localport=7860
```

## Public Internet Access (Optional)

If you want to share the app publicly over the internet:

### Option 1: Gradio Share (Temporary)
Change in `app.py`:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=True,  # Creates a public Gradio link (expires after 72 hours)
    ...
)
```

This creates a temporary public URL like: `https://xxxxx.gradio.live`

### Option 2: Port Forwarding (Permanent)
1. Configure your router to forward port 7860 to your computer
2. Find your public IP: https://whatismyipaddress.com
3. Others can access: `http://YOUR_PUBLIC_IP:7860`

⚠️ **Security Warning:** Public access exposes your app to the internet. Consider:
- Adding authentication
- Using HTTPS
- Limiting access to trusted users

## Troubleshooting

### "Connection Refused" Error
- Check Windows Firewall settings (see above)
- Ensure the app is running
- Verify the IP address is correct
- Make sure devices are on the same network

### Can't Access from Mobile/Other Device
- Ensure both devices are on the same Wi-Fi network
- Check firewall isn't blocking the connection
- Try accessing from the same device first to verify it works

### Port Already in Use
If port 7860 is already in use, change it in `app.py`:
```python
demo.launch(
    server_name="0.0.0.0",
    server_port=7861,  # Change to another port
    ...
)
```

## Security Considerations

### For Local Network Only:
- ✅ Safe for home/office networks
- ✅ No internet exposure
- ⚠️ Anyone on your network can access

### For Public Internet:
- ⚠️ Exposes your app to the internet
- ⚠️ No authentication by default
- ⚠️ Consider adding security measures

## Testing Network Access

1. Start the app: `python app.py`
2. Note the network IP address shown
3. On another device (phone, tablet, another computer):
   - Connect to the same Wi-Fi network
   - Open a web browser
   - Navigate to: `http://YOUR_IP:7860`
   - The app should load!

## Example Usage

```
Server starting...
Local access:    http://127.0.0.1:7860
Network access:  http://192.168.0.230:7860

Others on your network can access the app at:
  → http://192.168.0.230:7860
```

Then share `http://192.168.0.230:7860` with others on your network!

