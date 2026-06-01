@echo off
cloudflared tunnel --url http://localhost:8000 >> "C:\Users\icmr\Dev\docker-api\cloudflared.log" 2>&1
