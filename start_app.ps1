py -3.13 -m venv venv_313
.\venv_313\Scripts\python.exe -m pip install -r requirements.txt
Start-Process -NoNewWindow -FilePath ".\venv_313\Scripts\python.exe" -ArgumentList "app\app.py"
Start-Sleep -Seconds 5
Start-Process -NoNewWindow -FilePath "npx.cmd" -ArgumentList "localtunnel --port 5000" -RedirectStandardOutput "lt.txt"
Start-Sleep -Seconds 5
Get-Content lt.txt
