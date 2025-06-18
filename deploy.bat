@echo off
echo Removing existing git...
rmdir /s /q .git

echo Initializing new git repository...
git init

echo Adding files...
git add app.py
git add trainmodel.py
git add requirements.txt
git add matches.csv
git add wsgi.py
git add .gitignore
git add README.md

echo Creating commit...
git commit -m "Initial IPL API deployment"

echo Setting up remote...
git remote add origin https://github.com/ayushman46/IPL_api.git

echo Pushing to main branch...
git branch -M main
git push -f origin main

echo Done!
pause
