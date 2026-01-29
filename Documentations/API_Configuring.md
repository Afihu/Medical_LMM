You should never commit API keys or any other secrets directly into your code. Instead, we should use environment variables. This method completely separates your secrets from the code. It reads the key from the operating system's environment at runtime, so the key itself is never in the source file.

## Step 1: Create a API key for yourself
- Get into Google AI Studio via this link: https://aistudio.google.com/
- Login with your Google Account (without Google One is fine too)
- Click on "Get API key" at the bottom left of your screen (it should be on top of your account details)
- When seeing the "API key" page, which lists your avaiable keys, click on "Create API key" on the top left of your screen. You can put whatever name you want.
- Copy the API key that you have just created and save it for the next step.

## Step 2: Create a .env file for your secret key
Create a new file in the root of your project named .env (the name starts with a dot). This file will hold your secret keys. It should be in this form:
``` bash
GEMINI_API_KEY="YOUR_KEY_HERE"
```

## Step 2: Create a .gitignore File to Protect Your .env File
This is the most important step. You need to tell Git to ignore the .env file so it's never uploaded. I have added the file name ".env" to .gitignore, so this step should be fine. But if you use another name, then you MUST add it y yourself.

## Step 3: Check for updates in the main.py
Run the updated main.py script as before to check if the key has been updated. If it runs normally, then we are successful.
