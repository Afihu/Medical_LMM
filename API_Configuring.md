You should never commit API keys or any other secrets directly into your code.

Instead, use environment variables. This method completely separates your secrets from your code. The code reads the key from the operating system's environment at runtime, so the key itself is never in the source file.

## Step 1: Create a .env File for Your Secret Key
Create a new file in the root of your project named .env (the name starts with a dot). This file will hold your secret keys. It should be in this form:
``` bash
GEMINI_API_KEY="YOUR_KEY_HERE"
```

## Step 2: Create a .gitignore File to Protect Your .env File
This is the most important step. You need to tell Git to ignore the .env file so it's never uploaded. I have added the file name ".env" to .gitignore, so this step should be fine. But if you use another name, then you MUST add it y yourself.

## Step 3: Check for updates in the main.py
Run the updated main.py script as before to check if the key has been updated.

