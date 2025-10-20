# Update 20.10.2025
The system is still using one-shot prompting method. Meaning the result is usable for one time.
I will upgrade it to let user chat subsequently later.
Please take note:
- Make your prompt in /scripts/prompt.txt
- Always leave {user_input} and {cases_section} somewhere in the .txt file. It will be used to add user's input and the top 5 cases later.
- Always run the project in the root folder, i.e ./Medical_LMM with
```bash
uv run main.py
```