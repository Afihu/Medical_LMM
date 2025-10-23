# Update 23.10.2025
Turns out Qdrant does not support partial vector updates.
Two separate scripts for uploading text vectors and image vectors will be made. This will create 2 different Collections for a cases (either text_vec or image_vec, or both)
ETA: 2pm 24.10.2025
For more information, see at [General Update](GENERAL_UPDATE.md)

# Update 22.10.2025
I cannot think of a way except pushing all the content from the previous run, which is inefficient.
I will try to figure it out later 

# Update 21.10.2025
I have found a way that make Gemini remember the chat contents, even after we close the terminal (chat). 
We will upgrade the chatting methods into few-shot prompting. ETA: 4pm 22.10.2025

# Update 20.10.2025
The system is still using one-shot prompting method. Meaning the result is usable for one time.
I will upgrade it to let user chat subsequently later.
Also currently no user image yet. I will do it later (optimally will be finished before Wednesday morning)
Please take note:
- Make your prompt in /scripts/prompt.txt
- Always leave {user_input} and {cases_section} somewhere in the .txt file. It will be used to add user's input and the top 5 cases later.
- Always run the project in the root folder, i.e ./Medical_LMM with
```bash
uv run main.py
```