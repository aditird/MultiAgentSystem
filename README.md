🚀 Multi-Agent UI State Capture System

Submitted by: Aditi R. Deshpande

This project implements a multi-agent system that autonomously explores web applications, interacts with UI elements, and captures screenshots + metadata for each state. It supports real-time question generation, dynamic navigation, and handling of non-URL UI states such as modals.

🔧 How It Works
Agent A – Reasoning Agent (DialoGPT-small)

Generates intelligent navigation questions based on the user’s goal.

Uses conversation history to create follow-up questions in real time.

Agent B – Browser Exploration Agent (Playwright)

Launches the live website and detects interactive elements.

Interprets Agent A’s questions to decide where to click or navigate.

Captures screenshots and UI metadata at every step.

The system runs this loop for ~6 steps, creating a trace of UI states that form a full workflow.

📦 Repository Contents
main_app.py        # Main multi-agent system
requirements.txt   # Dependencies
hf_key.txt         # HuggingFace API key (add manually)
workflows/         # Auto-generated screenshots + workflow JSON

📝 Prerequisites

Create a HuggingFace token (READ access):
https://huggingface.co/settings/tokens

Save it in a file named hf_key.txt in the project directory.

Install dependencies:

pip install -r requirements.txt
playwright install

▶️ Running the Program

Run:

python main_app.py


You will see:

1. Run AI-powered test (Wikipedia)
2. Interactive mode with AI question generation
3. Quit


Choose 1 for a test or 2 to enter your own goal and URL (e.g., “Create project in Linear”).

⚙️ Notes

If the site requires login (e.g., Google Sign-In), the program pauses and waits for you to authenticate manually.

If a modal window requires manual input, the program pauses until you interact with it.

📊 Example Tasks Supported

Search articles in Wikipedia

Create project in Linear

Filter a database in Notion

Create tasks in Asana

Search products on Amazon

Explore repositories on GitHub

Each run generates a workflow folder containing screenshots (state_00.png, state_01.png, …) and workflow.json.

📸 Output

The system automatically saves:

Full-page screenshots

URLs, titles, number of interactive elements

Modal/form detection

AI-generated questions for each step

View results inside the workflows/ directory.
