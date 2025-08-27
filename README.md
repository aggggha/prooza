# Prooza - An AI Assistant for Your Pricing Needs

This project is an experimental chatbot build with Python, ReactJS (Hopefully) and the Ollama Model to choose. It demonstrate on how an AI Model create a context-aware answers based on given additional trained data via Retrieval-Augmented Generation (RAG).

This repository is part of **Machine Learning : Building AI Usecase using LLM Batch 2** project by Telkom Corporate University.

## Todo 🏁

- [x] Readme file
- [x] Data training preparation
- [x] Load document via RAG
  - [x] *.txt file
  - [x] CSV, JSON, EXCEL File?
- [ ] Create an API Endpoint using Express
- [ ] UI Frontend using ReactJS or Pure HTML & CSS

## Technologies Used ⚙️

| Category                   | Technology / Library                   |
| :------------------------- | :------------------------------------- |
| **Backend**                | Python 3+                              |
| **AI Model**               | Ollama (Default Model: gemma3:1b )     |
| **Web Framework**          | ReactJS (🤞)                |
| **Environment & Packages** | uv                                     |
| **PDF Processing**         | PyPDF2                                 |
| **API Key Management**     | python-dotenv                          |

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1\. Prerequisites

Ensure you have **Python 3.9** or newer installed on your system. You can download it from [python.org](https://www.python.org/).

And also, you'll need [Ollama](https://ollama.com/download) and pick up some model. Although, in this code we'll using Gemma3:1b from Google. So, after downloading the Ollama, go straight to Terminal or CMD (in Windows) and type:

```bash
ollama run gemma3:1b
# or
ollama pull gemma3:1b
```



### 2\. Clone the Repository

Open your terminal and clone this repository:

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 3\. Set Up the Virtual Environment

This project uses `uv` for fast and efficient environment management.

First, install `uv` if you haven't already:

```bash
pip install uv
```

Next, create and activate a virtual environment within the project folder:

```bash
# Create the virtual environment
uv venv

# Activate the environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 4\. Install Dependencies

With the virtual environment activated, install all project dependencies defined in `pyproject.toml` using a single command:

```bash
uv sync
```

This command reads the `pyproject.toml` and `uv.lock` files to create an environment that exactly matches the project's specifications.

## ▶️ Usage

Once the setup is complete, run the Streamlit application from your terminal:

```bash
uv run main.py
```

_(Note: Use `main.py` or `app.py` depending on your main script's filename)._

Your web browser will automatically open to the application's interface.

## 🙏 Disclaimer

This project is brought to you by [Agha Pradipta](mailto:agha.merdekawan@telkom.co.id) and is delivered to you "as is," without any warranties, express or implied. By cloning or using the codes, you agree that you have had the opportunity to inspect it and are accepting all known and unknown defects, faults, and imperfections. 