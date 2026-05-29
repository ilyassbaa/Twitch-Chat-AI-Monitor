# Code-Switching Detection: Darija, French, and English

## Project Overview
This project implements a Natural Language Processing (NLP) pipeline designed to detect and classify **Code-Switching** in trilingual Moroccan text. The system automatically identifies segments of **Darija**, **French**, and **English** within mixed-language sentences.

This project was developed as part of the academic curriculum in Computer Science at **Faculté Polydisciplinaire de Khouribga (FP Khouribga)**.

## Authors
* **Ilyass Bahri**
* **Oussama El Hichami**

## Project Architecture
The system utilizes a fine-tuned **XLM-RoBERTa** model to perform token classification. The pipeline is designed for local processing, ensuring data privacy and utilizing AMD GPU acceleration (via DirectML).

## Key Features
* **Trilingual Detection:** Classifies input tokens into Darija (DA), French (FR), and English (EN) labels using a BIO tagging schema.
* **Stateless Privacy:** The application is designed with privacy in mind. User inputs are processed in-memory and are never persisted to disk, ensuring that no sensitive user data is stored.
* **User Interface:** A real-time web interface built with **Gradio** to visualize token-level language identification.
* **Hardware Optimized:** Leverages local AMD GPU resources for high-performance training and inference.

## Technologies Used
* **Languages:** Python 3.11
* **Frameworks:** PyTorch, Hugging Face Transformers (`XLM-RoBERTa`)
* **Libraries:** Pandas (Data manipulation), Gradio (Web UI), Torch-DirectML (GPU support)
* **Data Format:** CSV-based token classification schema

## Getting Started

### Prerequisites
Ensure you have Python 3.11 installed. Clone this repository and set up your virtual environment:

```bash
# Clone the repository
git clone https://github.com/ilyassbaa/Language-Switching-Detection.git
cd Language-Switching-Detection

# Create and activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```

### Running the Application

Once dependencies are installed, launch the web interface with:

```bash
python app.py

```

Open the local URL provided in the terminal (usually `http://127.0.0.1:7860`) in your browser to start the detection demo.

### Example Input

Try testing the model with this trilingual sentence:

> *slm monsieur 3afak you can add chwiya diyal les mots*

The system will output a color-coded analysis showing the Darija, French, and English segments.
![Code-Switching Detection Interface](assets/interface.png)
