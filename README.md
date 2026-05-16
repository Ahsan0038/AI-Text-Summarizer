# AI Text Summarizer Web Application

## Project Overview

This project is a lightweight AI-based Text Summarizer Web Application developed using Flask, Python, NLTK, and NetworkX. The application generates extractive summaries from user-provided text using the TextRank algorithm.

The application provides:

* Flask-based web interface
* REST API endpoint for Postman testing
* Threading implementation
* Unit testing support
* Error handling and validation

The project is lightweight and works without GPU support.

---

# Original Project Credit

This project is based on the following open-source repository:

Original Repository:
https://github.com/DheerajKumar97/Text-Summarizer-Flask-Deployment

Original Author:
Dheeraj Kumar

This project was modified and enhanced for Software Construction and Development Lab requirements.

---

# Features

* AI-based text summarization
* Flask web application
* HTML/CSS frontend
* REST API support
* Threading implementation
* Unit testing using unittest
* Error handling and validation
* Lightweight CPU-compatible execution

---

# Technologies Used

* Python
* Flask
* HTML/CSS
* NLTK
* NetworkX
* NumPy
* Regex
* Threading
* unittest

---

# How the Model Works

The application uses the TextRank algorithm for extractive text summarization.

## Working Process

1. Input text is divided into sentences
2. Sentence similarity is calculated
3. A similarity graph is created
4. PageRank ranks the important sentences
5. Top-ranked sentences are returned as summary

---

# Input and Output

## Input

* User text
* Number of summary sentences

## Output

* AI-generated summarized text

---

# Additional Improvements

The following modifications were added to the original project:

* Added threading support
* Added unittest testing
* Added API endpoint for Postman
* Added input validation
* Added improved error handling
* Added deployment support files
* Improved project structure and comments

---

# Project Structure

Text-Summarizer-Flask-Deployment/

* app.py
* testing.py
* requirements.txt
* README.md
* Dockerfile
* Procfile
* templates/
* static/

---

# Installation Steps

## Clone Repository

git clone YOUR_GITHUB_REPOSITORY_LINK

## Open Project Folder

cd Text-Summarizer-Flask-Deployment

## Install Dependencies

pip install -r requirements.txt

## Run Application

py -3.8 app.py

---

# Web Application Usage

1. Open browser
2. Go to:

http://127.0.0.1:7860

3. Enter text
4. Select summary sentence count
5. Click Summarize

---

# API Testing using Postman

## Endpoint

POST /api/summarize

## Sample JSON

{
"text": "Artificial Intelligence is changing the world. Machine learning helps systems learn from data.",
"num_sentences": 2
}

---

# Unit Testing

Run:

py -3.8 testing.py

---

# Advantages

* Lightweight
* Fast summarization
* Easy to use
* CPU compatible
* Educational AI project

---

# Limitations

* Extractive summarization only
* Works best on structured text
* Not designed for large documents

---

# Future Improvements

* Abstractive summarization
* Multi-language support
* PDF upload support
* Better UI design

---

# Deployment

The project can be deployed on:

* Hugging Face Spaces
* Render
* Railway

---

# Author

Modified and enhanced for academic purposes and Software Construction and Development Lab Project.
