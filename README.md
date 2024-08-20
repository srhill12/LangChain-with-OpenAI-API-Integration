
# LangChain with OpenAI API Integration

This repository contains a series of Python scripts demonstrating the use of LangChain with OpenAI's GPT-3.5-turbo model for various tasks, including document question answering, Wikipedia content analysis, and interacting with external APIs. The scripts showcase how to leverage the power of large language models (LLMs) to process and analyze different types of data.

## Project Structure

- **Model and API Key Management:** The script loads environment variables using the `dotenv` package to securely manage API keys.
- **Documents Analysis:** Includes examples of loading and processing PDF documents and Wikipedia content for question-answering tasks.
- **External APIs:** Demonstrates how to integrate and interact with external APIs, such as the Numbers API and the New York Times API, using LangChain.

## Prerequisites

Before running the scripts, ensure that you have the following installed:

- **Python 3.x**
- **Required Python Packages:** You can install the necessary packages using pip:
  ```bash
  pip install langchain_openai langchain langchain_community python-dotenv
  ```

## Setting Up the Environment

### 1. Create a `.env` File

To securely manage your API keys, create a `.env` file in the root directory of this project. This file should contain the following environment variables:

```plaintext
OPENAI_API_KEY=your_openai_api_key_here
NYT_API_KEY=your_nyt_api_key_here
```

Replace `your_openai_api_key_here` and `your_nyt_api_key_here` with your actual API keys.

### 2. Ensure `.env` is Not Included in Version Control

To prevent accidental exposure of your API keys, make sure that the `.env` file is listed in your `.gitignore` file:

```plaintext
.env
```

## How to Run the Scripts

### 1. Model and API Key Setup

The script starts by loading environment variables and initializing the OpenAI model:

```python
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Set model name and API key
OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 2. Documents Analysis

#### PDF Document Analysis

The script loads and analyzes PDF documents using the `PyPDFLoader`:

```python
from langchain.document_loaders import PyPDFLoader
from langchain.chains.question_answering import load_qa_chain

# Load PDF document
pdf_loader = PyPDFLoader('path_to_your_pdf.pdf')
documents = pdf_loader.load()

# Initialize the model and QA chain
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)
chain = load_qa_chain(llm)

# Query the document
query = 'Your query here'
result = chain.invoke({"input_documents": documents, "question": query})
print(result["output_text"])
```

#### Wikipedia Content Analysis

The script also demonstrates how to load and query Wikipedia content:

```python
from langchain_community.document_loaders import WikipediaLoader

# Load Wikipedia content
wiki_topic = "Your topic here"
documents = WikipediaLoader(query=wiki_topic, load_max_docs=2, load_all_available_meta=True).load()

# Query the Wikipedia content
query = 'Your query here'
result = chain.invoke({"input_documents": documents, "question": query})
print(result["output_text"])
```

### 3. External APIs Integration

#### Numbers API Example

Interacting with the Numbers API to fetch trivia:

```python
from langchain.chains import APIChain

# Define API specification
spec = """API spec here"""

# Initialize the model and API chain
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=OPENAI_MODEL, temperature=0.0)
chain = APIChain.from_llm_and_api_docs(llm, api_docs=spec)

# Query the API
query = {"question": "Your query here"}
result = chain.invoke(query)
print(result["output"])
```

#### New York Times API Example

Interacting with the New York Times API:

```python
import json
import requests

# Load API spec
api_key = os.getenv("NYT_API_KEY")
spec = requests.get("NYT API spec URL").json()
spec["api_key"] = api_key

# Initialize model and API chain
chain = APIChain.from_llm_and_api_docs(llm, api_docs=json.dumps(spec))

# Query the API
query = {"question": "Your query here"}
result = chain.invoke(query)
print(result["output"])
```

## Suggested Improvements

- **Enhanced User Interaction:** Consider adding a user-friendly interface using Gradio or Streamlit to allow non-technical users to interact with the scripts more easily.
- **Better Error Handling:** Improve error handling to provide more informative messages when queries fail or exceed API limits.
- **Expand Functionality:** Add additional features like pagination handling for large datasets or more advanced filtering options.
- **Documentation:** Extend the documentation to include more detailed explanations of each script, along with usage examples and potential use cases.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
