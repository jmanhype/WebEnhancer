import sys
sys.path.append("../")

import logging
import black
import inspect
from jinja2 import Template as jinja2Template
import tempfile
from typing import Dict, List, Type
from transformers import AutoModelForCausalLM, AutoTokenizer  # Import from transformers
from src.tasks.utils_typing import AnnotationList
from guidelines import *  # Import your defined guidelines
from tasks.utils_scorer import RelationScorer  # Updated import path
import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load GoLLIE model and tokenizer from Hugging Face Model Hub
model_name = "HiTZ/GoLLIE-7B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to fetch a web page and extract text
def extract_text_from_web_page(url):
    try:
        # Fetch the web page content
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract the text content from the web page
            text_content = soup.get_text()
            return text_content
        else:
            return "Failed to fetch the web page. Please check the URL."
    except Exception as e:
        return str(e)

# Ask the user for the URL input
user_url = input("Please enter the URL you want to analyze: ")

# Fetch the web page and extract text
web_page_text = extract_text_from_web_page(user_url)

# Define input sentence and gold labels for GoLLIE
text = web_page_text  # Use the extracted text as input
gold = []  # No gold annotations are needed for this task

# Define the template for Relation Extraction
template_txt = """
{%- for definition in guidelines %}
{{ definition }}
{%- endfor %}

# This is the text to analyze
text = {{ text.__repr__() }}

# The annotation instances that take place in the text above are listed here
result = [
{%- for ann in annotations %}
    {{ ann }},
{%- endfor %}
]
"""

template = jinja2Template(template_txt)

# Fill the template
formatted_text = template.render(guidelines=guidelines, text=text, annotations=gold, gold=gold)

# Use Black Code Formatter to unify prompts
black_mode = black.Mode()
formatted_text = black.format_str(formatted_text, mode=black_mode)

# Tokenize the input sentence
model_input = tokenizer(formatted_text, add_special_tokens=True, return_tensors="pt")

# Remove the eos token from the input
model_input["input_ids"] = model_input["input_ids"][:, :-1]
model_input["attention_mask"] = model_input["attention_mask"][:, :-1]

# Run GoLLIE to generate predictions
model_output = model.generate(
    **model_input.to(model.device),
    max_new_tokens=128,
    do_sample=False,
    min_new_tokens=0,
    num_beams=1,
    num_return_sequences=1,
)

# Parse the output using the AnnotationList class
result = AnnotationList.from_output(
    tokenizer.decode(model_output[0], skip_special_tokens=True).split("result = ")[-1],
    task_module="guidelines"
)

# Instantiate the scorer
class MyScorer(RelationScorer):
    valid_types: List[Type] = ENTITY_DEFINITIONS

    def __call__(self, reference: List[Relation], predictions: List[Relation]) -> Dict[str, Dict[str, float]]:
        output = super().__call__(reference, predictions)
        return {"relations": output}

scorer = MyScorer()

# Compute F1 score
scorer_results = scorer(reference=[gold], predictions=[result])
print(scorer_results)

