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
def extract_text_from_web_page(url: str) -> str:
    """Fetch and extract text content from a web page.

    Args:
        url: The URL of the web page to fetch

    Returns:
        Extracted text content from the web page

    Raises:
        requests.RequestException: If the request fails
        ValueError: If the URL is invalid
    """
    if not url or not url.strip():
        raise ValueError("URL cannot be empty")

    if not url.startswith(('http://', 'https://')):
        raise ValueError("URL must start with http:// or https://")

    try:
        # Fetch the web page content with timeout
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raise exception for bad status codes

        # Parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Extract the text content from the web page
        text_content = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)

        if not text_content:
            raise ValueError("No text content could be extracted from the web page")

        return text_content

    except requests.Timeout:
        raise requests.RequestException("Request timed out. The server took too long to respond.")
    except requests.ConnectionError:
        raise requests.RequestException("Connection error. Please check your internet connection.")
    except requests.HTTPError as e:
        raise requests.RequestException(f"HTTP error occurred: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {str(e)}")

def main() -> None:
    """Main execution function with error handling."""
    try:
        # Ask the user for the URL input
        user_url = input("Please enter the URL you want to analyze: ")

        # Fetch the web page and extract text
        logging.info(f"Fetching content from: {user_url}")
        web_page_text = extract_text_from_web_page(user_url)
        logging.info(f"Successfully extracted {len(web_page_text)} characters of text")

        # Define input sentence and gold labels for GoLLIE
        text = web_page_text  # Use the extracted text as input
        gold: List = []  # No gold annotations are needed for this task

        process_text_with_gollie(text, gold)

    except ValueError as e:
        logging.error(f"Validation error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    except requests.RequestException as e:
        logging.error(f"Network error: {e}")
        print(f"Network Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        print("\nProcess interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


def process_text_with_gollie(text: str, gold: List) -> None:
    """Process text using GoLLIE model for information extraction.

    Args:
        text: Input text to analyze
        gold: Gold standard annotations (can be empty list)
    """
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

    logging.info("Running model inference...")

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

    logging.info(f"Extracted {len(result)} annotations")

    # Instantiate the scorer
    class MyScorer(RelationScorer):
        valid_types: List[Type] = ENTITY_DEFINITIONS

        def __call__(self, reference: List, predictions: List) -> Dict[str, Dict[str, float]]:
            output = super().__call__(reference, predictions)
            return {"relations": output}

    scorer = MyScorer()

    # Compute F1 score
    scorer_results = scorer(reference=[gold], predictions=[result])

    print("\n=== Extraction Results ===")
    print(f"Annotations found: {len(result)}")
    print(f"\nScores: {scorer_results}")
    print("\n=== Extracted Annotations ===")
    for i, annotation in enumerate(result, 1):
        print(f"{i}. {annotation}")


if __name__ == "__main__":
    main()

