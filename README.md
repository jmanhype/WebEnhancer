# WebEnhancer

A web scraping and analysis tool powered by GoLLIE (Guideline-following Large Language Model for Information Extraction) for extracting structured information from web pages.

## Overview

WebEnhancer fetches content from any URL and uses the GoLLIE-7B model to perform relation extraction and entity recognition on the extracted text. This allows for automated analysis and structuring of web content.

## Features

- 🌐 Web scraping with automatic text extraction
- 🤖 AI-powered entity and relation extraction using GoLLIE-7B
- 📊 Structured output with F1 score evaluation
- 🎯 Customizable extraction guidelines

## Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for model inference)
- At least 16GB RAM for model loading

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jmanhype/WebEnhancer.git
cd WebEnhancer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the GoLLIE dependencies (see Configuration section below)

## Configuration

WebEnhancer requires the following modules to be properly configured:

- `src/tasks/utils_typing.py` - Contains `AnnotationList` class
- `guidelines.py` - Defines entity and relation extraction guidelines
- `tasks/utils_scorer.py` - Contains `RelationScorer` class

These modules should define:
- `ENTITY_DEFINITIONS`: List of entity types for extraction
- `Relation`: Base class for relation annotations
- Custom guidelines for your specific use case

## Usage

Run the script and provide a URL when prompted:

```bash
python WebEnhancer.py
```

Example interaction:
```
Please enter the URL you want to analyze: https://example.com
```

The script will:
1. Fetch the web page content
2. Extract text from the HTML
3. Run GoLLIE model inference
4. Output extracted relations and entities
5. Display evaluation metrics

## How It Works

1. **Web Scraping**: Uses `requests` and `BeautifulSoup` to fetch and parse HTML
2. **Text Extraction**: Extracts clean text content from the web page
3. **Template Rendering**: Uses Jinja2 to format the input according to GoLLIE's expected format
4. **Model Inference**: Processes the text through the GoLLIE-7B model
5. **Result Parsing**: Parses model output into structured annotations
6. **Evaluation**: Computes precision, recall, and F1 scores

## Model Information

This project uses [GoLLIE-7B](https://huggingface.co/HiTZ/GoLLIE-7B), a large language model fine-tuned for information extraction tasks following specific guidelines.

## Dependencies

- `transformers`: Hugging Face transformers for model loading
- `torch`: PyTorch for model inference
- `requests`: HTTP library for web scraping
- `beautifulsoup4`: HTML parsing
- `jinja2`: Template rendering
- `black`: Code formatting for consistent prompts

## Limitations

- Requires significant computational resources (7B parameter model)
- First run will download ~14GB model weights
- Processing time depends on text length and hardware

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style patterns
- New features include appropriate documentation
- Changes are backwards compatible

## License

[Add your license information here]

## Acknowledgments

- GoLLIE model by HiTZ Center
- Built with Hugging Face Transformers

## Troubleshooting

**Issue**: Model fails to load
- Ensure you have sufficient RAM (16GB+)
- Check CUDA installation if using GPU

**Issue**: Import errors
- Verify all required modules are present in the correct paths
- Check that `guidelines.py` and task utilities are configured

**Issue**: Web scraping fails
- Verify URL is accessible
- Some websites may block automated requests
- Check your internet connection
