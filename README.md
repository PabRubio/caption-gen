## Project Overview

Caption-gen is an image caption generation web application using Hugging Face transformers. It provides a Flask web interface for uploading images and generating descriptive captions using the Salesforce BLIP model.

## Commands

### Setup and Installation
```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install flask pillow torch gunicorn
```

### Running the Application
```bash
# Development mode (Flask development server on port 5000)
python app.py

# Standalone caption generation (processes selfie.jpg)
python main.py

# Production mode (SystemD service)
sudo systemctl start caption-gen
sudo systemctl status caption-gen
sudo systemctl stop caption-gen
```

### Development Commands
```bash
# Run specific model tests
python main2.py  # DeepSeek model test
python main3.py  # Llama model test
```

## Architecture

### Core Components

1. **Web Application (`app.py`)**: Flask server with two routes:
   - `/` - Serves the HTML interface
   - `/generate_caption` - REST API endpoint accepting base64-encoded images

2. **Frontend (`index.html`)**: Simple JavaScript-based interface for image upload and caption display

3. **Model Pipeline**: Uses `Salesforce/blip-image-captioning-large` with configured generation parameters (beam search, repetition penalty, etc.)

### Production Deployment

The application is configured as a SystemD service (`caption-gen.service`) that:
- Runs Gunicorn with Unix socket at `/home/pabrubio/.caption-gen.pabrubio.hackclub.app.webserver.sock`
- Uses 1 worker with 600-second timeout
- Automatically restarts on failure
- Waits for network connectivity before starting

### Model Variants

- `main.py`: Enhanced caption generation using BLIP + OPT-125M with LangChain
- `main2.py`: DeepSeek R1 Distill model testing
- `main3.py`: Meta Llama 3.2 1B Instruct model testing

## Key Implementation Details

- Images are sent as base64-encoded data in JSON POST requests
- The BLIP model uses specific generation parameters for quality optimization
- Error verbosity is suppressed using `set_verbosity_error()`
- The service runs from `/home/pabrubio/pub/caption-gen` in production