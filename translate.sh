#!/bin/bash

echo "Markdown Korean Translator"
echo "-----------------------"

# Check if virtual environment exists and activate it
if [ -f "venv/bin/activate" ]; then
    echo "Using virtual environment..."
    source venv/bin/activate
fi

# Check if API key is set
if [ -z "$OPENAI_API_KEY" ]; then
    echo "NOTE: OPENAI_API_KEY environment variable is not set"
    echo "The application will try to load it from the .env file"
    echo "If not found there, you will need to provide it using the -k parameter"
fi

python translator.py "$@"

echo ""
echo "Done!"
