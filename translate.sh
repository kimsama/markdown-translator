#!/bin/bash

echo "Markdown Korean Translator"
echo "-----------------------"

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "NOTE: ANTHROPIC_API_KEY environment variable is not set"
    echo "The application will try to load it from the .env file"
    echo "If not found there, you will need to provide it using the -k parameter"
fi

python translator.py "$@"

echo ""
echo "Done!"
