#!/bin/bash

echo "🚀 Setting up Natural Conversation Example"
echo "=========================================="

# Check if Python 3.10+ is installed (macOS compatible)
python_version=$(python3 --version 2>&1 | sed 's/Python //' | cut -d. -f1,2)
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.10 or higher is required. You have Python $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Error installing dependencies"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.example .env
    echo "✅ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please edit .env file and add your API keys:"
    echo "   - GOOGLE_API_KEY: Get from https://makersuite.google.com/app/apikey"
    echo "   - CARTESIA_API_KEY: Get from https://cartesia.ai/"
    echo ""
else
    echo "✅ .env file already exists"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To run the example:"
echo "  python 22d-natural-conversation-gemini-audio.py --transport webrtc"
echo ""
echo "For more options, see README-natural-conversation.md" 