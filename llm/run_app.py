#!/usr/bin/env python3
"""
Startup script for the Streamlit Test Case Generator
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if required environment variables are set."""
    required_vars = ['HF_TOKEN1', 'opcode']
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True

def check_dependencies():
    """Check if required Python packages are installed."""
    try:
        import streamlit
        import pandas
        import lancedb
        import sentence_transformers
        import requests
        import dotenv
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def main():
    """Main startup function."""
    print("🚀 Starting AI Test Case Generator (Streamlit)...")
    
    # Check if we're in the right directory
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check environment variables
    if not check_environment():
        print("\n💡 Create a .env file with the following variables:")
        print("HF_TOKEN1=your_huggingface_token")
        print("opcode=your_openrouter_key")
        sys.exit(1)
    
    # Start the Streamlit app
    print("✅ Starting Streamlit server...")
    print("🌐 The app will open in your browser at http://localhost:8501")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ App failed to start: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

