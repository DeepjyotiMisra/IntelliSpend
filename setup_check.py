#!/usr/bin/env python3
"""IntelliSpend Setup Verification Script"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Required packages with import names
PACKAGES = {
    'agno': 'agno',
    'openai': 'openai', 
    'python-dotenv': 'dotenv',
    'duckduckgo-search': 'duckduckgo_search',
    'pypdf': 'pypdf',
    'lancedb': 'lancedb',
    'streamlit': 'streamlit',
    'pandas': 'pandas',
    'tantivy': 'tantivy',
    'yfinance': 'yfinance'
}

REQUIRED_ENV_VARS = [
    'OPENAI_API_KEY', 'MODEL_API_VERSION', 
    'MODEL_NAME', 'AZURE_OPENAI_ENDPOINT'
]

def check_python():
    """Check Python version"""
    version = sys.version_info
    is_ok = version.major == 3 and version.minor >= 12
    status = "✅" if is_ok else "❌"
    print(f"{status} Python {version.major}.{version.minor}.{version.micro}")
    return is_ok

def check_packages():
    """Check required packages"""
    missing = []
    for pip_name, import_name in PACKAGES.items():
        try:
            __import__(import_name)
            print(f"✅ {pip_name}")
        except ImportError:
            print(f"❌ {pip_name}")
            missing.append(pip_name)
    return not missing, missing

def check_env():
    """Check environment configuration"""
    if not Path('.env').exists():
        print("❌ .env file not found")
        return False, ['Copy .env.example to .env']
    
    load_dotenv()
    missing = []
    
    for var in REQUIRED_ENV_VARS:
        value = os.getenv(var)
        if not value or 'your_' in value or 'your-' in value:
            print(f"❌ {var}")
            missing.append(var)
        else:
            print(f"✅ {var}")
    
    return not missing, missing

def main():
    """Run setup verification"""
    print("🚀 IntelliSpend Setup Verification\n")
    
    print("📍 Python version:")
    python_ok = check_python()
    
    print("\n📍 Dependencies:")
    deps_ok, missing_deps = check_packages()
    
    print("\n📍 Environment:")
    env_ok, missing_env = check_env()
    
    print(f"\n{'🎉' if all([python_ok, deps_ok, env_ok]) else '⚠️'} Summary:")
    
    if python_ok and deps_ok and env_ok:
        print("All checks passed! Run: cd InitialAgents && python basicAgents.py")
        return 0
    
    if not python_ok:
        print("- Upgrade to Python 3.12+")
    if missing_deps:
        print(f"- pip install {' '.join(missing_deps)}")
    if missing_env:
        print("- Configure .env file with your values")
    
    return 1

if __name__ == "__main__":
    sys.exit(main())