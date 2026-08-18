import sys
import os
import streamlit.web.cli as stcli

def resolve_path(path):
    return os.path.abspath(os.path.join(os.getcwd(), path))

if __name__ == "__main__":
    # Force Streamlit to run via standard system arguments
    sys.argv = [
        "streamlit",
        "run",
        resolve_path("app.py"), 
        "--global.developmentMode=false"
    ]
    sys.exit(stcli.main())