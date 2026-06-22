#!/usr/bin/env python
"""
Titanic Survival Prediction - Desktop Application
Single file to run the entire project!
"""

import socket
import subprocess
import sys
import os
import time
import webbrowser
from pathlib import Path

def check_and_train():
    """Check if models exist, train if needed"""
    models_dir = Path("models")
    if not models_dir.exists() or not list(models_dir.glob("*.pkl")):
        print("📊 Models not found. Training first...")
        subprocess.run([sys.executable, "run_pipeline.py"])
        print("✅ Training complete!\n")

def wait_for_port(host, port, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def launch_app():
    """Launch the Streamlit app and open it in the browser."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     🚢 TITANIC SURVIVAL PREDICTION - DESKTOP APP 🚢      ║
    ║                                                          ║
    ║     The app will open in your browser automatically      ║
    ║                                                          ║
    ║     Close the terminal window to stop the app            ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    server_url = "http://localhost:8501"
    streamlit_command = [
        sys.executable, "-m", "streamlit", "run",
        "gui/app.py",
        "--server.headless=false",
        "--server.port=8501"
    ]

    process = subprocess.Popen(
        streamlit_command,
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    try:
        print("Starting Streamlit...")
        if wait_for_port("localhost", 8501, timeout=30):
            print(f"Opening browser at {server_url}")
            webbrowser.open(server_url, new=2, autoraise=True)
        else:
            if process.poll() is not None:
                print(f"Streamlit exited with code {process.returncode} before the app became available.")
            else:
                print("Streamlit did not become available on localhost:8501 within 30 seconds.")

        for line in iter(process.stdout.readline, ""):
            print(line, end="")

        process.wait()
    except KeyboardInterrupt:
        print("Stopping the app...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    # Make sure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Check and train if needed
    check_and_train()
    
    # Launch the app
    launch_app()