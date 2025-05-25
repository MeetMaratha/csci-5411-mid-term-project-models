import json
import subprocess
import time
from threading import Thread

import requests


class OllamaManager:
    def __init__(self, model_name="llama2", host="localhost", port=11434):
        self.base_url = f"http://{host}:{port}"
        self.model_name = model_name
        print(f"Model Name: {model_name}")
        self.service_process = None

    def _start_ollama_service(self):
        """Start Ollama service in background"""
        try:
            self.service_process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            Thread(target=self._monitor_service_output, daemon=True).start()
        except FileNotFoundError:
            raise RuntimeError(
                "Ollama not installed. Download from https://ollama.com/"
            )

    def _monitor_service_output(self):
        """Capture and log service output"""
        while self.service_process.poll() is None:
            line = self.service_process.stdout.readline()
            if line:
                print(f"[Ollama Service] {line.strip()}")

    def _service_health_check(self, retries=5, delay=2):
        """Check if service is responsive"""
        for _ in range(retries):
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=3)
                if response.status_code == 200:
                    return True
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(delay)
        return False

    def _model_exists(self):
        """Check if required model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            models = [m["name"] for m in response.json().get("models", [])]
            return any(self.model_name in m for m in models)
        except requests.RequestException:
            return False

    def _download_model(self):
        """Pull required model with progress tracking"""
        print(f"Downloading {self.model_name}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/pull", json={"name": self.model_name}, stream=True
            )

            for line in response.iter_lines():
                if line:
                    progress = json.loads(line)
                    if "completed" in progress and "total" in progress:
                        print(
                            f"Progress: {progress['completed']/progress['total']*100:.1f}%"
                        )
        except Exception as e:
            raise RuntimeError(f"Model download failed: {str(e)}")

    def ensure_ready(self):
        """Main entry point to ensure service and model are ready"""
        # Check service status
        if not self._service_health_check():
            print("Starting Ollama service...")
            self._start_ollama_service()
            if not self._service_health_check(retries=10, delay=3):
                raise RuntimeError("Failed to start Ollama service")

        # Check model availability
        if not self._model_exists():
            print(f"Model {self.model_name} not found")
            self._download_model()

        print("Ollama service and model are ready!")


# Usage
if __name__ == "__main__":
    manager = OllamaManager(model_name="llama3.1")
    manager.ensure_ready()

    # Now use Ollama normally
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "llama3.1", "prompt": "Hello world"},
    )
    print(response.json())
