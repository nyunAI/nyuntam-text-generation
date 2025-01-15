import os
import subprocess
import yaml
from huggingface_hub import snapshot_download

class LlamaModelConverter:
    def __init__(self, config_path: str):
        # Load YAML file
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_name = self.config['MODEL']
        self.output_path = self.config['OUTPUT_PATH']
        self.cache_path = self.config['CACHE_PATH']
        self.logging_path = self.config['LOGGING_PATH']
        self.cuda_ids = self.config.get('CUDA_ID', '0')
        self.script_path = self.config.get('SCRIPT_PATH')
        self.quantization = self.config.get('QUANTIZATION', 'f16')
        self.user_folder = self.config.get("USER_FOLDER")

        os.makedirs(self.user_folder, exist_ok = True)
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(self.logging_path, exist_ok=True)

    def download_model(self):
        """Download the Hugging Face model to the cache directory."""
        print(f"Downloading model '{self.model_name}' to cache...")
        local_model_dir = snapshot_download(
            repo_id=self.model_name,
            cache_dir=self.cache_path,
            allow_patterns=["*"]
        )
        print(f"Model downloaded to {local_model_dir}")
        return local_model_dir

    def run_script(self, script_path: str, args: list):
        """Run a given script with subprocess."""
        command = ["python", script_path] + args
        print(f"Running script: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error running script: {result.stderr}")
            raise RuntimeError(f"Script {script_path} failed with error: {result.stderr}")
        print(f"Script output: {result.stdout}")

    def convert_to_gguf(self, model_dir: str):
        """Convert the model to GGUF format."""
        print("Converting model to GGUF format...")
        convert_script = f"{self.script_path}/convert_hf_to_gguf.py"
        self.run_script(
            convert_script,
            ["--repo", model_dir, "--outfile", self.output_path, "--outtype", self.quantization]
        )
        
        print("Conversion to GGUF format completed!")

    def process(self):
        """Main method to download model, convert to GGUF, and save it."""
        try:
            # Download the Hugging Face model
            model_dir = self.download_model()
            
            # Convert the model to GGUF format
            self.convert_to_gguf(model_dir)
            
            print(f"Model successfully converted and saved to: {self.output_path}")
        except Exception as e:
            print(f"Error during processing: {str(e)}")
            raise
