
import sys
import os
import torch
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

def check_setup():
    print("🔍 BDH Environment Verification")
    print("===============================")
    
    # 1. Python Version
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info < (3, 10):
        print("❌ Warning: Python 3.10+ recommended.")
    else:
        print("✅ Python Version OK")

    # 2. PyTorch & CUDA
    print(f"\nPyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  CUDA Not Available (Running on CPU - will be slow)")

    # 3. Model Checks
    print("\nChecking Models...")
    try:
        import spacy
        try:
            spacy.load("en_core_web_trf")
            print("✅ SpaCy Transformer (en_core_web_trf) found!")
        except OSError:
            print("⚠️  SpaCy Transformer (en_core_web_trf) NOT found.")
            print("   Run: python -m spacy download en_core_web_trf")
            try:
                spacy.load("en_core_web_lg")
                print("   (Found en_core_web_lg fallback)")
            except:
                print("   (No large model found)")
    except ImportError:
        print("❌ SpaCy not installed.")

    # 4. Ollama Check
    print("\nChecking Ollama (Deep Reasoning)...")
    try:
        import requests
        try:
            r = requests.get("http://localhost:11434")
            if r.status_code == 200:
                print("✅ Ollama Service is RUNNING")
                # Check for models
                try:
                    r_tags = requests.get("http://localhost:11434/api/tags")
                    models = [m['name'] for m in r_tags.json().get('models', [])]
                    print(f"   Available Models: {', '.join(models)}")
                    if any('qwen' in m or 'mistral' in m for m in models):
                         print("   (Recommended models found)")
                    else:
                         print("   ⚠️  Warning: No 'qwen' or 'mistral' found. Run: ollama pull qwen2.5:7b")
                except:
                    print("   (Could not list models)")
            else:
                print(f"⚠️  Ollama Service returned {r.status_code}")
        except requests.exceptions.ConnectionError:
            print("⚠️  Ollama Service NOT running (Deep Reasoning disabled)")
            print("   Start it with 'ollama serve' if desired.")
    except ImportError:
        print("⚠️  requests library missing (install requirements.txt)")
    
    # 5. Pipeline Smoke Test
    print("\nRunning Pipeline Smoke Test...")
    try:
        sys.path.append(os.getcwd())
        from src.pipeline import create_sota_pipeline
        
        # Initialize with CPU to be safe/fast for check
        pipeline = create_sota_pipeline(device="cpu") 
        print("✅ Pipeline Initialized")
        
        # Tiny Ingestion
        print("   Ingesting mini-text...")
        pipeline.ingest_novel("Alice was a girl who lived in London. She had a cat.")
        
        # Tiny Check
        print("   Checking consistency...")
        verdict = pipeline.check_consistency("Alice lived in London.")
        if verdict.consistent:
            print("✅ Logic Check Passed (Consistent)")
        else:
            print("❌ Logic Check Failed (Unexpected Contradiction)")
            
    except Exception as e:
        print(f"❌ Pipeline Test Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nDone.")

if __name__ == "__main__":
    check_setup()
