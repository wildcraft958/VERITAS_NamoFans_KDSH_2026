"""
Pathway API server for the Narrative Auditor.

Based on: https://github.com/pathwaycom/pathway

Provides:
- CSV batch processing
- REST API endpoint for predictions
- Streaming data support
"""

import json
from pathlib import Path

from kdsh.config import settings
from kdsh.workflow import run_pipeline


def process_batch(
    backstories_path: str,
    novels_dir: str,
    output_path: str = "predictions.csv"
) -> None:
    """
    Process a batch of backstories against novels.
    
    Args:
        backstories_path: Path to CSV with backstories (id, book, backstory columns)
        novels_dir: Directory containing novel text files
        output_path: Path to write predictions CSV
    """
    import pandas as pd
    from tqdm import tqdm
    
    # Load backstories
    df = pd.read_csv(backstories_path)
    
    # Load novels
    novels = {}
    novels_path = Path(novels_dir)
    for novel_file in novels_path.glob("*.txt"):
        novel_name = novel_file.stem
        with open(novel_file) as f:
            novels[novel_name] = f.read()
    
    # Process each backstory
    results = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        backstory_id = str(row.get("id", idx))
        book_name = row.get("book", "")
        backstory = row.get("backstory", "")
        
        # Get novel text
        novel_text = novels.get(book_name, "")
        if not novel_text:
            # Try partial match
            for name, text in novels.items():
                if book_name.lower() in name.lower() or name.lower() in book_name.lower():
                    novel_text = text
                    break
        
        if not novel_text:
            results.append({
                "id": backstory_id,
                "prediction": 0,
                "confidence": 0.0,
                "error": f"Novel not found: {book_name}"
            })
            continue
        
        # Run pipeline
        try:
            result = run_pipeline(
                backstory=backstory,
                novel_text=novel_text,
                backstory_id=backstory_id,
                novel_id=book_name
            )
            
            results.append({
                "id": backstory_id,
                "prediction": result["prediction"],
                "confidence": result["confidence"],
                "reasoning": "; ".join(result.get("reasoning", [])[:2]),
                "error": result.get("error")
            })
        except Exception as e:
            results.append({
                "id": backstory_id,
                "prediction": 0,
                "confidence": 0.0,
                "error": str(e)
            })
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def run_server():
    """
    Run the Pathway REST API server.
    
    Provides /predict endpoint for single predictions.
    """
    try:
        import pathway as pw
    except ImportError:
        print("Pathway not installed. Running simple HTTP server instead.")
        _run_simple_server()
        return
    
    # Define schema for predictions table
    class PredictionSchema(pw.Schema):
        id: str
        prediction: int
        confidence: float
        reasoning: str
    
    # Create UDF for predictions
    @pw.udf
    def verify_narrative(backstory: str, novel_text: str, backstory_id: str) -> dict:
        result = run_pipeline(
            backstory=backstory,
            novel_text=novel_text,
            backstory_id=backstory_id
        )
        return {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "reasoning": json.dumps(result.get("reasoning", []))
        }
    
    # Set up REST connector
    host = settings.pathway.host
    port = settings.pathway.port
    
    print(f"Starting Pathway server at http://{host}:{port}")
    print("Endpoint: POST /predict")
    
    # Note: Full Pathway setup requires input source configuration
    # This is a simplified version - see Pathway docs for full setup
    
    pw.run()


def _run_simple_server():
    """
    Run a simple HTTP server without Pathway.
    
    Fallback when Pathway is not installed.
    """
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json
    
    class PredictHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/predict":
                content_length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(content_length)
                
                try:
                    data = json.loads(body)
                    backstory = data.get("backstory", "")
                    novel_text = data.get("novel_text", "")
                    backstory_id = data.get("id", "default")
                    
                    result = run_pipeline(
                        backstory=backstory,
                        novel_text=novel_text,
                        backstory_id=backstory_id
                    )
                    
                    response = {
                        "prediction": result["prediction"],
                        "confidence": result["confidence"],
                        "reasoning": result.get("reasoning", []),
                        "error": result.get("error")
                    }
                    
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
            else:
                self.send_response(404)
                self.end_headers()
        
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "healthy"}')
            else:
                self.send_response(404)
                self.end_headers()
    
    host = settings.pathway.host
    port = settings.pathway.port
    
    server = HTTPServer((host, port), PredictHandler)
    print(f"Starting simple HTTP server at http://{host}:{port}")
    print("Endpoints:")
    print("  - POST /predict: Verify narrative consistency")
    print("  - GET /health: Health check")
    
    server.serve_forever()


if __name__ == "__main__":
    run_server()
