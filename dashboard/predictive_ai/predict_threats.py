import os
import json
import sys

# Add the current directory to Python path to import mock_infra
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    from mock_infra import mock_infrastructure
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install required packages:")
    print("pip install google-generativeai python-dotenv")
    sys.exit(1)

# Get project root
root_dir = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(root_dir, '.env')

# Load environment variables
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    print(f"Warning: .env file not found at {env_path}")
    print("Please ensure you have a .env file with GEMINI_API_KEY")

# Configure Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in environment variables")
    print("Please set GEMINI_API_KEY in your .env file")
    # Create a dummy prediction to avoid crashes
    output_file = os.path.join(current_dir, "predicted_threats.json")
    dummy_predictions = [
        {
            "threat_type": "Configuration Error",
            "predicted_time": "",
            "description": "API key not configured. Please set GEMINI_API_KEY in .env file to enable AI threat analysis.",
            "risk_level": "Medium",
            "affected_systems": ["AI Analysis System"],
            "suggested_fixes": ["Set GEMINI_API_KEY in .env file", "Install required dependencies"],
            "confidence_score": 0.0,
            "confidence_reasoning": "API not configured",
            "file": "config_error.txt"
        }
    ]
    with open(output_file, 'w') as f:
        json.dump(dummy_predictions, f, indent=4)
    print(f"Created dummy prediction file at {output_file}")
    sys.exit(0)

genai.configure(api_key=api_key)

threats_folder = os.path.join(current_dir, "threats")
output_file = os.path.join(current_dir, "predicted_threats.json")

model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Load infrastructure for prompt
infra_text = json.dumps(mock_infrastructure, indent=2)

# Load existing predictions and avoid duplicates
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        predictions = json.load(f)
    processed_files = {item["file"] for item in predictions}
else:
    predictions = []
    processed_files = set()

# Check if threats folder exists
if not os.path.exists(threats_folder):
    print(f"Threats folder not found at {threats_folder}")
    print("Please run fetch_threats.py first to download CVE data")
    sys.exit(1)

threat_files = [f for f in os.listdir(threats_folder) if f.endswith(".txt")]
if not threat_files:
    print("No threat files found in threats folder")
    print("Please run fetch_threats.py first to download CVE data")
    sys.exit(1)

prompt_template = """You are a cybersecurity AI assistant for an organization. Analyze the following CVE threat. Your response must:
- Give a clear, modern, technical description of the threat.
- Identify realistic affected systems based on this infrastructure:
{infra}
- Suggest specific, actionable fixes for the organization.
- Assign a risk level (Medium, High, Critical).
- Provide a realistic confidence score (0-1) with reasoning based on severity, exploitability, and relevance to the provided infrastructure.

DO NOT include references from the report.

CVE Report:
\"\"\" {threat_report}\"\"\"

Provide your response strictly as valid JSON with the following fields:
- threat_type: Short threat title.
- predicted_time: Leave blank, we will auto-fill.
- description: Detailed, practical threat explanation.
- risk_level: One of "Medium", "High", or "Critical".
- affected_systems: List of realistically affected systems (hostnames or devices).
- suggested_fixes: Specific, actionable fixes."""

# Track if any new predictions were added
new_added = False

for filename in threat_files:
    if filename in processed_files:
        continue
    
    file_path = os.path.join(threats_folder, filename)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            threat_report = f.read()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='latin-1') as f:
            threat_report = f.read()
    
    prompt = prompt_template.format(threat_report=threat_report, infra=infra_text)
    
    try:
        response = model.generate_content(prompt)
        ai_output = response.text.strip()
        print(f"\n----- RAW AI OUTPUT for {filename} -----\n{ai_output}\n--------------------------------------\n")
        
        # Remove wrapping triple backticks if present
        if ai_output.startswith("\`\`\`json"):
            ai_output = ai_output[len("\`\`\`json"):].strip()
        if ai_output.endswith("\`\`\`"):
            ai_output = ai_output[:-3].strip()
        
        if not ai_output:
            print(f"Empty AI output for {filename}, skipping...\n")
            continue
        
        prediction = json.loads(ai_output)
        prediction["file"] = filename
        prediction["confidence_score"] = prediction.get("confidence_score", 0.8)
        prediction["confidence_reasoning"] = prediction.get("confidence_reasoning", "AI analysis based on CVE data")
        predictions.append(prediction)
        new_added = True
        print(f"Processed {filename}\n")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}\n")

# Only save if new predictions were added
if new_added:
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    print(f"\nPredictions saved to {output_file}")
else:
    print("\nNo new CVEs. File unchanged.")
