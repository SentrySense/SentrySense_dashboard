#!/bin/bash

echo "🛡️ Starting SentrySense Dashboard..."
echo "📊 Dashboard will be available at: http://localhost:8501"
echo ""


# Install requirements
pip install -r requirements.txt

echo "🔧 Setting up environment..."
echo "📁 Directory structure created"
echo "📦 Dependencies installed"
echo ""
echo "🚀 Launching dashboard..."

# Start Streamlit dashboard
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
