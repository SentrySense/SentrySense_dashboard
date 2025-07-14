import streamlit as st
import json
import time
import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="SentrySense Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced dark theme and animations
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #ffffff !important;
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 0 4px 8px rgba(0,0,0,0.5);
    }
    
    @keyframes glow {
        from { text-shadow: 0 0 20px #00ff88, 0 0 30px #00ff88; }
        to { text-shadow: 0 0 30px #00ff88, 0 0 40px #00ff88; }
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #00ff88;
        margin: 0.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 255, 136, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 255, 136, 0.2);
    }
    
    .anomaly-card {
        background: linear-gradient(135deg, #2d1b1b 0%, #3d2b2b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
        box-shadow: 0 6px 20px rgba(255, 68, 68, 0.2);
        transition: all 0.3s ease;
    }
    
    .anomaly-card:hover {
        transform: translateX(5px);
        box-shadow: 0 8px 25px rgba(255, 68, 68, 0.3);
    }
    
    .threat-card {
        background: linear-gradient(135deg, #1b1b2d 0%, #2b2b3d 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #444;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
        transition: all 0.3s ease;
    }
    
    .threat-card:hover {
        transform: scale(1.02);
        box-shadow: 0 12px 35px rgba(0, 0, 0, 0.6);
    }
    
    .badge-high {
        background: linear-gradient(45deg, #ff4444, #ff6666);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 68, 68, 0.3);
    }
    
    .badge-medium {
        background: linear-gradient(45deg, #ff8800, #ffaa00);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 136, 0, 0.3);
    }
    
    .badge-low {
        background: linear-gradient(45deg, #00aa44, #00cc55);
        color: white;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(0, 170, 68, 0.3);
    }
    
    .alert-banner {
        background: linear-gradient(90deg, #ff4444, #ff6666, #ff4444);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        animation: pulse-alert 2s infinite, slide-in 0.5s ease-out;
        box-shadow: 0 8px 25px rgba(255, 68, 68, 0.4);
    }
    
    @keyframes pulse-alert {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.8; transform: scale(1.02); }
    }
    
    @keyframes slide-in {
        from { transform: translateY(-100px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    .status-active {
        background-color: #00ff88;
        box-shadow: 0 0 15px #00ff88;
    }
    
    .status-warning {
        background-color: #ff8800;
        box-shadow: 0 0 15px #ff8800;
    }
    
    .status-critical {
        background-color: #ff4444;
        box-shadow: 0 0 15px #ff4444;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 100%);
    }
    
    .feature-card {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #00ff88;
        margin: 0.5rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        background: linear-gradient(135deg, #2d2d2d 0%, #3d3d3d 100%);
        transform: translateY(-2px);
    }
    
    .explainability-card {
        background: linear-gradient(135deg, #2d2d1b 0%, #3d3d2b 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffaa00;
        margin: 0.5rem 0;
    }

    .element-container:empty {
        display: none !important;
    }

    .stPlotlyChart:empty {
        display: none !important;
    }

    div[data-testid="column"]:empty {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'last_anomaly_count' not in st.session_state:
    st.session_state.last_anomaly_count = 0
if 'last_threat_count' not in st.session_state:
    st.session_state.last_threat_count = 0
if 'show_alert' not in st.session_state:
    st.session_state.show_alert = False
if 'alert_message' not in st.session_state:
    st.session_state.alert_message = ""
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

def get_badge_html(level):
    """Generate HTML for severity badges"""
    if isinstance(level, bool):
        level = "high" if level else "normal"
    
    level_lower = str(level).lower()
    if level_lower in ['high', 'critical', 'true']:
        return f'<span class="badge-high">HIGH</span>'
    elif level_lower in ['medium', 'moderate']:
        return f'<span class="badge-medium">MEDIUM</span>'
    elif level_lower in ['low', 'low_level']:
        return f'<span class="badge-low">LOW</span>'
    else:
        return f'<span class="badge-low">NORMAL</span>'

def load_anomaly_data():
    """Execute stream_inference.py and load anomaly data from stream_logs.jsonl"""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Step 1: Run stream_inference.py to update the log
        stream_script = Path("simulation_and_detection_\\src\\stream_inference.py")
        log_file = Path("simulation_and_detection_\\logs\\stream_logs.jsonl")
        
        if not stream_script.exists():
            st.error(f"Inference script not found: {stream_script}")
            # Try to read existing file if available
            if log_file.exists():
                st.warning("Using cached anomaly data.")
            else:
                return []

        try:
            with st.spinner("Running anomaly detection..."):
                result = subprocess.run(
                    [sys.executable, str(stream_script)],
                    cwd=Path.cwd(),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    st.error(f"Error executing stream_inference.py:\n{result.stderr}")
                    # Try to read existing file if available
                    if not log_file.exists():
                        return []
                else:
                    if result.stdout:
                        st.success("Anomaly inference completed!")
                        
        except subprocess.TimeoutExpired:
            st.error("Anomaly inference script timed out after 60 seconds")
            # Try to read existing file if available
            if not log_file.exists():
                return []
        except subprocess.SubprocessError as e:
            st.error(f"Failed to run anomaly inference: {str(e)}")
            # Try to read existing file if available
            if not log_file.exists():
                return []

        # Step 2: Read the log file
        if not log_file.exists():
            st.warning("stream_logs.jsonl not found. Please check anomaly pipeline.")
            return []

        anomalies = []
        with open(log_file, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line.strip())
                    if 'timestamp' not in data:
                        data['timestamp'] = datetime.now().isoformat()
                    anomalies.append(data)

        return anomalies[-10:] if len(anomalies) >= 10 else anomalies
    except Exception as e:
        st.error(f"Error loading anomaly data: {e}")
        return []

def load_threat_data():
    """Load threat data by executing fetch_threats.py, then predict_threats.py, and reading predicted_threats.json"""
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        # Path to the scripts and output file
        fetch_script = Path("predictive_ai\\fetch_threats.py")
        predict_script = Path("predictive_ai\\predict_threats.py")
        output_file = Path("predictive_ai\\predicted_threats.json")
        
        # Step 1: Execute fetch_threats.py to get CVE data
        if not fetch_script.exists():
            st.error(f"Threat fetching script not found: {fetch_script}")
            # Try to read existing file if available
            if output_file.exists():
                st.warning("Using cached threat data.")
            else:
                return []
        else:
            try:
                with st.spinner("🔍 Fetching latest CVE threats from NVD..."):
                    result = subprocess.run(
                        [sys.executable, str(fetch_script)],
                        cwd=Path.cwd(),
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout for CVE fetching
                    )
                    
                    if result.returncode != 0:
                        st.error(f"Error executing CVE fetching script: {result.stderr}")
                        # Continue to prediction step anyway
                    else:
                        if result.stdout:
                            st.success("✅ CVE data fetched successfully!")
                            
            except subprocess.TimeoutExpired:
                st.error("CVE fetching script timed out after 2 minutes")
                # Continue to prediction step anyway
            except subprocess.SubprocessError as e:
                st.error(f"Failed to execute CVE fetching script: {str(e)}")
                # Continue to prediction step anyway
        
        # Step 2: Execute predict_threats.py to analyze threats with AI
        if not predict_script.exists():
            st.error(f"Threat prediction script not found: {predict_script}")
            # Try to read existing file if available
            if output_file.exists():
                st.warning("Using existing threat predictions.")
            else:
                return []
        else:
            try:
                with st.spinner("🤖 Analyzing threats with AI intelligence..."):
                    result = subprocess.run(
                        [sys.executable, str(predict_script)],
                        cwd=Path.cwd(),
                        capture_output=True,
                        text=True,
                        timeout=180  # 3 minute timeout for AI analysis
                    )
                    
                    # Only show errors if something actually failed
                    if result.returncode != 0:
                        st.error(f"Threat prediction failed (return code: {result.returncode})")
                        if "GEMINI_API_KEY" not in result.stdout:
                            st.error(f"Error details: {result.stderr}")
                
            except subprocess.TimeoutExpired:
                st.error("Threat prediction script timed out after 3 minutes")
                # Try to read existing file if available
                if not output_file.exists():
                    return []
            except subprocess.SubprocessError as e:
                st.error(f"Failed to execute threat prediction script: {str(e)}")
                # Try to read existing file if available
                if not output_file.exists():
                    return []
            except Exception as e:
                st.error(f"Unexpected error running threat prediction: {str(e)}")
                import traceback
                st.error(f"Traceback: {traceback.format_exc()}")
                # Try to read existing file if available
                if not output_file.exists():
                    return []
        
        # Step 3: Read the predicted_threats.json file
        if not output_file.exists():
            st.warning("AI-predicted threats file not found. Please run the threat analysis pipeline manually.")
            return []
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                threats_data = json.load(f)
            
            # Validate that we have a list
            if not isinstance(threats_data, list):
                st.error("Invalid format in predicted_threats.json - expected a list of threats")
                return []
            
            # Validate and clean AI-predicted threat data
            validated_threats = []
            for i, threat in enumerate(threats_data):
                if not isinstance(threat, dict):
                    st.warning(f"Skipping invalid threat entry at index {i} - not a dictionary")
                    continue
                
                # Map AI prediction fields to dashboard format
                validated_threat = {
                    "cve_id": threat.get("file", f"AI-THREAT-{i}").replace('.txt', ''),
                    "threat_type": threat.get("threat_type", "Unknown AI-Predicted Threat"),
                    "description": threat.get("description", "No description available from AI analysis"),
                    "severity": threat.get("risk_level", "Unknown").upper(),
                    "score": str(threat.get("confidence_score", "Unknown")),
                    "published_date": threat.get("predicted_time") or "AI Predicted",
                    "affected_systems": threat.get("affected_systems", []),
                    "suggested_fixes": threat.get("suggested_fixes", []),
                    "confidence": threat.get("confidence_score", 0.0),
                    "references": [],  # AI predictions don't include references
                    "ai_generated": True,  # Flag to indicate this is AI-generated
                    "source_file": threat.get("file", "Unknown")
                }
                
                validated_threats.append(validated_threat)
            
            st.success(f"🎯 Successfully loaded {len(validated_threats)} AI-predicted threat(s)")
            return validated_threats
            
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse AI predictions file: {str(e)}")
            return []
        except UnicodeDecodeError as e:
            st.error(f"Failed to read AI predictions file due to encoding issue: {str(e)}")
            return []
        except Exception as e:
            st.error(f"Unexpected error reading AI threat predictions: {str(e)}")
            return []
            
    except ImportError as e:
        st.error(f"Missing required module for threat analysis: {str(e)}")
        return []
    except Exception as e:
        st.error(f"Unexpected error in AI threat analysis pipeline: {str(e)}")
        return []

def create_anomaly_timeline(anomalies):
    """Create timeline chart for anomalies"""
    if not anomalies:
        return None
    
    df = pd.DataFrame(anomalies)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Create severity mapping
    severity_map = {'low_level': 1, 'medium': 2, 'high': 3, True: 3, False: 0}
    df['severity_num'] = df['anomaly'].map(lambda x: severity_map.get(x, 1))
    
    fig = px.scatter(df, x='timestamp', y='severity_num', 
                     color='severity_num',
                     color_continuous_scale=['green', 'orange', 'red'],
                     title="Anomaly Detection Timeline",
                     labels={'severity_num': 'Severity Level', 'timestamp': 'Time'})
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#00ff88'
    )
    
    return fig

def create_threat_distribution(threats):
    """Create threat severity distribution chart"""
    if not threats:
        return None
    
    severity_counts = {}
    for threat in threats:
        severity = threat.get('severity', 'Unknown')
        severity_counts[severity] = severity_counts.get(severity, 0) + 1
    
    colors = {'HIGH': '#ff4444', 'MEDIUM': '#ff8800', 'LOW': '#00aa44', 'Unknown': '#666666'}
    
    fig = go.Figure(data=[
        go.Bar(x=list(severity_counts.keys()), 
               y=list(severity_counts.values()),
               marker_color=[colors.get(k, '#666666') for k in severity_counts.keys()])
    ])
    
    fig.update_layout(
        title="Threat Severity Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        title_font_color='#00ff88'
    )
    
    return fig

def display_anomaly_explainability(anomaly):
    """Display GNN model explainability for anomalies"""
    if isinstance(anomaly, dict) and 'why' in anomaly and anomaly['why']:
        st.markdown("#### 🔍 Model Explainability")
        st.markdown("**Top contributing features to anomaly score:**")
        
        for i, feature in enumerate(anomaly['why'][:3], 1):
            if isinstance(feature, dict) and all(key in feature for key in ['feature', 'original', 'reconstructed', 'abs_error']):
                with st.container():
                    st.markdown(f"""
                    <div class="explainability-card">
                        <strong>{i}. {feature['feature']}</strong><br>
                        Original: {feature['original']:.4f} | 
                        Reconstructed: {feature['reconstructed']:.4f} | 
                        <strong>Error: {feature['abs_error']:.4f}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info(f"Feature {i}: Data format not compatible for display")

def check_for_new_data(anomalies, threats):
    """Check for new data and trigger alerts"""
    current_anomaly_count = len(anomalies)
    current_threat_count = len(threats)
    
    if current_anomaly_count > st.session_state.last_anomaly_count:
        st.session_state.show_alert = True
        st.session_state.alert_message = "⚠️ New Anomaly Detected!"
    elif current_threat_count > st.session_state.last_threat_count:
        st.session_state.show_alert = True
        st.session_state.alert_message = "🚨 New Threat Identified!"
    
    st.session_state.last_anomaly_count = current_anomaly_count
    st.session_state.last_threat_count = current_threat_count

def display_latest_entries(anomalies, threats):
    st.markdown("## 🔁 Latest Fetched Entries")
    
    # Real-Time Anomalies
    st.markdown("### 🧠 Real-Time Anomalies")
    if anomalies:
        for anomaly in reversed(anomalies[-3:]):
            st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
            st.markdown(f"""
                <strong>Stream #{anomaly.get("stream_index", "—")}</strong> | 
                <em>{anomaly.get("timestamp", "")[:19].replace('T', ' ')}</em><br>
                <strong>Reason:</strong> {anomaly.get("reason") or f'Score: {anomaly.get("score", "—")}'}<br>
                {get_badge_html(anomaly.get("anomaly"))}
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No anomalies detected in latest inference.")
    
    # AI-Predicted CVE Threats
    st.markdown("### 🎯 AI-Predicted CVE Threats")
    if threats:
        for threat in reversed(threats[-3:]):
            st.markdown('<div class="threat-card">', unsafe_allow_html=True)
            st.markdown(f"""
                <strong>{threat.get("cve_id", "⚠️ Missing CVE ID")}</strong><br>
                <em>{threat.get("published_date") or "Date N/A"}</em><br>
                <strong>Type:</strong> {threat.get("threat_type") or "—"}<br>
                <strong>Score:</strong> {threat.get("score") or "—"}<br>
                {get_badge_html(threat.get("severity"))}
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No recent threats identified.")

def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🛡️ SentrySense Control Panel")
        
        # System Status
        st.markdown("### 📊 System Status")
        
        # Auto-refresh controls
        st.session_state.auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        refresh_interval = st.selectbox("Refresh Interval", [15, 30, 60], index=0)
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Quick Stats
        anomalies = load_anomaly_data()
        threats = load_threat_data()
        
        st.markdown("### 📈 Quick Stats")
        st.metric("Active Anomalies", len(anomalies))
        st.metric("Threat Alerts", len(threats))
        
        high_severity_threats = sum(1 for t in threats if t.get('severity') == 'HIGH')
        st.metric("High Severity", high_severity_threats, delta=high_severity_threats if high_severity_threats > 0 else None)
        
        st.markdown("---")
        
        # System Health
        st.markdown("### 🏥 System Health")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<span class="status-indicator status-active"></span>**GNN Model**', unsafe_allow_html=True)
        with col2:
            st.markdown("🟢 Active")
            
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<span class="status-indicator status-active"></span>**CVE Monitor**', unsafe_allow_html=True)
        with col2:
            st.markdown("🟢 Active")
        
        if high_severity_threats > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<span class="status-indicator status-critical"></span>**Threat Level**', unsafe_allow_html=True)
            with col2:
                st.markdown("🔴 High")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<span class="status-indicator status-active"></span>**Threat Level**', unsafe_allow_html=True)
            with col2:
                st.markdown("🟢 Normal")

    # Main content
    st.markdown('<h1 class="main-header">SentrySense Dashboard</h1>', unsafe_allow_html=True)
    
# Show alert banner if needed
    if st.session_state.show_alert:
        st.markdown(f'<div class="alert-banner">{st.session_state.alert_message}</div>', 
                   unsafe_allow_html=True)
        if st.button("✕ Dismiss Alert"):
            st.session_state.show_alert = False
            #st.rerun()

    # Load data
    anomalies = load_anomaly_data()
    threats = load_threat_data()
    
    # Check for new data and show alerts
    check_for_new_data(anomalies, threats)
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        anomaly_count = len(anomalies)
        high_anomalies = len([a for a in anomalies if a.get('anomaly') in [True, 'high', 'medium']])
        st.metric(
            label="🔍 Total Anomalies", 
            value=anomaly_count, 
            delta=f"+{high_anomalies}" if high_anomalies > 0 else None
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        threat_count = len(threats)
        high_severity_threats = sum(1 for t in threats if t.get('severity') == 'HIGH')
        st.metric(
            label="🎯 CVE Threats", 
            value=threat_count, 
            delta=f"+{high_severity_threats} High" if high_severity_threats > 0 else None
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        last_update = datetime.now().strftime("%H:%M:%S")
        st.metric(
            label="⏰ Last Update", 
            value=last_update, 
            delta="Live"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if high_severity_threats > 2:
            system_status = "Critical"
            status_delta = "🔴"
        elif high_severity_threats > 0:
            system_status = "Warning" 
            status_delta = "🟡"
        else:
            system_status = "Normal"
            status_delta = "🟢"
        
        st.metric(
            label="🛡️ Security Status", 
            value=system_status,
            delta=status_delta
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display latest entries section
    display_latest_entries(anomalies, threats)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔍 Anomaly Detection", "🎯 Threat Intelligence", "📊 Analytics Dashboard"])
    
    with tab1:
        st.markdown("## 🔍 GNN-Based Anomaly Detection")
        
        if not anomalies:
            st.info("No anomalies detected recently. System is running normally.")
        else:
            # Timeline chart
            timeline_fig = create_anomaly_timeline(anomalies)
            if timeline_fig:
                st.plotly_chart(timeline_fig, use_container_width=True)
            
            st.markdown("### Recent Anomaly Events")
            
            for anomaly in reversed(anomalies[-5:]):  # Show last 5
                with st.container():
                    st.markdown('<div class="anomaly-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns([1, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Stream #{anomaly.get('stream_index', 'N/A')}**")
                        timestamp = anomaly.get('timestamp', datetime.now().isoformat())
                        st.markdown(f"*{timestamp[:19].replace('T', ' ')}*")
                    
                    with col2:
                        if 'reason' in anomaly:
                            st.markdown(f"**Reason:** {anomaly['reason']}")
                        elif 'score' in anomaly:
                            st.markdown(f"**Anomaly Score:** {anomaly['score']:,.2f}")
                        
                        # Show raw features if available
                        if 'raw_features' in anomaly:
                            with st.expander("🔧 Raw Features"):
                                st.json(anomaly['raw_features'])
                    
                    with col3:
                        st.markdown(get_badge_html(anomaly.get('anomaly', 'unknown')), unsafe_allow_html=True)
                    
                    # Show explainability if available
                    if 'why' in anomaly:
                        display_anomaly_explainability(anomaly)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
    
    with tab2:
        st.markdown("## 🎯 CVE Threat Intelligence")
        
        if not threats:
            st.info("No high/medium severity threats detected currently.")
        else:
            # Threat distribution chart
            dist_fig = create_threat_distribution(threats)
            if dist_fig:
                st.plotly_chart(dist_fig, use_container_width=True)
            
            st.markdown("### Active Threat Alerts")
            
            for threat in threats:
                with st.container():
                    st.markdown('<div class="threat-card">', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Show if this is AI-generated
                        if threat.get('ai_generated'):
                            st.markdown(f"### 🤖 {threat.get('cve_id', 'Unknown CVE')} (AI Predicted)")
                            st.markdown(f"**Source:** {threat.get('source_file', 'Unknown')}")
                        else:
                            st.markdown(f"### {threat.get('cve_id', 'Unknown CVE')}")
                        
                        if threat.get('threat_type') and threat['threat_type'] != 'Unknown Threat':
                            st.markdown(f"**Type:** {threat['threat_type']}")
                        st.markdown(f"**Published:** {threat.get('published_date', 'Unknown')}")
                        
                        if 'score' in threat and threat['score'] != 'Unknown':
                            if threat.get('ai_generated'):
                                st.markdown(f"**AI Confidence:** {threat['score']}")
                            else:
                                st.markdown(f"**CVSS Score:** {threat['score']}")
                        
                        if 'confidence' in threat and threat['confidence'] > 0:
                            confidence_label = "AI Confidence" if threat.get('ai_generated') else "Confidence"
                            st.progress(threat['confidence'], text=f"{confidence_label}: {threat['confidence']:.1%}")
                    
                    with col2:
                        severity = threat.get('severity', 'Unknown')
                        st.markdown(get_badge_html(severity), unsafe_allow_html=True)
                        
                        # Add AI indicator
                        if threat.get('ai_generated'):
                            st.markdown('<span style="color: #00ff88; font-size: 0.8rem;">🤖 AI Analysis</span>', unsafe_allow_html=True)
                    
                    # Affected Systems
                    if threat.get('affected_systems'):
                        st.markdown(f"**Affected Systems:** {', '.join(threat['affected_systems'])}")
                    
                    # Description
                    with st.expander("📋 AI Threat Analysis" if threat.get('ai_generated') else "📋 Description"):
                        st.markdown(threat.get('description', 'No description available.'))
                    
                    # Suggested Fixes
                    if threat.get('suggested_fixes'):
                        with st.expander("🔧 AI-Recommended Fixes" if threat.get('ai_generated') else "🔧 Suggested Fixes"):
                            for i, fix in enumerate(threat['suggested_fixes'], 1):
                                st.markdown(f"{i}. {fix}")
                    
                    # References (only for non-AI threats)
                    if threat.get('references') and not threat.get('ai_generated'):
                        with st.expander("🔗 References"):
                            for ref in threat['references']:
                                if ref.startswith('http'):
                                    st.markdown(f"- [{ref}]({ref})")
                                else:
                                    st.markdown(f"- {ref}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("")
    
    with tab3:
        st.markdown("## 📊 Security Analytics Dashboard")
        
        # Only show charts if we have data
        has_anomaly_data = anomalies and len(anomalies) > 0
        has_threat_data = threats and len(threats) > 0
        
        if has_anomaly_data or has_threat_data:
            col1, col2 = st.columns(2)
            
            with col1:
                if has_anomaly_data:
                    timeline_fig = create_anomaly_timeline(anomalies)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                else:
                    st.markdown("### 📈 Anomaly Timeline")
                    st.info("No anomaly data available. Run the GNN inference script to generate data.")
            
            with col2:
                if has_threat_data:
                    dist_fig = create_threat_distribution(threats)
                    if dist_fig:
                        st.plotly_chart(dist_fig, use_container_width=True)
                else:
                    st.markdown("### 🎯 Threat Distribution")
                    st.info("No threat data available. Run the CVE fetching script to generate data.")
            

            # Feature importance heatmap for recent anomalies
            if anomalies:
                st.markdown("### 🔥 Feature Importance Analysis")
                
                feature_data = []
                for anomaly in anomalies:
                    if isinstance(anomaly, dict) and 'why' in anomaly and anomaly['why']:
                        for feature in anomaly['why']:
                            if isinstance(feature, dict) and all(key in feature for key in ['feature', 'abs_error']):
                                feature_data.append({
                                    'stream_index': anomaly.get('stream_index', 0),
                                    'feature': feature['feature'],
                                    'abs_error': feature['abs_error']
                                })
                
                if feature_data:
                    df_features = pd.DataFrame(feature_data)
                    if len(df_features) > 0:
                        try:
                            pivot_df = df_features.pivot(index='feature', columns='stream_index', values='abs_error')
                            
                            fig = px.imshow(pivot_df, 
                                           color_continuous_scale='Reds',
                                           title="Feature Contribution to Anomaly Detection")
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font_color='white',
                                title_font_color='#667eea'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.info("Feature importance data not available in current format")
                else:
                    st.info("No feature importance data available. Run GNN inference to generate explainable anomalies.")
        else:
            st.markdown("### 🚀 Getting Started")
            st.info("No data available yet. Please run the background scripts to start monitoring:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                **🔍 Start Anomaly Detection:**
                \`\`\`bash
                python simulation_and_detection_/src/stream_inference.py
                \`\`\`
                """)
            
            with col2:
                st.markdown("""
                **🎯 Start Threat Monitoring:**
                \`\`\`bash
                python predictive_ai/fetch_threats.py
                \`\`\`
                """)
    
    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

if __name__ == "__main__":
    main()
