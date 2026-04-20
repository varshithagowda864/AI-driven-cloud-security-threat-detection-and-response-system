import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import time
import io # Used for handling file-like objects

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AI Local Cloud Security Dashboard")

# --- Session State Initialization ---
# This ensures that variables persist across reruns of the Streamlit app
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = 'Dashboard'
if 'uploaded_file_name' not in st.session_state:
    st.session_state.uploaded_file_name = None
if 'file_scan_status' not in st.session_state:
    st.session_state.file_scan_status = 'Pending' # 'Pending', 'CLEAN', 'MALICIOUS'
if 'file_content' not in st.session_state:
    st.session_state.file_content = "" # Stores the content of the uploaded/manipulated file

# --- Functions (Data Generation & Model Training) ---

@st.cache_data
def generate_log_data(n_samples=1000, n_anomalies=15):
    """Generates synthetic log data for general anomaly detection."""
    np.random.seed(42)
    data = {
        'Login_Attempts': np.random.normal(5, 1.5, n_samples),
        'API_Calls_Per_Minute': np.random.normal(20, 5, n_samples),
        'Data_Accessed_MB': np.abs(np.random.normal(10, 3, n_samples)),
    }
    df = pd.DataFrame(data)
    # Introduce Anomaly 1: Brute Force Attempt
    df.iloc[700:700 + n_anomalies, df.columns.get_loc('Login_Attempts')] = np.random.uniform(20, 30, n_anomalies) 
    # Introduce Anomaly 2: Data Exfiltration
    df.iloc[850:850 + n_anomalies, df.columns.get_loc('Data_Accessed_MB')] = np.random.uniform(50, 100, n_anomalies)
    df = df.apply(lambda x: np.maximum(x, 0))
    df['Resource_ID'] = [f'res-000{i}' for i in range(n_samples)]
    df['User_ID'] = [f'user_{np.random.randint(100, 110)}' for _ in range(n_samples)]
    return df

@st.cache_data
def generate_ddos_data(n_samples=200): # Reduced samples for faster graph refresh
    """Generates synthetic network data for DDoS classification and flow graph."""
    np.random.seed(int(time.time())) # Use current time for more dynamic DDoS simulation
    
    time_index = pd.to_datetime(pd.date_range('2025-01-01', periods=n_samples, freq='s'))
    
    # Normal Traffic (mostly low packet rate)
    packet_rate = np.abs(np.random.normal(10, 3, n_samples))
    
    # Introduce random DDoS spike
    if np.random.rand() < 0.5: # 50% chance of a DDoS attack in this simulation window
        attack_start_idx = np.random.randint(n_samples // 4, n_samples // 2)
        attack_end_idx = attack_start_idx + np.random.randint(n_samples // 8, n_samples // 4)
        attack_end_idx = min(attack_end_idx, n_samples)
        
        packet_rate[attack_start_idx:attack_end_idx] = np.abs(np.random.normal(150, 40, attack_end_idx - attack_start_idx))
    
    df = pd.DataFrame({
        'Timestamp': time_index,
        'Packet_Rate': packet_rate,
        'Avg_Packet_Size': np.abs(np.random.normal(100, 20, n_samples)), # Keep Avg_Packet_Size mostly normal or slightly low during attack
    })
    
    # Assign attack label for training and real-time classification
    df['Is_DDoS_Attack'] = 0
    # Simulate based on Packet_Rate directly for simplicity in data generation
    df.loc[df['Packet_Rate'] > 50, 'Is_DDoS_Attack'] = 1 
    
    return df

@st.cache_resource
def train_models(log_df, ddos_df_for_training):
    # Model 1: General Anomaly Detection (Isolation Forest)
    log_features = ['Login_Attempts', 'API_Calls_Per_Minute', 'Data_Accessed_MB']
    if_model = IsolationForest(contamination=0.03, random_state=42)
    if_model.fit(log_df[log_features])

    # Model 2: DDoS Classification (KNN)
    ddos_features_train = ['Packet_Rate', 'Avg_Packet_Size']
    # Use a small subset of the DDoS data for classification training
    X_ddos = ddos_df_for_training[ddos_features_train]
    y_ddos = ddos_df_for_training['Is_DDoS_Attack'] # 1 is DDoS, 0 is NORMAL
    
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_ddos, y_ddos)
    
    return if_model, log_features, knn_model, ddos_features_train

# --- Threat Detection & Visualization Logic ---

def detect_threats(if_model, log_df, log_features):
    """Applies Isolation Forest to cloud logs and prepares data for visualization."""
    log_df['Anomaly_Score'] = if_model.decision_function(log_df[log_features])
    log_df['Prediction'] = if_model.predict(log_df[log_features])
    
    threats_df = log_df[log_df['Prediction'] == -1].sort_values(by='Anomaly_Score')
    
    # Determine Threat Type
    threats_df['Threat_Type'] = np.where(
        (threats_df['Login_Attempts'] > 15), 'Brute Force Attempt', 
        np.where(
            (threats_df['Data_Accessed_MB'] > 40), 'Data Exfiltration',
            'Generic Anomaly'
        )
    )
    threats_df['Threat_ID'] = [f'LOG-{i+1:03d}' for i in range(len(threats_df))]
    return threats_df

def detect_ddos_realtime(knn_model, ddos_features):
    """Applies KNN to simulated current traffic for real-time detection."""
    current_ddos_data = generate_ddos_data(n_samples=50) # Simulate a smaller, recent window of traffic
    
    X_current = current_ddos_data[ddos_features]
    predictions = knn_model.predict(X_current)
    
    # 1 is DDoS, 0 is NORMAL
    ddos_count = np.sum(predictions == 1)
    
    # Simple threshold: If more than 20% of flows are classified as DDoS, trigger alert
    is_ddos = ddos_count > len(current_ddos_data) * 0.2
        
    # Add prediction column for visualization
    current_ddos_data['Predicted_Attack'] = predictions 
    
    return is_ddos, ddos_count, len(current_ddos_data), current_ddos_data

# --- Streamlit Dashboard Layout ---

def render_dashboard(if_model, log_features, knn_model, ddos_features):
    st.title("🛡️ AI-Driven Cloud Security Dashboard")
    st.caption("Real-time Monitoring & Automated Response for Local Cloud Resources (Simulated)")
    st.markdown("---")

    # --- Metrics Section ---
    # Trigger detection to update metrics
    log_data = generate_log_data()
    threats_df = detect_threats(if_model, log_data, log_features)
    ddos_alert, ddos_count, total_flows, current_traffic_data = detect_ddos_realtime(knn_model, ddos_features)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Resources Monitored", "1,024 VMs")
    col2.metric("Active Anomaly Alerts", f"{len(threats_df)}", delta=f"{len(threats_df)} potential incidents", delta_color="inverse")
    
    if ddos_alert:
        col3.metric("DDoS Attack Status", "CRITICAL", delta=f"High Volume Traffic ({ddos_count} flows detected)", delta_color="inverse")
    else:
        col3.metric("DDoS Attack Status", "NORMAL", delta="No Attack Detected", delta_color="normal")
        
    st.markdown("---")

    # --- Detection & Visualization Tabs ---
    tab1, tab2 = st.tabs(["🔒 General Anomaly Detection", "🌐 DDoS Detection & Mitigation"])

    with tab1:
        st.subheader("Cloud API & User Behavioral Analysis")
        
        if len(threats_df) > 0:
            st.error(f"🚨 **{len(threats_df)} Critical Anomalies Detected!**")
            
            # Threat Type Distribution Graph
            threat_counts = threats_df['Threat_Type'].value_counts().reset_index()
            threat_counts.columns = ['Threat_Type', 'Count']
            
            fig_anomaly, ax_anomaly = plt.subplots(figsize=(8, 4))
            ax_anomaly.bar(threat_counts['Threat_Type'], threat_counts['Count'], color=['red', 'orange', 'gray'])
            ax_anomaly.set_title('Threat Type Distribution')
            ax_anomaly.set_ylabel('Number of Incidents')
            ax_anomaly.tick_params(axis='x', rotation=15)
            st.pyplot(fig_anomaly)
            
            # Automated Response Table
            st.warning("Automated Response: Isolating highest-risk users/resources.")
            st.dataframe(threats_df[['Threat_ID', 'Threat_Type', 'User_ID', 'Login_Attempts', 'Data_Accessed_MB']], use_container_width=True)
        else:
            st.success("No critical general anomalies detected in the latest log scan.")

    with tab2:
        st.subheader("Ingress Network Traffic Flow Analysis")
        
        # DDoS Traffic Flow Graph
        fig_ddos, ax_ddos = plt.subplots(figsize=(10, 5))
        
        # Plot total packet rate
        ax_ddos.plot(current_traffic_data['Timestamp'], current_traffic_data['Packet_Rate'], 
                     label='Packet Rate', color='blue')
        
        # Highlight AI-predicted DDoS periods
        attack_periods = current_traffic_data[current_traffic_data['Predicted_Attack'] == 1]
        if not attack_periods.empty:
            ax_ddos.scatter(attack_periods['Timestamp'], attack_periods['Packet_Rate'], 
                           color='red', label='AI Predicted DDoS', zorder=5)

        ax_ddos.set_xlabel('Time')
        ax_ddos.set_ylabel('Packet Rate (Flows/sec)')
        ax_ddos.set_title('Real-Time Network Traffic Flow')
        ax_ddos.legend()
        st.pyplot(fig_ddos)
        
        if ddos_alert:
            st.error("🔴 **DDoS ATTACK IN PROGRESS:** Immediate Mitigation Required!")
            st.warning("⚡ **Mitigation Action:** Activating **Rate Limiting** and **Black-Holing** (Simulated SOAR Playbook).")
        else:
            st.success("Network traffic flows are currently stable.")

# --- Simulated File/Folder Interface with Manipulation ---

def render_file_explorer():
    st.title("📁 Local Cloud Storage Explorer & Secure File Manipulation")
    st.caption("Simulated file operations by **User B** on **User A's** folder—monitored for threat detection.")
    st.markdown("---")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. File Upload & AI Scanning")
        uploaded_file = st.file_uploader("Select a file to upload (Triggers malware scan):", type=['txt', 'csv', 'json'], key="file_uploader")
        
        if uploaded_file is not None:
            # Only process if a new file is uploaded or if the name changed
            if st.session_state.uploaded_file_name != uploaded_file.name:
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.file_scan_status = 'Pending'
                # Decode content if it's text-based
                try:
                    st.session_state.file_content = uploaded_file.getvalue().decode("utf-8") 
                except UnicodeDecodeError:
                    st.session_state.file_content = "[Binary content - not displayed]" # Handle binary files
                st.success(f"File **{uploaded_file.name}** uploaded. Click 'Run AI Malware Scan'.")
            
            if st.session_state.file_scan_status == 'Pending':
                if st.button("Run AI Malware Scan", key="run_scan_button"):
                    with st.spinner(f"AI scanning {st.session_state.uploaded_file_name} for malicious signatures..."):
                        time.sleep(1.5)
                    
                    # Simple simulation: 10% chance of malware detection
                    if np.random.rand() < 0.15: # Slightly increased chance for demo
                        st.session_state.file_scan_status = 'MALICIOUS'
                        st.session_state.file_content = "" # Clear content if malicious
                        st.error("❌ **ALERT: File Classified as Malicious!** Access denied and file quarantined.")
                    else:
                        st.session_state.file_scan_status = 'CLEAN'
                        st.success("✅ File scan clean. Access granted for manipulation.")
                        
        else: # If file uploader is empty
            st.session_state.uploaded_file_name = None
            st.session_state.file_scan_status = 'Pending'
            st.session_state.file_content = ""
            st.info("Upload a file to begin.")

    with col2:
        st.subheader("2. File Manipulation (Read/Write)")
        
        if st.session_state.file_scan_status == 'CLEAN':
            
            st.success(f"Security Check Passed for: **{st.session_state.uploaded_file_name}**")
            
            # Read and Write Simulation
            edited_content = st.text_area(
                "Simulated File Content (Manipulation Allowed):", 
                value=st.session_state.file_content,
                height=250,
                key="file_content_editor"
            )
            
            if st.button("💾 Save Changes (Write Operation)", key="save_file_button"):
                st.session_state.file_content = edited_content
                
                # Simulate a secondary AI check on write operation (e.g., policy check, sensitive data)
                with st.spinner("Re-analyzing content for security policy violations..."):
                    time.sleep(1)
                
                # Simple check for "sensitive" keyword after modification
                if "sensitive" in edited_content.lower() and "sensitive" not in st.session_state.file_content.lower():
                     st.warning("⚠️ **POLICY ALERT:** Added 'sensitive' keyword to file. Content monitoring triggered.")

                st.success("File content saved securely. Write operation logged for UEBA audit.")
            
            st.download_button(
                label="Download Current Content",
                data=st.session_state.file_content,
                file_name=st.session_state.uploaded_file_name if st.session_state.uploaded_file_name else "manipulated_file.txt",
                mime="text/plain",
                key="download_file_button"
            )
            
            st.caption("Every read/write action is logged and feeds into the Anomaly Detection model.")

        elif st.session_state.file_scan_status == 'MALICIOUS':
            st.error("Access Revoked. Cannot manipulate a quarantined file.")
        else:
            st.warning("Please upload a file and run the AI Malware Scan to enable manipulation.")

# --- Main App Execution ---

def app():
    # Load and Train Models once at startup
    log_data = generate_log_data()
    # Ensure ddos_data for training is distinct and consistent
    ddos_data_for_training = generate_ddos_data(n_samples=500) 
    if_model, log_features, knn_model, ddos_features = train_models(log_data, ddos_data_for_training)

    # Sidebar Navigation
    with st.sidebar:
        st.header("Navigation")
        if st.button("🏠 Dashboard View", key="nav_dashboard"):
            st.session_state.current_tab = 'Dashboard'
        if st.button("📁 File Explorer (Simulated)", key="nav_file"):
            st.session_state.current_tab = 'File Explorer'
        st.markdown("---")
        st.header("Security Status")
        st.metric("Last File Scan Status", st.session_state.file_scan_status)

    if st.session_state.current_tab == 'Dashboard':
        render_dashboard(if_model, log_features, knn_model, ddos_features)
    elif st.session_state.current_tab == 'File Explorer':
        render_file_explorer()

if __name__ == "__main__":
    app()