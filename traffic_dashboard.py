import streamlit as st
import pandas as pd
import numpy as np
import folium
import os
import cv2
import time
from streamlit_folium import st_folium
from ultralytics import YOLO

# ================= Page Config =================
st.set_page_config(page_title="TrafficPulse Dashboard", page_icon="🚦", layout="wide")

# ================ Custom CSS =================
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #2c3e50 0%, #34495e 100%);
    padding: 2rem;
    border-radius: 10px;
    margin-bottom: 2rem;
    color: white;
    text-align: center;
}

.authentic-badge {
    background: linear-gradient(45deg, #3498db, #27ae60);
    color: white;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: bold;
    display: inline-block;
    margin: 1rem 0;
}

.info-note {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-left: 4px solid #3498db;
    padding: 1rem;
    margin: 1rem 0;
    border-radius: 5px;
    font-size: 0.9rem;
    color: #2c3e50;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid #e9ecef;
    text-align: center;
}

.section-header {
    color: #2c3e50;
    font-weight: 600;
    margin: 2rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #3498db;
}

.status-indicator {
    display: inline-block;
    width: 12px;
    height: 12px;
    border-radius: 50%;
    margin-right: 8px;
}

.footer {
    background: #2c3e50;
    color: white;
    text-align: center;
    padding: 1rem;
    border-radius: 10px;
    margin-top: 3rem;
}

.debug-info {
    background: #f1f2f6;
    border: 1px solid #ddd;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
    font-family: monospace;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# ================ OpenCV Video Processing =================
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('yolov8s.pt')
        return model
    except:
        st.error("❌ YOLO model not found. Please ensure 'yolov8s.pt' is in the project directory.")
        return None

def process_video_frame(frame, model, tracker, down, up, counter_down, counter_up):
    if model is None:
        return frame, down, up, counter_down, counter_up
    
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")
    list = []

    # Filter for vehicles
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'][d]
        if 'car' in c:
            list.append([x1, y1, x2, y2])
    
    bbox_id = tracker.update(list)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        red_line_y = 198
        blue_line_y = 268
        offset = 6

        if red_line_y < (cy + offset) and red_line_y > (cy - offset):
            down[id] = time.time()
        if id in down:
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]
                if counter_down.count(id) == 0:
                    counter_down.append(id)
                    distance = 10
                    a_speed_ms = distance / elapsed_time
                    a_speed_kh = a_speed_ms * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()
        if id in up:
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed1_time = time.time() - up[id]
                if counter_up.count(id) == 0:
                    counter_up.append(id)
                    distance1 = 10
                    a_speed_ms1 = distance1 / elapsed1_time
                    a_speed_kh1 = a_speed_ms1 * 3.6
                    cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    text_color = (0, 0, 0)
    yellow_color = (0, 255, 255)
    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)

    cv2.rectangle(frame, (0, 0), (250, 90), yellow_color, -1)
    cv2.line(frame, (172, 198), (774, 198), red_color, 2)
    cv2.putText(frame, ('Red Line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
    cv2.putText(frame, ('Blue Line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Going Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
    cv2.putText(frame, ('Going Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

    return frame, down, up, counter_down, counter_up

# ================ Helper Functions =================
@st.cache_data
def load_data(csv_path='traffic_data.csv'):
    try:
        df = pd.read_csv(csv_path)
        if 'traffic_volume' in df.columns:
            max_volume = df['traffic_volume'].max()
            df['congestion'] = (df['traffic_volume'] / max_volume * 100).round(1)
            df['speed'] = np.clip(80 - (df['traffic_volume'] / max_volume * 40), 20, 80).round(1)
            df['camera'] = 'Camera-' + df['location'].astype(str)
        return df
    except:
        n = 6
        df = pd.DataFrame({
            'camera': [f'Camera-{i+1}' for i in range(n)],
            'location': [f'Sensor-{i+1}' for i in range(n)],
            'latitude': np.linspace(12.97, 12.99, n),
            'longitude': np.linspace(77.59, 77.61, n),
            'speed': np.random.normal(40, 8, size=n),
            'congestion': np.random.uniform(20, 60, size=n)
        })
        return df

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def create_map(df):
    if df.empty:
        return None
    
    lat_col = 'latitude' if 'latitude' in df.columns else 'Latitude'
    lon_col = 'longitude' if 'longitude' in df.columns else 'Longitude'
    camera_col = 'camera' if 'camera' in df.columns else 'Camera'
    location_col = 'location' if 'location' in df.columns else 'Location'
    speed_col = 'speed' if 'speed' in df.columns else 'Speed'
    congestion_col = 'congestion' if 'congestion' in df.columns else 'Congestion'
    
    center_lat = df[lat_col].mean()
    center_lon = df[lon_col].mean()
    
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    for _, row in df.iterrows():
        color = 'green' if row[congestion_col] < 40 else 'orange' if row[congestion_col] < 70 else 'red'
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=8,
            color=color,
            fill=True,
            fill_opacity=0.7
        ).add_to(m)
        
        folium.Marker(
            location=[row[lat_col], row[lon_col]],
            popup=f"<b>{row[camera_col]}</b><br/>{row[location_col]}<br/>Speed: {row[speed_col]:.1f} km/h<br/>Congestion: {row[congestion_col]:.1f}%",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
    
    return m

# ================= Load Data & Models =================
df = load_data()
model = load_yolo_model()

camera_col = 'camera' if 'camera' in df.columns else 'Camera'
location_col = 'location' if 'location' in df.columns else 'Location'
speed_col = 'speed' if 'speed' in df.columns else 'Speed'
congestion_col = 'congestion' if 'congestion' in df.columns else 'Congestion'

# ================= Sidebar =================
st.sidebar.title("🚦 TrafficPulse")
st.sidebar.markdown("---")

camera_options = ['Demo 1', 'Demo 2']
location_options = ['Demo 1', 'Demo 2']

camera_option = st.sidebar.selectbox("Camera", camera_options, index=0)
location_option = st.sidebar.selectbox("Location", location_options, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 System Status")
st.sidebar.markdown('<div class="status-indicator" style="background-color: #27ae60;"></div> **Online**', unsafe_allow_html=True)
st.sidebar.markdown('<div class="status-indicator" style="background-color: #3498db;"></div> **Active Sensors**', unsafe_allow_html=True)

st.sidebar.markdown("### 📈 Quick Stats")
st.sidebar.metric("Avg Speed", f"{df[speed_col].mean():.1f} km/h")
st.sidebar.metric("Avg Congestion", f"{df[congestion_col].mean():.1f}%")

# ================= Main Content =================
st.markdown('<div class="main-header"><h1>🚦 TrafficPulse Dashboard</h1><div class="authentic-badge">Smart Analytics</div></div>', unsafe_allow_html=True)
st.markdown('<div class="info-note"><strong>ℹ️ Technical Note:</strong> Speed calculations are based on pixel-by-pixel analysis using OpenCV algorithms and may slightly differ from accurate sensor measurements.</div>', unsafe_allow_html=True)

# ================= Debug Information =================
st.markdown('<h2 class="section-header">🔍 Debug Information</h2>', unsafe_allow_html=True)
debug_col1, debug_col2 = st.columns(2)

with debug_col1:
    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
    st.write(f"**Current Working Directory:** {os.getcwd()}")
    st.write(f"**Camera Selection:** {camera_option}")
    st.write(f"**Location Selection:** {location_option}")
    st.markdown('</div>', unsafe_allow_html=True)

with debug_col2:
    st.markdown('<div class="debug-info">', unsafe_allow_html=True)
    # Check for video files
    current_dir = os.getcwd()
    video_files = [f for f in os.listdir(current_dir) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    st.write(f"**Video files found:** {video_files}")
    
    # Specific file checks
    highway2_path = os.path.join(current_dir, 'highway2.mp4')
    output_path = os.path.join(current_dir, 'output.mp4')
    st.write(f"**highway2.mp4 exists:** {os.path.exists(highway2_path)}")
    st.write(f"**output.mp4 exists:** {os.path.exists(output_path)}")
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<h2 class="section-header">📹 Live Camera Feed with OpenCV Processing</h2>', unsafe_allow_html=True)

# Initialize tracking variables
if 'tracker' not in st.session_state:
    st.session_state.tracker = None
if 'down' not in st.session_state:
    st.session_state.down = {}
if 'up' not in st.session_state:
    st.session_state.up = {}
if 'counter_down' not in st.session_state:
    st.session_state.counter_down = []
if 'counter_up' not in st.session_state:
    st.session_state.counter_up = []

# ================= Enhanced Video Playback Logic =================
video_file = None
playback_rate = 1.0
current_dir = os.getcwd()

# Build absolute paths
highway2_path = os.path.join(current_dir, 'highway2.mp4')
output_path = os.path.join(current_dir, 'output.mp4')

# Enhanced debugging
st.write("### 🔍 Detailed File Check:")
col1, col2 = st.columns(2)

with col1:
    st.write(f"**Highway2 Path:** {highway2_path}")

# ================= Alternative File Check =================
st.markdown("### 🔍 Alternative File Discovery")
try:
    # Look for any video files in the directory
    all_files = os.listdir(current_dir)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    found_videos = [f for f in all_files if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if found_videos:
        st.success(f"✅ Found video files: {found_videos}")
        
        # Show a selectbox to manually choose any found video
        if len(found_videos) > 0:
            st.markdown("**Manual Video Selection:**")
            selected_video = st.selectbox("Choose a video to play:", ['None'] + found_videos)
            
            if selected_video != 'None':
                manual_video_path = os.path.join(current_dir, selected_video)
                st.video(manual_video_path)
                st.success(f"🎬 Now playing: {selected_video}")
    else:
        st.warning("⚠️ No video files found in the current directory")
        
except Exception as e:
    st.error(f"Error checking directory: {str(e)}")

# ================= Metrics Section =================
st.markdown('<h2 class="section-header">📊 Traffic Metrics</h2>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="metric-card"><h3>🚗 Average Speed</h3><h2 style="color: #3498db;">{:.1f} km/h</h2></div>'.format(df[speed_col].mean()), unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-card"><h3>🚦 Congestion Level</h3><h2 style="color: #e74c3c;">{:.1f}%</h2></div>'.format(df[congestion_col].mean()), unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-card"><h3>📡 Active Sensors</h3><h2 style="color: #27ae60;">{}</h2></div>'.format(len(df)), unsafe_allow_html=True)

# ================= Map Section =================
st.markdown('<h2 class="section-header">🗺️ Sensor Network Map</h2>', unsafe_allow_html=True)
map_obj = create_map(df)
if map_obj:
    st_folium(map_obj, height=450)
else:
    st.info("No sensor data available for map display.")

# ================= Footer =================
st.markdown('<div class="footer"><p>© 2024 TrafficPulse - Intelligent Traffic Monitoring System</p></div>', unsafe_allow_html=True)