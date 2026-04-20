import os
import json
import re
import requests
import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="RoadScan AI", page_icon="🛣️", layout="wide", initial_sidebar_state="collapsed")

MODEL_PATH = "road_damage_model.h5"
LOG_FILE   = "prediction_logs.csv"
UPLOAD_DIR = "saved_uploads"
LOGO_PATH  = "logo.png"
TRAINING_HISTORY_PATH = "training_history.json"
CLASS_INDEX_PATH = "class_indices.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@st.cache_resource
def load_trained_model():
    return load_model(MODEL_PATH)
model = load_trained_model()

if "page_mode" not in st.session_state:
    st.session_state.page_mode = "home"

def preprocess_image(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224)) / 255.0
    return np.expand_dims(img, axis=0)

def is_valid_road_image(image):
    img = np.array(image.convert("RGB"))
    h, w = img.shape[:2]
    if h < 120 or w < 120:
        return False, "Image is too small. Please upload a clearer road photo."

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    edges = cv2.Canny(gray, 60, 160)

    edge_ratio = float(np.mean(edges > 0))
    mean_intensity = float(np.mean(gray))
    low_sat_ratio = float(np.mean(hsv[:, :, 1] < 80))
    bright_ratio = float(np.mean(gray > 90))

    # Block obvious human selfies/portraits
    face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) > 0:
        return False, "Human face detected. Please upload or capture a road surface image only."

    valid = (
        0.02 <= edge_ratio <= 0.15 and
        60 <= mean_intensity <= 180 and
        low_sat_ratio >= 0.25 and
        bright_ratio >= 0.20
    )

    if not valid:
        return False, "Invalid image detected. Please upload or capture a clear road surface image only."
    return True, ""

def predict_damage(image):
    pred = model.predict(preprocess_image(image), verbose=0)[0][0]
    if pred < 0.5:
        return "Crack Detected", (1-pred)*100, "Poor", "High Risk"
    return "No Crack", pred*100, "Good", "Safe Surface"

def get_speed_recommendation(confidence, label):
    if label == "No Crack":
        if confidence >= 90: return 100,"SAFE","#00e5a0",None,"Road surface is in excellent condition. No significant deterioration expected."
        elif confidence >= 75: return 80,"MODERATE","#f5c842",8,"Minor surface wear. Estimated full deterioration ~8 years without maintenance."
        else: return 60,"CAUTION","#ff9f43",4,"Low model confidence. Surface condition uncertain — proceed with caution."
    else:
        if confidence >= 90: return 20,"DANGER","#ff3b3b",0.5,"Severe cracking confirmed. Road may become impassable in ~6 months without urgent repair."
        elif confidence >= 75: return 40,"HIGH RISK","#ff6b35",1.5,"Significant crack network. Estimated structural failure in ~1.5 years."
        else: return 50,"ELEVATED","#ffd166",2.5,"Cracks at moderate confidence. Estimated deterioration ~2.5 years."

def get_damage_timeline(label, confidence):
    _,_,_,damage_years,_ = get_speed_recommendation(confidence, label)
    if label == "No Crack":
        if damage_years is None:
            return list(range(0,16)), [min(100,i*3) for i in range(0,16)]
        years = list(range(0,int(damage_years*2)+3))
        return years, [min(100,(i/damage_years)*100) for i in years]
    years = [i*0.25 for i in range(0,int(damage_years*6)+4)]
    start = min(60,confidence*0.6)
    return years, [min(100,start+(100-start)*(t/damage_years)) for t in years]

def save_uploaded_image(active_input, source_name=None):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = (source_name or getattr(active_input, "name", "camera_capture.png")).replace("/", "_").replace(chr(92), "_")
    path = os.path.join(UPLOAD_DIR, f"{ts}_{safe_name}")
    data = active_input.getvalue() if hasattr(active_input, "getvalue") else active_input.read()
    with open(path, "wb") as f:
        f.write(data)
    return path

def append_log(filename, prediction, confidence, health, severity, saved_path, speed=None, risk=None, damage_years=None):
    row = {"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),"filename":filename,"prediction":prediction,
           "confidence":round(confidence,2),"road_health":health,"severity":severity,"saved_path":saved_path,
           "recommended_speed_kmh":speed or "","risk_level":risk or "","damage_timeline_years":damage_years or ""}
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        for col in row:
            if col not in df.columns: df[col]=""
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(LOG_FILE, index=False)

def load_logs():
    cols=["timestamp","filename","prediction","confidence","road_health","severity","saved_path","recommended_speed_kmh","risk_level","damage_timeline_years"]
    if os.path.exists(LOG_FILE):
        df=pd.read_csv(LOG_FILE)
        for c in cols:
            if c not in df.columns: df[c]=""
        return df
    return pd.DataFrame(columns=cols)

NH_DATABASE = {
    "NH 44": {
        "full_name":"National Highway 44","alias":"North-South Corridor","from_to":"Srinagar → Kanyakumari",
        "length_km":3745,"states":["J&K","Himachal Pradesh","Punjab","Haryana","Delhi","Uttar Pradesh","Madhya Pradesh","Maharashtra","Telangana","Andhra Pradesh","Karnataka","Tamil Nadu"],
        "lanes":"4–6 lane divided carriageway","surface":"Bituminous / Concrete","authority":"NHAI",
        "inaugurated":"2018 (renumbered)","daily_traffic":"38,000–95,000 PCUs","toll_plazas":47,
        "key_cities":["Jalandhar","Delhi","Agra","Bhopal","Nagpur","Hyderabad","Bangalore"],
        "notable":"Longest NH in India. Passes through 12 states and the UT of J&K.",
        "segments":[
            {"name":"Srinagar – Jammu","lat":33.7,"lon":75.8,"condition":"Crack Detected","conf":88,"terrain":"Mountainous"},
            {"name":"Jammu – Jalandhar","lat":31.8,"lon":75.5,"condition":"No Crack","conf":82,"terrain":"Plains"},
            {"name":"Jalandhar – Delhi","lat":29.5,"lon":77.2,"condition":"No Crack","conf":91,"terrain":"Plains"},
            {"name":"Delhi – Agra","lat":27.5,"lon":77.8,"condition":"Crack Detected","conf":76,"terrain":"Plains"},
            {"name":"Agra – Gwalior","lat":26.2,"lon":78.2,"condition":"No Crack","conf":85,"terrain":"Plains"},
            {"name":"Gwalior – Bhopal","lat":24.2,"lon":77.5,"condition":"Crack Detected","conf":91,"terrain":"Plateau"},
            {"name":"Bhopal – Nagpur","lat":22.5,"lon":78.4,"condition":"No Crack","conf":88,"terrain":"Plateau"},
            {"name":"Nagpur – Hyderabad","lat":20.3,"lon":79.1,"condition":"Crack Detected","conf":74,"terrain":"Deccan Plateau"},
            {"name":"Hyderabad – Kurnool","lat":16.9,"lon":78.0,"condition":"No Crack","conf":79,"terrain":"Deccan Plateau"},
            {"name":"Kurnool – Bangalore","lat":15.8,"lon":77.6,"condition":"No Crack","conf":92,"terrain":"Plateau"},
            {"name":"Bangalore – Krishnagiri","lat":12.9,"lon":77.8,"condition":"No Crack","conf":87,"terrain":"Hills"},
            {"name":"Krishnagiri – Kanyakumari","lat":10.5,"lon":77.5,"condition":"Crack Detected","conf":69,"terrain":"Coastal Plains"},
        ]
    },
    "NH 48": {
        "full_name":"National Highway 48","alias":"Delhi–Chennai Corridor","from_to":"Delhi → Chennai",
        "length_km":2807,"states":["Delhi","Haryana","Rajasthan","Gujarat","Maharashtra","Karnataka","Tamil Nadu"],
        "lanes":"4–8 lane divided","surface":"Bituminous","authority":"NHAI",
        "inaugurated":"2018 (renumbered from NH 8)","daily_traffic":"45,000–1,20,000 PCUs","toll_plazas":38,
        "key_cities":["Delhi","Gurugram","Jaipur","Ahmedabad","Surat","Mumbai","Pune","Bangalore","Chennai"],
        "notable":"One of India's busiest highways. Part of the Golden Quadrilateral.",
        "segments":[
            {"name":"Delhi – Gurugram","lat":28.5,"lon":77.0,"condition":"No Crack","conf":94,"terrain":"Urban"},
            {"name":"Gurugram – Jaipur","lat":27.4,"lon":76.5,"condition":"Crack Detected","conf":68,"terrain":"Semi-arid Plains"},
            {"name":"Jaipur – Ajmer","lat":26.8,"lon":75.4,"condition":"Crack Detected","conf":87,"terrain":"Arid Plains"},
            {"name":"Ajmer – Udaipur","lat":25.5,"lon":74.6,"condition":"No Crack","conf":79,"terrain":"Hilly"},
            {"name":"Udaipur – Vadodara","lat":23.4,"lon":73.8,"condition":"Crack Detected","conf":76,"terrain":"Plains"},
            {"name":"Vadodara – Surat","lat":22.3,"lon":73.2,"condition":"No Crack","conf":91,"terrain":"Coastal Plains"},
            {"name":"Surat – Mumbai","lat":20.8,"lon":73.0,"condition":"No Crack","conf":84,"terrain":"Coastal"},
            {"name":"Mumbai – Pune","lat":18.9,"lon":73.5,"condition":"Crack Detected","conf":82,"terrain":"Ghats"},
        ]
    },
    "NH 19": {
        "full_name":"National Highway 19","alias":"Grand Trunk Road (East)","from_to":"Delhi → Kolkata",
        "length_km":1435,"states":["Delhi","Uttar Pradesh","Bihar","Jharkhand","West Bengal"],
        "lanes":"4–6 lane divided","surface":"Bituminous / Concrete","authority":"NHAI",
        "inaugurated":"Part of historic GT Road, renumbered 2018","daily_traffic":"52,000–1,10,000 PCUs","toll_plazas":29,
        "key_cities":["Delhi","Agra","Kanpur","Varanasi","Patna","Kolkata"],
        "notable":"Follows the historic Grand Trunk Road. One of Asia's oldest and longest roads.",
        "segments":[
            {"name":"Delhi – Agra","lat":27.9,"lon":78.0,"condition":"No Crack","conf":88,"terrain":"Plains"},
            {"name":"Agra – Kanpur","lat":27.0,"lon":80.3,"condition":"Crack Detected","conf":79,"terrain":"Gangetic Plains"},
            {"name":"Kanpur – Allahabad","lat":25.7,"lon":81.8,"condition":"Crack Detected","conf":93,"terrain":"Gangetic Plains"},
            {"name":"Allahabad – Varanasi","lat":25.3,"lon":82.7,"condition":"No Crack","conf":72,"terrain":"Gangetic Plains"},
            {"name":"Varanasi – Patna","lat":25.6,"lon":84.5,"condition":"Crack Detected","conf":85,"terrain":"Plains"},
            {"name":"Patna – Dhanbad","lat":24.5,"lon":86.2,"condition":"Crack Detected","conf":77,"terrain":"Plains / Hills"},
            {"name":"Dhanbad – Kolkata","lat":23.5,"lon":87.5,"condition":"No Crack","conf":90,"terrain":"Plains"},
        ]
    },
    "NH 27": {
        "full_name":"National Highway 27","alias":"East–West Corridor","from_to":"Porbandar → Silchar",
        "length_km":3187,"states":["Gujarat","Rajasthan","Madhya Pradesh","Uttar Pradesh","Bihar","West Bengal","Assam"],
        "lanes":"4 lane divided","surface":"Bituminous","authority":"NHAI",
        "inaugurated":"2018 (renumbered)","daily_traffic":"22,000–65,000 PCUs","toll_plazas":34,
        "key_cities":["Porbandar","Rajkot","Ahmedabad","Udaipur","Bhopal","Varanasi","Patna","Silchar"],
        "notable":"Second longest NH in India. Part of India's East-West Corridor mega project.",
        "segments":[
            {"name":"Porbandar – Rajkot","lat":22.1,"lon":70.5,"condition":"No Crack","conf":86,"terrain":"Coastal / Plains"},
            {"name":"Rajkot – Ahmedabad","lat":22.9,"lon":72.0,"condition":"Crack Detected","conf":71,"terrain":"Plains"},
            {"name":"Ahmedabad – Udaipur","lat":24.1,"lon":73.4,"condition":"No Crack","conf":83,"terrain":"Hilly"},
            {"name":"Udaipur – Bhopal","lat":23.2,"lon":76.2,"condition":"Crack Detected","conf":88,"terrain":"Plateau"},
            {"name":"Bhopal – Jabalpur","lat":23.1,"lon":79.9,"condition":"No Crack","conf":77,"terrain":"Plateau"},
            {"name":"Jabalpur – Raipur","lat":22.0,"lon":81.6,"condition":"Crack Detected","conf":92,"terrain":"Vindhya Hills"},
            {"name":"Raipur – Varanasi","lat":23.8,"lon":83.5,"condition":"No Crack","conf":81,"terrain":"Plains"},
            {"name":"Varanasi – Silchar","lat":25.0,"lon":88.0,"condition":"Crack Detected","conf":74,"terrain":"Plains / Hills"},
        ]
    },
    "NH 52": {
        "full_name":"National Highway 52","alias":"Pathankot–Talcher Highway","from_to":"Pathankot → Talcher",
        "length_km":1900,"states":["Punjab","Himachal Pradesh","Haryana","Uttar Pradesh","Bihar","Jharkhand","Odisha"],
        "lanes":"4 lane divided","surface":"Bituminous","authority":"NHAI",
        "inaugurated":"2018 (renumbered)","daily_traffic":"18,000–50,000 PCUs","toll_plazas":22,
        "key_cities":["Pathankot","Hamirpur","Lucknow","Gaya","Ranchi","Talcher"],
        "notable":"Connects the northern Punjab plains to Odisha's mineral belt.",
        "segments":[
            {"name":"Pathankot – Hamirpur","lat":31.8,"lon":76.3,"condition":"No Crack","conf":80,"terrain":"Hilly"},
            {"name":"Hamirpur – Lucknow","lat":27.5,"lon":80.5,"condition":"Crack Detected","conf":72,"terrain":"Gangetic Plains"},
            {"name":"Lucknow – Gaya","lat":25.2,"lon":83.5,"condition":"Crack Detected","conf":86,"terrain":"Plains"},
            {"name":"Gaya – Ranchi","lat":23.5,"lon":84.8,"condition":"No Crack","conf":78,"terrain":"Chota Nagpur Plateau"},
            {"name":"Ranchi – Talcher","lat":21.5,"lon":85.5,"condition":"Crack Detected","conf":81,"terrain":"Plateau / Mining belt"},
        ]
    },
}

NH_ROUTE_POOL = [
    ("Srinagar", "Jammu"), ("Delhi", "Chandigarh"), ("Jaipur", "Udaipur"), ("Ahmedabad", "Vadodara"),
    ("Mumbai", "Pune"), ("Nagpur", "Hyderabad"), ("Bhopal", "Indore"), ("Lucknow", "Varanasi"),
    ("Patna", "Ranchi"), ("Kolkata", "Durgapur"), ("Guwahati", "Silchar"), ("Chennai", "Bengaluru"),
    ("Mysuru", "Mangaluru"), ("Kochi", "Thiruvananthapuram"), ("Raipur", "Bilaspur"), ("Kanpur", "Prayagraj"),
    ("Jabalpur", "Sagar"), ("Agra", "Gwalior"), ("Surat", "Mumbai"), ("Dehradun", "Haridwar")
]
NH_STATE_POOL = [
    ["Jammu & Kashmir", "Punjab"], ["Delhi", "Haryana", "Punjab"], ["Rajasthan"], ["Gujarat"],
    ["Maharashtra"], ["Maharashtra", "Telangana"], ["Madhya Pradesh"], ["Uttar Pradesh"],
    ["Bihar", "Jharkhand"], ["West Bengal"], ["Assam"], ["Tamil Nadu", "Karnataka"],
    ["Karnataka"], ["Kerala"], ["Chhattisgarh"], ["Uttar Pradesh"], ["Madhya Pradesh"],
    ["Uttar Pradesh", "Madhya Pradesh"], ["Gujarat", "Maharashtra"], ["Uttarakhand"]
]
NH_TERRAINS = ["Plains", "Hills", "Plateau", "Urban", "Coastal", "Semi-arid", "Forest Belt", "River Plain"]
NH_SURFACES = ["Bituminous", "Concrete", "Bituminous / Concrete"]
NH_LANES = ["2 lane", "4 lane divided", "4–6 lane divided", "6 lane divided"]

ROUTE_COORDS = {
    ("Srinagar", "Jammu"): [
        {"name": "Srinagar Bypass", "lat": 34.0837, "lon": 74.7973},
        {"name": "Qazigund Stretch", "lat": 33.5962, "lon": 75.1420},
        {"name": "Banihal Tunnel Zone", "lat": 33.4360, "lon": 75.1934},
        {"name": "Udhampur Section", "lat": 32.9253, "lon": 75.1352},
        {"name": "Jammu Entry", "lat": 32.7266, "lon": 74.8570},
    ],
    ("Delhi", "Chandigarh"): [
        {"name": "Delhi Outbound", "lat": 28.6139, "lon": 77.2090},
        {"name": "Sonipat Stretch", "lat": 28.9931, "lon": 77.0151},
        {"name": "Panipat Section", "lat": 29.3909, "lon": 76.9635},
        {"name": "Karnal Section", "lat": 29.6857, "lon": 76.9905},
        {"name": "Chandigarh Entry", "lat": 30.7333, "lon": 76.7794},
    ],
    ("Jaipur", "Udaipur"): [
        {"name": "Jaipur Exit", "lat": 26.9124, "lon": 75.7873},
        {"name": "Ajmer Stretch", "lat": 26.4499, "lon": 74.6399},
        {"name": "Bhilwara Section", "lat": 25.3478, "lon": 74.6408},
        {"name": "Nathdwara Belt", "lat": 24.9381, "lon": 73.8235},
        {"name": "Udaipur Entry", "lat": 24.5854, "lon": 73.7125},
    ],
    ("Ahmedabad", "Vadodara"): [
        {"name": "Ahmedabad Exit", "lat": 23.0225, "lon": 72.5714},
        {"name": "Nadiad Stretch", "lat": 22.6939, "lon": 72.8619},
        {"name": "Anand Section", "lat": 22.5645, "lon": 72.9289},
        {"name": "Karjan Belt", "lat": 22.0401, "lon": 73.1236},
        {"name": "Vadodara Entry", "lat": 22.3072, "lon": 73.1812},
    ],
    ("Mumbai", "Pune"): [
        {"name": "Mumbai Exit", "lat": 19.0760, "lon": 72.8777},
        {"name": "Panvel Section", "lat": 18.9894, "lon": 73.1175},
        {"name": "Lonavala Ghat", "lat": 18.7546, "lon": 73.4062},
        {"name": "Talegaon Stretch", "lat": 18.7353, "lon": 73.6750},
        {"name": "Pune Entry", "lat": 18.5204, "lon": 73.8567},
    ],
    ("Nagpur", "Hyderabad"): [
        {"name": "Nagpur Bypass", "lat": 21.1458, "lon": 79.0882},
        {"name": "Adilabad Stretch", "lat": 19.6667, "lon": 78.5333},
        {"name": "Nirmal Section", "lat": 19.0952, "lon": 78.3446},
        {"name": "Kamareddy Belt", "lat": 18.3256, "lon": 78.3418},
        {"name": "Hyderabad Entry", "lat": 17.3850, "lon": 78.4867},
    ],
    ("Bhopal", "Indore"): [
        {"name": "Bhopal Exit", "lat": 23.2599, "lon": 77.4126},
        {"name": "Sehore Stretch", "lat": 23.2032, "lon": 77.0851},
        {"name": "Ashta Section", "lat": 23.0179, "lon": 76.7221},
        {"name": "Dewas Belt", "lat": 22.9676, "lon": 76.0534},
        {"name": "Indore Entry", "lat": 22.7196, "lon": 75.8577},
    ],
    ("Lucknow", "Varanasi"): [
        {"name": "Lucknow Exit", "lat": 26.8467, "lon": 80.9462},
        {"name": "Sultanpur Stretch", "lat": 26.2589, "lon": 82.0727},
        {"name": "Jaunpur Section", "lat": 25.7464, "lon": 82.6837},
        {"name": "Babtpur Belt", "lat": 25.4524, "lon": 82.8593},
        {"name": "Varanasi Entry", "lat": 25.3176, "lon": 82.9739},
    ],
    ("Patna", "Ranchi"): [
        {"name": "Patna Exit", "lat": 25.5941, "lon": 85.1376},
        {"name": "Nawada Stretch", "lat": 24.8867, "lon": 85.5436},
        {"name": "Koderma Section", "lat": 24.4671, "lon": 85.5930},
        {"name": "Hazaribagh Belt", "lat": 23.9966, "lon": 85.3691},
        {"name": "Ranchi Entry", "lat": 23.3441, "lon": 85.3096},
    ],
    ("Kolkata", "Durgapur"): [
        {"name": "Kolkata Exit", "lat": 22.5726, "lon": 88.3639},
        {"name": "Dankuni Stretch", "lat": 22.6711, "lon": 88.2871},
        {"name": "Bardhaman Section", "lat": 23.2324, "lon": 87.8615},
        {"name": "Panagarh Belt", "lat": 23.4500, "lon": 87.4333},
        {"name": "Durgapur Entry", "lat": 23.5204, "lon": 87.3119},
    ],
    ("Guwahati", "Silchar"): [
        {"name": "Guwahati Exit", "lat": 26.1445, "lon": 91.7362},
        {"name": "Nagaon Stretch", "lat": 26.3500, "lon": 92.6833},
        {"name": "Hojai Section", "lat": 26.0000, "lon": 92.8667},
        {"name": "Badarpur Belt", "lat": 24.8680, "lon": 92.5961},
        {"name": "Silchar Entry", "lat": 24.8333, "lon": 92.7789},
    ],
    ("Chennai", "Bengaluru"): [
        {"name": "Chennai Exit", "lat": 13.0827, "lon": 80.2707},
        {"name": "Vellore Stretch", "lat": 12.9165, "lon": 79.1325},
        {"name": "Krishnagiri Section", "lat": 12.5186, "lon": 78.2137},
        {"name": "Hosur Belt", "lat": 12.7409, "lon": 77.8253},
        {"name": "Bengaluru Entry", "lat": 12.9716, "lon": 77.5946},
    ],
    ("Mysuru", "Mangaluru"): [
        {"name": "Mysuru Bypass", "lat": 12.2958, "lon": 76.6394},
        {"name": "Hunsur Stretch", "lat": 12.3030, "lon": 76.9050},
        {"name": "Madikeri Section", "lat": 12.4244, "lon": 75.7382},
        {"name": "Sakleshpur Ghat", "lat": 12.9417, "lon": 75.7842},
        {"name": "Mangaluru Ring Road", "lat": 12.9141, "lon": 74.8560},
    ],
    ("Kochi", "Thiruvananthapuram"): [
        {"name": "Kochi Exit", "lat": 9.9312, "lon": 76.2673},
        {"name": "Alappuzha Stretch", "lat": 9.4981, "lon": 76.3388},
        {"name": "Kollam Section", "lat": 8.8932, "lon": 76.6141},
        {"name": "Attingal Belt", "lat": 8.6969, "lon": 76.8153},
        {"name": "Thiruvananthapuram Entry", "lat": 8.5241, "lon": 76.9366},
    ],
    ("Raipur", "Bilaspur"): [
        {"name": "Raipur Exit", "lat": 21.2514, "lon": 81.6296},
        {"name": "Simga Stretch", "lat": 21.6333, "lon": 81.7000},
        {"name": "Bhatapara Section", "lat": 21.7333, "lon": 81.9500},
        {"name": "Tilda Belt", "lat": 21.5500, "lon": 81.7000},
        {"name": "Bilaspur Entry", "lat": 22.0797, "lon": 82.1391},
    ],
    ("Kanpur", "Prayagraj"): [
        {"name": "Kanpur Exit", "lat": 26.4499, "lon": 80.3319},
        {"name": "Fatehpur Stretch", "lat": 25.9300, "lon": 80.8000},
        {"name": "Khaga Section", "lat": 25.7667, "lon": 81.1000},
        {"name": "Handia Belt", "lat": 25.3667, "lon": 82.1833},
        {"name": "Prayagraj Entry", "lat": 25.4358, "lon": 81.8463},
    ],
    ("Jabalpur", "Sagar"): [
        {"name": "Jabalpur Exit", "lat": 23.1815, "lon": 79.9864},
        {"name": "Barela Stretch", "lat": 23.1000, "lon": 79.7500},
        {"name": "Damoh Section", "lat": 23.8333, "lon": 79.4500},
        {"name": "Rahatgarh Belt", "lat": 23.7833, "lon": 78.3833},
        {"name": "Sagar Entry", "lat": 23.8388, "lon": 78.7378},
    ],
    ("Agra", "Gwalior"): [
        {"name": "Agra Exit", "lat": 27.1767, "lon": 78.0081},
        {"name": "Dholpur Stretch", "lat": 26.7000, "lon": 77.9000},
        {"name": "Morena Section", "lat": 26.5000, "lon": 78.0000},
        {"name": "Banmore Belt", "lat": 26.5667, "lon": 78.1167},
        {"name": "Gwalior Entry", "lat": 26.2183, "lon": 78.1828},
    ],
    ("Surat", "Mumbai"): [
        {"name": "Surat Exit", "lat": 21.1702, "lon": 72.8311},
        {"name": "Vapi Stretch", "lat": 20.3700, "lon": 72.9000},
        {"name": "Dahanu Section", "lat": 19.9667, "lon": 72.7333},
        {"name": "Virar Belt", "lat": 19.4500, "lon": 72.8000},
        {"name": "Mumbai Entry", "lat": 19.0760, "lon": 72.8777},
    ],
    ("Dehradun", "Haridwar"): [
        {"name": "Dehradun Exit", "lat": 30.3165, "lon": 78.0322},
        {"name": "Doiwala Stretch", "lat": 30.1833, "lon": 78.1167},
        {"name": "Raiwala Section", "lat": 30.0167, "lon": 78.1833},
        {"name": "Jwalapur Belt", "lat": 29.9333, "lon": 78.1500},
        {"name": "Haridwar Entry", "lat": 29.9457, "lon": 78.1642},
    ],
}


def build_segment_chain(start_city, end_city, idx):
    route_key = (start_city, end_city)
    if route_key in ROUTE_COORDS:
        coords = ROUTE_COORDS[route_key]
        segments = []
        for s_idx, point in enumerate(coords):
            crack = ((idx + s_idx) % 3 == 0)
            conf = 68 + ((idx * 7 + s_idx * 5) % 28)
            segments.append({
                "name": point["name"],
                "lat": point["lat"],
                "lon": point["lon"],
                "condition": "Crack Detected" if crack else "No Crack",
                "conf": conf,
                "terrain": NH_TERRAINS[(idx + s_idx) % len(NH_TERRAINS)]
            })
        return segments

    base_lat = 8.0 + ((idx * 1.7) % 25)
    base_lon = 68.0 + ((idx * 2.3) % 20)
    segment_names = [
        f"{start_city} Bypass", f"{start_city} – Midway", f"Midway – {end_city}",
        f"{end_city} Approach", f"{end_city} Ring Road"
    ]
    segments = []
    for s_idx, name in enumerate(segment_names):
        crack = ((idx + s_idx) % 3 == 0)
        conf = 68 + ((idx * 7 + s_idx * 5) % 28)
        segments.append({
            "name": name,
            "lat": round(base_lat + s_idx * 0.65, 2),
            "lon": round(base_lon + s_idx * 0.72, 2),
            "condition": "Crack Detected" if crack else "No Crack",
            "conf": conf,
            "terrain": NH_TERRAINS[(idx + s_idx) % len(NH_TERRAINS)]
        })
    return segments


def ensure_extended_nh_database(limit=50):
    for nh_no in range(1, limit + 1):
        key = f"NH {nh_no}"
        if key in NH_DATABASE:
            continue
        pool_idx = (nh_no - 1) % len(NH_ROUTE_POOL)
        start_city, end_city = NH_ROUTE_POOL[pool_idx]
        states = NH_STATE_POOL[pool_idx]
        NH_DATABASE[key] = {
            "full_name": f"National Highway {nh_no}",
            "alias": f"Regional Corridor {nh_no}",
            "from_to": f"{start_city} → {end_city}",
            "length_km": 180 + nh_no * 29,
            "states": states,
            "lanes": NH_LANES[nh_no % len(NH_LANES)],
            "surface": NH_SURFACES[nh_no % len(NH_SURFACES)],
            "authority": "NHAI",
            "inaugurated": f"201{nh_no % 10} (representative data)",
            "daily_traffic": f"{18000 + nh_no * 850:,} PCUs",
            "toll_plazas": 2 + (nh_no % 9),
            "key_cities": [start_city, f"Central Hub {nh_no}", end_city],
            "notable": f"Representative route profile added for NH Scanner coverage up to NH {limit}.",
            "segments": build_segment_chain(start_city, end_city, nh_no)
        }


ensure_extended_nh_database(50)



def build_generated_segments(nh_number, count=6):
    base_lat = 8.5 + (nh_number % 25) * 0.8
    base_lon = 70.0 + (nh_number % 18) * 0.9
    terrains = ["Plains", "Hills", "Plateau", "Coastal", "Urban", "Semi-arid"]
    segments = []
    for idx in range(count):
        crack = ((nh_number + idx) % 3 == 0)
        conf = 68 + ((nh_number * 7 + idx * 5) % 27)
        segments.append({
            "name": f"Segment {chr(65 + idx)}",
            "lat": round(base_lat + idx * 0.45, 2),
            "lon": round(base_lon + idx * 0.38, 2),
            "condition": "Crack Detected" if crack else "No Crack",
            "conf": conf,
            "terrain": terrains[(nh_number + idx) % len(terrains)]
        })
    return segments


def extend_nh_database_to_50():
    for nh_number in range(1, 51):
        key = f"NH {nh_number}"
        if key in NH_DATABASE:
            continue
        start_city = f"City {nh_number}A"
        end_city = f"City {nh_number}B"
        NH_DATABASE[key] = {
            "full_name": f"National Highway {nh_number}",
            "alias": f"NH-{nh_number} Corridor",
            "from_to": f"{start_city} → {end_city}",
            "length_km": 180 + nh_number * 37,
            "states": [f"State {(nh_number % 8) + 1}", f"State {((nh_number + 2) % 8) + 1}"],
            "lanes": ["2 lane", "4 lane divided", "6 lane divided"][nh_number % 3],
            "surface": ["Bituminous", "Concrete", "Bituminous / Concrete"][nh_number % 3],
            "authority": "NHAI",
            "inaugurated": "2018 (project dataset)",
            "daily_traffic": f"{12000 + nh_number * 950:,} PCUs",
            "toll_plazas": 2 + (nh_number % 9),
            "key_cities": [start_city, f"City {nh_number}Mid", end_city],
            "notable": f"Representative project dataset entry for NH {nh_number} used for scanner demonstration.",
            "segments": build_generated_segments(nh_number, 6)
        }


extend_nh_database_to_50()

def extract_route_points(from_to):
    if "→" in from_to:
        start, end = [part.strip() for part in from_to.split("→", 1)]
        return start, end
    if "->" in from_to:
        start, end = [part.strip() for part in from_to.split("->", 1)]
        return start, end
    return "N/A", "N/A"

def infer_road_type(nh_data):
    lanes = nh_data.get("lanes", "N/A")
    surface = nh_data.get("surface", "N/A")
    terrain_types = sorted({seg.get("terrain", "Unknown") for seg in nh_data.get("segments", [])})
    primary_terrain = ", ".join(terrain_types[:3]) if terrain_types else "Mixed terrain"
    return f"{lanes} · {surface} · {primary_terrain}"

def normalize_nh_key(value):
    value = value.strip().upper().replace("-", " ")
    value = " ".join(value.split())
    compact = value.replace(" ", "")
    if compact.startswith("NH") and compact[2:].isdigit():
        return f"NH {int(compact[2:])}"
    return value


def get_nh_data(road_name):
    key = normalize_nh_key(road_name)
    if key in NH_DATABASE:
        v = NH_DATABASE[key]
        start_point, end_point = extract_route_points(v.get("from_to", ""))
        enriched = dict(v)
        enriched["start_point"] = start_point
        enriched["end_point"] = end_point
        enriched["road_type"] = infer_road_type(enriched)
        return enriched

    for k, v in NH_DATABASE.items():
        if normalize_nh_key(k) == key:
            start_point, end_point = extract_route_points(v.get("from_to", ""))
            enriched = dict(v)
            enriched["start_point"] = start_point
            enriched["end_point"] = end_point
            enriched["road_type"] = infer_road_type(enriched)
            return enriched

    fallback = {
        "full_name":f"{road_name} — National Highway","alias":"National Highway","from_to":"Route unavailable",
        "length_km":0,"states":[],"lanes":"N/A","surface":"N/A","authority":"NHAI","inaugurated":"N/A",
        "daily_traffic":"N/A","toll_plazas":0,"key_cities":[],"notable":"No additional data available.",
        "segments":[
            {"name":"Segment A","lat":20.0,"lon":78.0,"condition":"No Crack","conf":88,"terrain":"Plains"},
            {"name":"Segment B","lat":21.0,"lon":78.5,"condition":"Crack Detected","conf":76,"terrain":"Plains"},
            {"name":"Segment C","lat":22.0,"lon":79.0,"condition":"No Crack","conf":91,"terrain":"Plains"},
            {"name":"Segment D","lat":23.0,"lon":79.5,"condition":"Crack Detected","conf":83,"terrain":"Plateau"},
        ]
    }
    start_point, end_point = extract_route_points(fallback.get("from_to", ""))
    fallback["start_point"] = start_point
    fallback["end_point"] = end_point
    fallback["road_type"] = infer_road_type(fallback)
    return fallback


def safe_load_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

TRAINING_HISTORY = safe_load_json(TRAINING_HISTORY_PATH)
CLASS_INDEX_DATA = safe_load_json(CLASS_INDEX_PATH)

def get_best_training_metrics():
    phase1 = TRAINING_HISTORY.get("phase1", {})
    phase2 = TRAINING_HISTORY.get("phase2", {})
    best_val_acc = max((phase1.get("val_accuracy") or []) + (phase2.get("val_accuracy") or []) or [0])
    best_val_auc = max((phase1.get("val_auc") or []) + (phase2.get("val_auc") or []) or [0])
    return round(best_val_acc * 100, 2), round(best_val_auc * 100, 2)

def get_sorted_nh_keys():
    def sort_key(x):
        m = re.search(r"(\d+)", x)
        return int(m.group(1)) if m else 9999
    return sorted(NH_DATABASE.keys(), key=sort_key)

def get_segment_health_dataframe():
    rows = []
    for highway_name, highway_data in NH_DATABASE.items():
        for seg in highway_data.get("segments", []):
            speed, risk, _, damage_years, _ = get_speed_recommendation(seg["conf"], seg["condition"])
            rows.append({
                "highway": highway_name,
                "segment": seg.get("name", "Segment"),
                "terrain": seg.get("terrain", "Unknown"),
                "condition": seg.get("condition", "Unknown"),
                "confidence": seg.get("conf", 0),
                "speed_kmh": speed,
                "risk": risk,
                "damage_years": damage_years if damage_years is not None else "Stable",
                "lat": seg.get("lat"),
                "lon": seg.get("lon"),
            })
    return pd.DataFrame(rows)


PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8",family="Barlow, sans-serif",size=12),
    title_font=dict(color="#e2e8f0",family="Barlow Condensed, sans-serif",size=16),
    xaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="#1e293b",tickfont=dict(color="#64748b")),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)",zerolinecolor="#1e293b",tickfont=dict(color="#64748b")),
    margin=dict(t=50,b=40,l=50,r=20),
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&family=Barlow:wght@300;400;500;600&display=swap');
.stApp{background-color:#020617;background-image:linear-gradient(rgba(56,189,248,0.05) 1px,transparent 1px),linear-gradient(90deg,rgba(56,189,248,0.05) 1px,transparent 1px);background-size:48px 48px;color:#e2e8f0;font-family:'Barlow',sans-serif;}
.block-container{padding-top:0.8rem;padding-bottom:3rem;max-width:1320px;}
header[data-testid="stHeader"]{background:transparent!important;}
::-webkit-scrollbar{width:6px;}::-webkit-scrollbar-track{background:#0b1220;}::-webkit-scrollbar-thumb{background:#38bdf8;border-radius:3px;}
.nav-outer{border-bottom:1px solid #38bdf8;padding:0.7rem 0;margin-bottom:2rem;display:flex;align-items:center;gap:0.5rem;background:rgba(13,13,13,0.97);position:sticky;top:0;z-index:100;}
.nav-logo-text{font-family:'Barlow Condensed',sans-serif;font-size:1.5rem;font-weight:800;color:#38bdf8;letter-spacing:0.06em;text-transform:uppercase;margin-right:1.5rem;white-space:nowrap;}
div.stButton>button{background:transparent;border:1px solid #334155;color:#cbd5e1;font-family:'Barlow Condensed',sans-serif;font-size:0.9rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;border-radius:4px;height:38px;transition:all 0.18s ease;padding:0 1rem;}
div.stButton>button:hover{background:#38bdf8;border-color:#38bdf8;color:#020617;}
.overline{font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#38bdf8;letter-spacing:0.18em;text-transform:uppercase;margin-bottom:0.35rem;}
.page-hero{border-left:4px solid #38bdf8;padding:1.2rem 1.6rem;margin-bottom:2rem;background:rgba(56,189,248,0.08);}
.hero-title{font-family:'Barlow Condensed',sans-serif;font-size:2.8rem;font-weight:800;color:#f8fafc;line-height:1;letter-spacing:0.02em;text-transform:uppercase;}
.hero-sub{font-size:0.95rem;color:#94a3b8;margin-top:0.5rem;font-weight:300;line-height:1.7;max-width:680px;}
.card{background:#0f172a;border:1px solid #1e293b;border-radius:6px;padding:1.4rem;margin-bottom:1rem;}
.card-accent{background:#0f172a;border:1px solid #1e293b;border-top:3px solid #38bdf8;border-radius:6px;padding:1.4rem;margin-bottom:1rem;}
.card-danger{background:#170a0a;border:1px solid #3a1010;border-top:3px solid #ff3b3b;border-radius:6px;padding:1.4rem;margin-bottom:1rem;}
.card-safe{background:#0a1710;border:1px solid #103a20;border-top:3px solid #00e5a0;border-radius:6px;padding:1.4rem;margin-bottom:1rem;}
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:#1e293b;border:1px solid #1e293b;border-radius:6px;overflow:hidden;margin-bottom:1.5rem;}
.stat-cell{background:#0f172a;padding:1.2rem 1.4rem;}
.stat-label{font-family:'JetBrains Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.3rem;}
.stat-value{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:700;color:#38bdf8;line-height:1;}
.stat-unit{font-size:0.8rem;color:#94a3b8;margin-top:0.2rem;font-weight:300;}
.speed-readout{display:flex;align-items:baseline;gap:0.4rem;margin:0.6rem 0;}
.speed-num{font-family:'Barlow Condensed',sans-serif;font-size:4.5rem;font-weight:800;line-height:1;}
.speed-kmh{font-family:'JetBrains Mono',monospace;font-size:0.9rem;color:#94a3b8;letter-spacing:0.08em;}
.risk-badge{display:inline-block;font-family:'JetBrains Mono',monospace;font-size:0.72rem;font-weight:500;letter-spacing:0.14em;padding:0.28rem 0.7rem;border-radius:3px;text-transform:uppercase;}
.pred-label{font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;line-height:1.1;}
.conf-bar-outer{height:6px;background:#1e293b;border-radius:3px;margin:0.6rem 0 0.3rem;}
.conf-bar-inner{height:6px;border-radius:3px;}
.nh-info-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:1px;background:#1e293b;margin-bottom:1rem;}
.nh-info-cell{background:#0b1220;padding:0.8rem 1rem;}
.nh-info-key{font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:#64748b;letter-spacing:0.14em;text-transform:uppercase;margin-bottom:0.2rem;}
.nh-info-val{font-size:0.88rem;color:#d0c8b8;font-weight:400;}
.seg-row{display:flex;align-items:center;gap:0.8rem;padding:0.7rem 0.9rem;border-bottom:1px solid #172033;font-size:0.85rem;}
.seg-row:last-child{border-bottom:none;}
.seg-dot{width:10px;height:10px;border-radius:50%;flex-shrink:0;}
.seg-name{color:#cbd5e1;flex:1;font-weight:500;}
.seg-terrain{color:#4a4038;font-size:0.75rem;flex:0.8;}
.seg-conf{font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#94a3b8;width:50px;text-align:right;}
.seg-speed{font-family:'Barlow Condensed',sans-serif;font-size:0.95rem;font-weight:700;width:60px;text-align:right;}
.tag-row{display:flex;flex-wrap:wrap;gap:0.4rem;margin:0.6rem 0;}
.tag{font-family:'JetBrains Mono',monospace;font-size:0.65rem;letter-spacing:0.1em;padding:0.22rem 0.6rem;border:1px solid #334155;border-radius:3px;color:#94a3b8;text-transform:uppercase;}
.divider{height:1px;background:linear-gradient(90deg,#38bdf8 0%,#0f172a 60%,transparent 100%);margin:1.5rem 0;}
.sec-title{font-family:'Barlow Condensed',sans-serif;font-size:1.1rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;color:#e2e8f0;margin-bottom:0.8rem;}
.feat-cell{background:#0b1220;border:1px solid #1f2937;border-radius:4px;padding:1.3rem;height:100%;}
.feat-icon{font-size:1.4rem;margin-bottom:0.6rem;}
.feat-title{font-family:'Barlow Condensed',sans-serif;font-size:1rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#38bdf8;margin-bottom:0.3rem;}
.feat-desc{font-size:0.82rem;color:#64748b;line-height:1.6;}
div[data-testid="stFileUploader"]{background:#0b1220;border:1px dashed #334155;border-radius:6px;padding:0.5rem;}
div[data-testid="metric-container"]{background:#0b1220;border:1px solid #1f2937;border-radius:6px;padding:1rem;}
div[data-testid="metric-container"] label{color:#64748b!important;font-family:'JetBrains Mono',monospace!important;font-size:0.65rem!important;letter-spacing:0.12em!important;text-transform:uppercase!important;}
div[data-testid="metric-container"] [data-testid="metric-value"]{color:#38bdf8!important;font-family:'Barlow Condensed',sans-serif!important;font-size:2rem!important;font-weight:700!important;}
div[data-testid="stTextInput"] input{background:#0b1220!important;border:1px solid #334155!important;border-radius:4px!important;color:#e2e8f0!important;font-family:'JetBrains Mono',monospace!important;font-size:0.9rem!important;height:42px!important;}
div[data-testid="stTextInput"] input:focus{border-color:#38bdf8!important;box-shadow:0 0 0 1px #38bdf8!important;}
div[data-testid="stSelectbox"] > div{min-height:42px!important;}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div{min-height:42px!important;display:flex!important;align-items:center!important;}
div.stButton > button{height:42px!important;}
div[data-testid="stDataFrame"]{border-radius:6px;overflow:hidden;border:1px solid #1f2937;}
.footer{text-align:center;font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#334155;letter-spacing:0.12em;text-transform:uppercase;padding:2rem 0 0.5rem;border-top:1px solid #172033;margin-top:2rem;}
@media(max-width:768px){.stat-grid{grid-template-columns:repeat(2,1fr);}.hero-title{font-size:2rem;}.nh-info-grid{grid-template-columns:1fr;}}
</style>
""", unsafe_allow_html=True)

# NAV
st.markdown('<div class="nav-outer">', unsafe_allow_html=True)
nav_logo, nb1, nb2, nb3, nb4, nb5, nb6 = st.columns([1.9,1.05,1.05,1.2,1.35,1.0,1.0])
with nav_logo:
    st.markdown('<div class="nav-logo-text">⬡ RoadScan AI</div>', unsafe_allow_html=True)
pages=[("Home","home"),("Detect","detect"),("NH Scanner","gps"),("Dashboard","history"),("Cloud","cloud"),("About","about")]
for col,(label,key) in zip([nb1,nb2,nb3,nb4,nb5,nb6],pages):
    with col:
        if st.button(label,key=f"nav_{key}",use_container_width=True):
            st.session_state.page_mode=key; st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════ HOME ══════════════════════════
if st.session_state.page_mode == "home":
    hero_l, hero_r = st.columns([1.5,1], gap="large")
    with hero_l:
        if os.path.exists(LOGO_PATH): st.image(LOGO_PATH, width=72)
        st.markdown("""
        <div class="page-hero">
            <div class="overline">Final Year Project · AI Road Infrastructure</div>
            <div class="hero-title">Road Damage<br>Detection System</div>
            <div class="hero-sub">Deep learning-powered road surface inspection. Upload images for crack analysis, safety speed advisory, damage timeline prediction, and GPS-based National Highway health scanning.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="tag-row">', unsafe_allow_html=True)
        for t in ["MobileNetV2","Transfer Learning","OpenStreetMap","NHAI Data","Plotly Maps","Azure Ready"]:
            st.markdown(f'<span class="tag">{t}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.write("")
        cb1,cb2=st.columns(2)
        with cb1:
            if st.button("→ Start Detection",use_container_width=True): st.session_state.page_mode="detect";st.rerun()
        with cb2:
            if st.button("→ NH Scanner",use_container_width=True): st.session_state.page_mode="gps";st.rerun()
    with hero_r:
        best_val_acc, best_val_auc = get_best_training_metrics()
        best_val_acc_display = f"{best_val_acc:.2f}%" if best_val_acc else "N/A"
        best_val_auc_display = f"{best_val_auc:.2f}%" if best_val_auc else "N/A"
        class_mapping_display = " / ".join([f"{k}:{v}" for k, v in CLASS_INDEX_DATA.items()]) if CLASS_INDEX_DATA else "crack:0 / no_crack:1"
        st.markdown("""
        <div class="stat-grid">
            <div class="stat-cell"><div class="stat-label">Model</div><div class="stat-value" style="font-size:1.3rem;color:#e2e8f0;">MobileNetV2</div><div class="stat-unit">Transfer Learning</div></div>
            <div class="stat-cell"><div class="stat-label">Input Size</div><div class="stat-value">224</div><div class="stat-unit">px × 224px RGB</div></div>
            <div class="stat-cell"><div class="stat-label">NH Roads</div><div class="stat-value">50+</div><div class="stat-unit">Pre-loaded in DB</div></div>
            <div class="stat-cell"><div class="stat-label">Classes</div><div class="stat-value">2</div><div class="stat-unit">Crack / No Crack</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="margin-top:1rem;padding:1rem 1.2rem;">
            <div class="overline">Training Snapshot</div>
            <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:#1e293b;border:1px solid #1e293b;border-radius:6px;overflow:hidden;margin-top:0.4rem;">
                <div class="stat-cell"><div class="stat-label">Best Val Accuracy</div><div class="stat-value" style="font-size:1.45rem;">{best_val_acc_display}</div><div class="stat-unit">from training history</div></div>
                <div class="stat-cell"><div class="stat-label">Best Val AUC</div><div class="stat-value" style="font-size:1.45rem;">{best_val_auc_display}</div><div class="stat-unit">classification quality</div></div>
                <div class="stat-cell"><div class="stat-label">Class Mapping</div><div class="stat-value" style="font-size:1.05rem;color:#e2e8f0;">{class_mapping_display}</div><div class="stat-unit">loaded at startup</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        f1,f2=st.columns(2)
        feats=[("🔍","Crack Detection","Binary classification with confidence score per image."),
               ("🚗","Speed Advisory","Safe travel speed based on damage severity."),
               ("⏳","Damage Timeline","Projected road lifespan and deterioration curve."),
               ("🗺️","NH GPS Scanner","Segment-by-segment highway condition map.")]
        for col,(icon,title,desc) in zip([f1,f2,f1,f2],feats):
            with col:
                st.markdown(f'<div class="feat-cell"><div class="feat-icon">{icon}</div><div class="feat-title">{title}</div><div class="feat-desc">{desc}</div></div>', unsafe_allow_html=True)
                st.write("")

# ══════════════════════════ DETECT ══════════════════════════
elif st.session_state.page_mode == "detect":
    st.markdown("""
    <div class="page-hero">
        <div class="overline">Analysis Module</div>
        <div class="hero-title">Road Surface Inspection</div>
        <div class="hero-sub">Upload a road photograph. The model outputs crack detection, confidence score, recommended travel speed, and projected damage timeline.</div>
    </div>""", unsafe_allow_html=True)

    upload_tab, camera_tab = st.tabs(["Upload Image", "Use Camera"])
    uploaded_file = None
    captured_image = None
    with upload_tab:
        uploaded_file = st.file_uploader("Upload Road Image", type=["jpg","jpeg","png"])
    with camera_tab:
        captured_image = st.camera_input("Capture Road Image")

    active_input = uploaded_file if uploaded_file is not None else captured_image

    if active_input:
        image = Image.open(active_input).convert("RGB")
        source_name = getattr(active_input, "name", "camera_capture.png")
        img_col, res_col = st.columns([1,1.1], gap="large")
        with img_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f'<div style="margin-top:0.6rem;"><div class="overline">File</div><div style="font-family:JetBrains Mono,monospace;font-size:0.78rem;color:#64748b;">{source_name}</div></div></div>', unsafe_allow_html=True)
        with res_col:
            is_valid, validation_message = is_valid_road_image(image)
            if not is_valid:
                st.markdown("""
                <div class="card-danger">
                    <div class="overline">Input Validation</div>
                    <div class="pred-label" style="color:#ff3b3b;">❌ Invalid Image</div>
                    <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.7;margin-top:0.6rem;">""" + validation_message + """</div>
                </div>""", unsafe_allow_html=True)
                st.info("Use a clear image of a road surface for crack detection.")
                st.stop()

            with st.spinner("Running inference..."):
                prediction,confidence,health,severity = predict_damage(image)

            if confidence < 70:
                st.markdown("""
                <div class="card-danger">
                    <div class="overline">Input Validation</div>
                    <div class="pred-label" style="color:#ff3b3b;">❌ Invalid / Unclear Image</div>
                    <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.7;margin-top:0.6rem;">The image does not look like a clear road surface image for reliable detection. Please capture the road properly.</div>
                </div>""", unsafe_allow_html=True)
                st.info("Try using a clearer road image with the road surface visible in the frame.")
                st.stop()

            speed,risk,risk_color,damage_years,damage_note = get_speed_recommendation(confidence,prediction)
            saved_path = save_uploaded_image(active_input, source_name)
            append_log(source_name,prediction,confidence,health,severity,saved_path,speed,risk,damage_years)
            is_crack = prediction=="Crack Detected"
            card_cls = "card-danger" if is_crack else "card-safe"
            pred_color = "#ff3b3b" if is_crack else "#00e5a0"
            dot = "🔴" if is_crack else "🟢"
            bar_pct = int(confidence)
            st.markdown(f"""
            <div class="{card_cls}">
                <div class="overline">Prediction</div>
                <div class="pred-label" style="color:{pred_color};">{dot} {prediction}</div>
                <div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{bar_pct}%;background:{pred_color};"></div></div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.75rem;color:#64748b;">CONFIDENCE &nbsp;<span style="color:{pred_color};">{confidence:.1f}%</span>&nbsp;·&nbsp;HEALTH &nbsp;<span style="color:#cbd5e1;">{health}</span></div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="card-accent">
                <div class="overline">Safe Travel Speed</div>
                <div class="speed-readout"><span class="speed-num" style="color:{risk_color};">{speed}</span><span class="speed-kmh">km/h</span></div>
                <span class="risk-badge" style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}44;">{risk}</span>
            </div>""", unsafe_allow_html=True)
            if damage_years is not None:
                tl = f"{int(damage_years*12)} months" if damage_years<1 else f"{damage_years} years"
                st.markdown(f"""
                <div class="card">
                    <div class="overline">Estimated Time to Complete Damage</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2.4rem;font-weight:800;color:#38bdf8;line-height:1;margin:0.3rem 0;">{tl}</div>
                    <div style="font-size:0.82rem;color:#64748b;line-height:1.6;margin-top:0.5rem;">{damage_note}</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-safe">
                    <div class="overline">Road Lifespan</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;font-weight:800;color:#00e5a0;line-height:1;margin:0.3rem 0;">Excellent</div>
                    <div style="font-size:0.82rem;color:#64748b;line-height:1.6;margin-top:0.4rem;">{damage_note}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Damage Progression Forecast</div>', unsafe_allow_html=True)
        years,damage_pct = get_damage_timeline(prediction,confidence)
        line_c = "#ff3b3b" if is_crack else "#00e5a0"
        fill_c = "rgba(255,59,59,0.08)" if is_crack else "rgba(0,229,160,0.07)"
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=years,y=damage_pct,mode="lines",fill="tozeroy",fillcolor=fill_c,line=dict(color=line_c,width=2),name="Damage %"))
        fig.add_hline(y=100,line_dash="dash",line_color="rgba(255,59,59,0.3)",annotation_text="Complete Failure",annotation_font_color="#ff3b3b",annotation_font_size=11)
        fig.add_hline(y=70,line_dash="dot",line_color="rgba(255,160,0,0.3)",annotation_text="Critical Threshold",annotation_font_color="#38bdf8",annotation_font_size=11)
        fig.update_layout(**PLOT_LAYOUT,xaxis_title="Years from today",yaxis_title="Damage Level (%)",yaxis_range=[0,115],height=280)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">⚠ Estimates are model-based. Actual lifespan varies with traffic, climate, soil & maintenance.</div>', unsafe_allow_html=True)

# ══════════════════════════ NH SCANNER ══════════════════════════
elif st.session_state.page_mode == "gps":
    st.markdown("""
    <div class="page-hero">
        <div class="overline">GPS Highway Analysis</div>
        <div class="hero-title">NH Road Scanner</div>
        <div class="hero-sub">Search any Indian National Highway for segment-by-segment road condition analysis, speed advisories, damage timelines, plus start point, end point, and road type details.</div>
    </div>""", unsafe_allow_html=True)

    nh_options = get_sorted_nh_keys()
    quick_pick = st.session_state.get("quick_nh_pick", "")

    pick_col, srch_col, btn_col = st.columns([1.5, 2.8, 1], gap="small")
    with pick_col:
        selected_nh = st.selectbox(
            "Choose Highway",
            ["Select NH"] + nh_options,
            index=(( ["Select NH"] + nh_options).index(quick_pick) if quick_pick in nh_options else 0)
        )
    with srch_col:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        manual_nh = st.text_input(
            "",
            value="" if quick_pick else "",
            placeholder="Enter highway — e.g. NH 1, NH 7, NH 19, NH 27, NH 44, NH 50",
            label_visibility="collapsed"
        )
    with btn_col:
        st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
        scan_btn = st.button("SCAN →", use_container_width=True)


    feat1, feat2, feat3, feat4, feat5 = st.columns(5)
    for col, feat_nh in zip([feat1, feat2, feat3, feat4, feat5], ["NH 44", "NH 48", "NH 19", "NH 27", "NH 52"]):
        with col:
            if st.button(feat_nh, key=f"quick_{feat_nh}", use_container_width=True):
                st.session_state.quick_nh_pick = feat_nh
                st.rerun()

    road_input = manual_nh.strip() if manual_nh.strip() else (selected_nh if selected_nh != "Select NH" else quick_pick)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#475569;letter-spacing:0.1em;margin-bottom:1.5rem;">PRELOADED · NH 1 to NH 50 available · dropdown + manual search enabled</div>', unsafe_allow_html=True)

    if scan_btn and road_input.strip():
        nh=get_nh_data(road_input.strip())
        segs=nh["segments"]
        n_segs=len(segs)
        n_crack=sum(1 for s in segs if s["condition"]=="Crack Detected")
        n_safe=n_segs-n_crack
        health=round((n_safe/n_segs)*100)
        avg_spd=round(sum(get_speed_recommendation(s["conf"],s["condition"])[0] for s in segs)/n_segs)
        health_color="#00e5a0" if health>=60 else "#38bdf8" if health>=40 else "#ff3b3b"

        st.markdown(f"""
        <div class="stat-grid" style="grid-template-columns:repeat(5,1fr);">
            <div class="stat-cell"><div class="stat-label">Highway</div><div class="stat-value" style="font-size:1.4rem;color:#e2e8f0;">{road_input.strip().upper()}</div><div class="stat-unit">{nh['alias']}</div></div>
            <div class="stat-cell"><div class="stat-label">Total Length</div><div class="stat-value">{nh['length_km']:,}</div><div class="stat-unit">kilometres</div></div>
            <div class="stat-cell"><div class="stat-label">Health Score</div><div class="stat-value" style="color:{health_color};">{health}%</div><div class="stat-unit">{n_safe}/{n_segs} segments safe</div></div>
            <div class="stat-cell"><div class="stat-label">Avg Safe Speed</div><div class="stat-value">{avg_spd}</div><div class="stat-unit">km/h</div></div>
            <div class="stat-cell"><div class="stat-label">Toll Plazas</div><div class="stat-value">{nh['toll_plazas']}</div><div class="stat-unit">on this route</div></div>
        </div>""", unsafe_allow_html=True)

        info_col,map_col=st.columns([1,1.6],gap="large")
        with info_col:
            st.markdown(f"""
            <div class="overline">Highway Details</div>
            <div class="card" style="padding:0;">
            <div class="nh-info-grid">
                <div class="nh-info-cell" style="grid-column:span 2;"><div class="nh-info-key">Full Name</div><div class="nh-info-val" style="font-weight:600;color:#e2e8f0;">{nh['full_name']}</div></div>
                <div class="nh-info-cell" style="grid-column:span 2;"><div class="nh-info-key">Route</div><div class="nh-info-val" style="color:#38bdf8;">{nh['from_to']}</div></div>
                <div class="nh-info-cell"><div class="nh-info-key">Start Point</div><div class="nh-info-val">{nh['start_point']}</div></div>
                <div class="nh-info-cell"><div class="nh-info-key">End Point</div><div class="nh-info-val">{nh['end_point']}</div></div>
                <div class="nh-info-cell" style="grid-column:span 2;"><div class="nh-info-key">Road Type</div><div class="nh-info-val">{nh['road_type']}</div></div>
                <div class="nh-info-cell"><div class="nh-info-key">Authority</div><div class="nh-info-val">{nh['authority']}</div></div>
                <div class="nh-info-cell"><div class="nh-info-key">Inaugurated</div><div class="nh-info-val">{nh['inaugurated']}</div></div>
                <div class="nh-info-cell"><div class="nh-info-key">Lanes</div><div class="nh-info-val">{nh['lanes']}</div></div>
                <div class="nh-info-cell"><div class="nh-info-key">Surface</div><div class="nh-info-val">{nh['surface']}</div></div>
                <div class="nh-info-cell" style="grid-column:span 2;"><div class="nh-info-key">Daily Traffic</div><div class="nh-info-val">{nh['daily_traffic']}</div></div>
            </div></div>""", unsafe_allow_html=True)
            if nh['states']:
                states_html="".join(f'<span class="tag">{s}</span>' for s in nh['states'])
                st.markdown(f'<div class="overline" style="margin-top:1rem;">States Covered</div><div class="tag-row">{states_html}</div>', unsafe_allow_html=True)
            if nh['key_cities']:
                cities_html=" → ".join(f'<span style="color:#cbd5e1;">{c}</span>' for c in nh['key_cities'])
                st.markdown(f'<div class="card" style="margin-top:0.8rem;padding:1rem;"><div class="overline">Key Cities</div><div style="font-size:0.85rem;line-height:2;">{cities_html}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="card" style="border-left:3px solid #38bdf8;border-top:none;padding:1rem;"><div class="overline">Notable</div><div style="font-size:0.85rem;color:#cbd5e1;line-height:1.7;font-style:italic;">"{nh["notable"]}"</div></div>', unsafe_allow_html=True)

        with map_col:
            lats=[s["lat"] for s in segs]; lons=[s["lon"] for s in segs]
            colors=["#ff3b3b" if s["condition"]=="Crack Detected" else "#00e5a0" for s in segs]
            speeds=[get_speed_recommendation(s["conf"],s["condition"])[0] for s in segs]
            hover=[f"<b>{s['name']}</b><br>Terrain: {s['terrain']}<br>Condition: {s['condition']}<br>Confidence: {s['conf']}%<br>Safe Speed: {speeds[i]} km/h" for i,s in enumerate(segs)]
            fig_map=go.Figure()
            fig_map.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="lines",
                line=dict(width=4,color="rgba(56,189,248,0.55)"),
                showlegend=False,
                hoverinfo="skip"
            ))
            fig_map.add_trace(go.Scattermapbox(
                lat=lats,
                lon=lons,
                mode="markers",
                marker=dict(size=13,color=colors,opacity=0.95),
                hovertext=hover,
                hoverinfo="text",
                showlegend=False
            ))
            center_lat = sum(lats)/len(lats)
            center_lon = sum(lons)/len(lons)
            lat_span = max(lats) - min(lats)
            lon_span = max(lons) - min(lons)
            max_span = max(lat_span, lon_span)
            if max_span < 1:
                zoom = 7
            elif max_span < 2:
                zoom = 6
            elif max_span < 4:
                zoom = 5
            else:
                zoom = 4
            fig_map.update_layout(
                mapbox=dict(style="carto-positron",center=dict(lat=center_lat,lon=center_lon),zoom=zoom),
                paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=0,r=0,t=0,b=0),height=460
            )
            st.plotly_chart(fig_map,use_container_width=True)
            st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#475569;letter-spacing:0.1em;">● RED = CRACK DETECTED &nbsp;&nbsp; ● GREEN = NO CRACK &nbsp;&nbsp; ROUTE-ALIGNED VIEW</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Segment Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="padding:0.4rem 0;">', unsafe_allow_html=True)
        st.markdown('<div class="seg-row" style="border-bottom:1px solid #334155;padding-bottom:0.5rem;margin-bottom:0.2rem;"><div style="width:10px;"></div><div class="seg-name" style="color:#64748b;font-size:0.65rem;letter-spacing:0.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace;">Segment</div><div class="seg-terrain" style="color:#475569;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace;">Terrain</div><div class="seg-conf" style="color:#475569;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace;">Conf</div><div class="seg-speed" style="color:#475569;font-size:0.62rem;letter-spacing:0.1em;text-transform:uppercase;font-family:JetBrains Mono,monospace;">Speed</div></div>', unsafe_allow_html=True)
        for seg in segs:
            spd,risk,rc,dam_y,_=get_speed_recommendation(seg["conf"],seg["condition"])
            dot_color="#ff3b3b" if seg["condition"]=="Crack Detected" else "#00e5a0"
            dam_html=f'<span style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#64748b;"> · {int(dam_y*12)}mo</span>' if dam_y and dam_y<1 else (f'<span style="font-family:JetBrains Mono,monospace;font-size:0.65rem;color:#64748b;"> · {dam_y}yr</span>' if dam_y else "")
            st.markdown(f'<div class="seg-row"><div class="seg-dot" style="background:{dot_color};"></div><div class="seg-name">{seg["name"]}{dam_html}</div><div class="seg-terrain">{seg["terrain"]}</div><div class="seg-conf">{seg["conf"]}%</div><div class="seg-speed" style="color:{rc};">{spd}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Speed Advisory per Segment</div>', unsafe_allow_html=True)
        seg_names=[s["name"] for s in segs]
        seg_speeds=[get_speed_recommendation(s["conf"],s["condition"])[0] for s in segs]
        seg_colors=[get_speed_recommendation(s["conf"],s["condition"])[2] for s in segs]
        fig_bar=go.Figure(go.Bar(x=seg_names,y=seg_speeds,marker_color=seg_colors,marker_line_width=0,text=[f"{v} km/h" for v in seg_speeds],textposition="outside",textfont=dict(color="#94a3b8",size=11,family="JetBrains Mono")))
        fig_bar.update_layout(**PLOT_LAYOUT,yaxis_range=[0,130],yaxis_title="Safe Speed (km/h)",xaxis_tickfont=dict(size=10),height=300)
        st.plotly_chart(fig_bar,use_container_width=True)
        st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">⚠ Data is representative. Production version integrates live Street View imagery via Google Maps API.</div>', unsafe_allow_html=True)

    elif scan_btn:
        st.warning("Choose a highway from the dropdown, featured buttons, or type an NH name to scan.")

# ══════════════════════════ DASHBOARD ══════════════════════════
elif st.session_state.page_mode == "history":
    st.markdown("""
    <div class="page-hero">
        <div class="overline">Analytics</div>
        <div class="hero-title">Detection Dashboard</div>
        <div class="hero-sub">Review detection history, confidence distributions, speed advisories, and export records.</div>
    </div>""", unsafe_allow_html=True)

    logs_df=load_logs()
    total=len(logs_df)
    crack_c=int((logs_df["prediction"]=="Crack Detected").sum()) if total else 0
    safe_c=total-crack_c
    avg_conf=float(logs_df["confidence"].mean()) if total else 0.0
    m1,m2,m3,m4=st.columns(4)
    with m1: st.metric("Total Predictions",total)
    with m2: st.metric("Crack Cases",crack_c)
    with m3: st.metric("Safe Cases",safe_c)
    with m4: st.metric("Avg Confidence",f"{avg_conf:.1f}%")
    st.write("")
    if logs_df.empty:
        st.info("No predictions yet. Go to Detect to analyse a road image.")
    else:
        c1,c2=st.columns(2,gap="large")
        with c1:
            pc=logs_df["prediction"].value_counts().reset_index(); pc.columns=["Prediction","Count"]
            pie=px.pie(pc,names="Prediction",values="Count",hole=0.5,title="Crack vs No Crack",color="Prediction",color_discrete_map={"Crack Detected":"#ff3b3b","No Crack":"#00e5a0"})
            pie.update_layout(**PLOT_LAYOUT,legend=dict(orientation="h",y=-0.15,font=dict(color="#94a3b8")))
            st.plotly_chart(pie,use_container_width=True)
        with c2:
            sc=logs_df["severity"].value_counts().reset_index(); sc.columns=["Severity","Count"]
            bar=px.bar(sc,x="Severity",y="Count",title="Severity Breakdown",text="Count",color="Severity",color_discrete_sequence=["#38bdf8","#ff6b35","#ff3b3b","#00e5a0"])
            bar.update_layout(**PLOT_LAYOUT,xaxis_title="",yaxis_title="Count")
            st.plotly_chart(bar,use_container_width=True)
        if "recommended_speed_kmh" in logs_df.columns:
            spd_df=logs_df[logs_df["recommended_speed_kmh"]!=""].copy()
            if not spd_df.empty:
                spd_df["recommended_speed_kmh"]=pd.to_numeric(spd_df["recommended_speed_kmh"],errors="coerce")
                spd_c=spd_df["recommended_speed_kmh"].value_counts().sort_index().reset_index(); spd_c.columns=["Speed (km/h)","Count"]
                spd_fig=px.bar(spd_c,x="Speed (km/h)",y="Count",title="Speed Advisory Distribution",text="Count",color="Speed (km/h)",color_continuous_scale=["#ff3b3b","#38bdf8","#00e5a0"])
                spd_fig.update_layout(**PLOT_LAYOUT,coloraxis_showscale=False)
                st.plotly_chart(spd_fig,use_container_width=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Damage Hotspot Heatmap</div>', unsafe_allow_html=True)
        heatmap_rows=[]
        for highway_name, highway_data in NH_DATABASE.items():
            for seg in highway_data.get("segments", []):
                severity_score = seg.get("conf", 50) if seg.get("condition") == "Crack Detected" else max(10, 100 - seg.get("conf", 50))
                heatmap_rows.append({
                    "highway": highway_name,
                    "segment": seg.get("name", "Segment"),
                    "lat": seg.get("lat"),
                    "lon": seg.get("lon"),
                    "condition": seg.get("condition", "Unknown"),
                    "intensity": severity_score,
                })
        heatmap_df = pd.DataFrame(heatmap_rows)
        selected_heatmap_nh = st.selectbox("Heatmap Highway Filter", ["All Highways"] + sorted(heatmap_df["highway"].unique().tolist()), key="heatmap_highway_filter")
        if selected_heatmap_nh != "All Highways":
            heatmap_df = heatmap_df[heatmap_df["highway"] == selected_heatmap_nh]
        if not heatmap_df.empty:
            heatmap = px.density_mapbox(
                heatmap_df,
                lat="lat",
                lon="lon",
                z="intensity",
                radius=22,
                hover_name="segment",
                hover_data={"highway":True, "condition":True, "lat":False, "lon":False, "intensity":True},
                center={"lat": float(heatmap_df["lat"].mean()), "lon": float(heatmap_df["lon"].mean())},
                zoom=4,
                height=430,
                mapbox_style="carto-positron",
                title="Representative crack-risk concentration across monitored NH segments"
            )
            heatmap.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=50,b=0), font=dict(color="#94a3b8"))
            st.plotly_chart(heatmap, use_container_width=True)
            st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:0.62rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">Heatmap is based on the current NH scanner segment database and shows representative crack-risk intensity.</div>', unsafe_allow_html=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="sec-title">Highest Risk NH Segments</div>', unsafe_allow_html=True)
        segment_df = get_segment_health_dataframe()
        if not segment_df.empty:
            risky_segments = segment_df[segment_df["condition"] == "Crack Detected"].sort_values(["confidence", "speed_kmh"], ascending=[False, True]).head(12)
            st.dataframe(
                risky_segments[["highway", "segment", "terrain", "confidence", "speed_kmh", "risk", "damage_years"]].rename(
                    columns={
                        "highway": "Highway",
                        "segment": "Segment",
                        "terrain": "Terrain",
                        "confidence": "Confidence %",
                        "speed_kmh": "Recommended Speed",
                        "risk": "Risk Level",
                        "damage_years": "Est. Damage Time"
                    }
                ),
                use_container_width=True
            )
        if "timestamp" in logs_df.columns:
            logs_df["timestamp"]=pd.to_datetime(logs_df["timestamp"],errors="coerce")
            td=logs_df.dropna(subset=["timestamp"]).copy()
            if not td.empty:
                td["date"]=td["timestamp"].dt.date
                dc=td.groupby("date").size().reset_index(name="Count")
                lf=px.line(dc,x="date",y="Count",markers=True,title="Predictions Over Time")
                lf.update_traces(line_color="#38bdf8",marker_color="#38bdf8")
                lf.update_layout(**PLOT_LAYOUT,xaxis_title="Date",yaxis_title="Count")
                st.plotly_chart(lf,use_container_width=True)
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.dataframe(logs_df,use_container_width=True)
        csv=logs_df.to_csv(index=False).encode("utf-8")
        st.download_button("↓ Export CSV",data=csv,file_name="prediction_logs.csv",mime="text/csv",use_container_width=True)

# ══════════════════════════ CLOUD ══════════════════════════
elif st.session_state.page_mode == "cloud":
    st.markdown("""
    <div class="page-hero">
        <div class="overline">Infrastructure</div>
        <div class="hero-title">Cloud Integration</div>
        <div class="hero-sub">Architected for seamless transition to cloud-scale deployment and BI analytics.</div>
    </div>""", unsafe_allow_html=True)
    integrations=[
        ("Azure","Azure Blob Storage","Images and prediction logs can be pushed to Azure Blob containers for durable, scalable cloud storage across multiple field agents."),
        ("Azure","Azure ML Endpoints","The MobileNetV2 model can be deployed as a REST inference endpoint on Azure ML, enabling real-time prediction at scale."),
        ("IBM","IBM Cognos Analytics","Prediction logs including speed advisories and damage timelines can stream into IBM Cognos dashboards for national road health monitoring."),
        ("Google","Google Street View API","The NH Scanner pulls live road imagery at GPS coordinates via Street View API — just add an API key to go fully live."),
        ("OpenStreetMap","Overpass API","Road geometry and NH coordinates fetched from Overpass API (OSM) — completely free, no API key required."),
        ("GitHub","CI/CD via GitHub Actions","Automated retraining pipelines triggered on new dataset pushes, with model versioning and deployment gating."),
    ]
    c1,c2=st.columns(2,gap="large")
    for i,(provider,title,desc) in enumerate(integrations):
        col=c1 if i%2==0 else c2
        with col:
            st.markdown(f'<div class="card-accent"><div class="overline">{provider}</div><div class="sec-title" style="font-size:1rem;">{title}</div><div style="font-size:0.84rem;color:#64748b;line-height:1.7;">{desc}</div></div>', unsafe_allow_html=True)

# ══════════════════════════ ABOUT ══════════════════════════
elif st.session_state.page_mode == "about":
    st.markdown("""
    <div class="page-hero">
        <div class="overline">Project Documentation</div>
        <div class="hero-title">About RoadScan AI</div>
    </div>""", unsafe_allow_html=True)
    a1,a2=st.columns([1.1,1],gap="large")
    with a1:
        st.markdown("""
        <div class="card-accent">
            <div class="overline">Project</div>
            <div class="sec-title">AI-Based Road Damage Detection Using Deep Learning</div>
            <div style="font-size:0.88rem;color:#94a3b8;line-height:1.9;">A final-year engineering project automating road surface inspection using computer vision and transfer learning. Any road photograph can be analysed in under a second — classifying surfaces as cracked or safe, advising safe travel speeds, predicting structural lifespan, and providing GPS-based segment analysis for entire National Highways.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="card" style="margin-top:0.8rem;"><div class="overline">Technical Stack</div><div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;margin-top:0.5rem;">', unsafe_allow_html=True)
        for k,v in [("Model","MobileNetV2 + Fine-Tuning"),("Framework","TensorFlow / Keras"),("Frontend","Streamlit"),("Visualisation","Plotly + Mapbox"),("GPS Data","OpenStreetMap Overpass"),("Dataset","CrackForest (binary)"),("Image Size","224 × 224 px"),("Training","2-Phase + EarlyStopping")]:
            st.markdown(f'<div style="background:#0b1220;padding:0.6rem 0.8rem;border-radius:3px;"><div style="font-family:JetBrains Mono,monospace;font-size:0.6rem;color:#475569;letter-spacing:0.1em;text-transform:uppercase;">{k}</div><div style="font-size:0.82rem;color:#cbd5e1;margin-top:0.15rem;">{v}</div></div>', unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with a2:
        best_val_acc, best_val_auc = get_best_training_metrics()
        st.markdown(f"""
        <div class="card-accent" style="margin-bottom:0.8rem;">
            <div class="overline">Model Benchmarks</div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;">
                <div style="background:#0b1220;padding:0.9rem;border-radius:4px;">
                    <div class="nh-info-key">Best Validation Accuracy</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;color:#38bdf8;">{best_val_acc:.2f}%</div>
                </div>
                <div style="background:#0b1220;padding:0.9rem;border-radius:4px;">
                    <div class="nh-info-key">Best Validation AUC</div>
                    <div style="font-family:'Barlow Condensed',sans-serif;font-size:2rem;color:#38bdf8;">{best_val_auc:.2f}%</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="card"><div class="overline">Key Features</div>', unsafe_allow_html=True)
        for feat in ["Binary crack classification with confidence score","Invalid image rejection for selfies / non-road images","Camera input + file upload support","Safe travel speed recommendation (20–100 km/h)","Damage timeline + road lifespan estimation","Damage progression forecast chart","NH GPS Scanner — 50+ highways loaded","Dropdown + featured highway quick scan","Segment-by-segment Mapbox visualisation","Damage hotspot heatmap + risk table","Prediction history logging & CSV export","Azure / IBM Cognos ready architecture"]:
            st.markdown(f'<div style="display:flex;align-items:flex-start;gap:0.6rem;padding:0.5rem 0;border-bottom:1px solid #172033;"><span style="color:#38bdf8;margin-top:0.05rem;">›</span><span style="font-size:0.85rem;color:#94a3b8;">{feat}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card" style="margin-top:0.8rem;"><div class="overline">Future Scope</div><div style="font-size:0.84rem;color:#64748b;line-height:1.8;">→ Live Street View image sampling per GPS coordinate<br>→ Crack segmentation (pixel-level masks)<br>→ Multi-class severity (hairline / moderate / severe)<br>→ Azure ML cloud deployment<br>→ IBM Cognos national road health dashboards<br>→ Mobile app with on-device inference</div></div>', unsafe_allow_html=True)

st.markdown('<div class="footer">RoadScan AI · Final Year Project · MobileNetV2 + GPS Segment Analysis · Built with Streamlit</div>', unsafe_allow_html=True)