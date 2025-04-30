# Intelligent Parking Guidance System – for Campus & Public Garage Navigation

## 🛠️ Problem Being Solved
Drivers frequently waste time circling for parking in large lots. Most smart parking systems only show the number of available spots, without real-time routing to specific vacancies. This causes delays, emissions, and frustration.

## ✅ Solution Overview
A real-time intelligent parking guidance system using a monocular camera to detect spot occupancy and navigate users to the nearest available space using a dynamic web interface.

## 🧠 Technical Details
- **YOLOv5x6** model for vehicle detection; IoU-based thresholding for occupancy
- **Edge Processing:** NVIDIA Jetson AGX Orin
- **GPS Routing:** Google Maps + Distance Matrix API
- **Web Interface:** Built in React, shows spot availability and directions
- **Backend:** MongoDB, AWS Lambda, GStreamer for streaming

## 📊 Key Results
- <4s latency end-to-end
- 94%+ detection accuracy in all lighting/weather
- Real-time, user-specific GPS routing to open spots
- Scalable and cost-effective for smart campus/garage deployment

## ▶️ Demo Video
[Watch on YouTube](https://youtu.be/_Ke6rJwfDII)

