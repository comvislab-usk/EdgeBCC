import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
import time
import pandas as pd
import altair as alt
from io import BytesIO
import base64

# Load default image
default_image = Image.open("imagebcc.jpg")

# Model paths
MODEL_PATHS = {
    "EfficientNet-B0": "modeleffb0.onnx",
    "ResNet34": "modelr34.onnx"
}

# Normalization values
mean = np.array([0.7777, 0.7363, 0.6791], dtype=np.float32)
std = np.array([0.1700, 0.1890, 0.2412], dtype=np.float32)

# Preprocessing
def preprocess_image(image):
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    return np.expand_dims(image, axis=0)

# Prediction
def predict(image, session):
    image = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: image})
    probs = F.softmax(torch.tensor(outputs[0][0]), dim=0).numpy()
    return np.argmax(probs), np.max(probs)

# Load ONNX sessions
session_bcc = ort.InferenceSession(MODEL_PATHS["EfficientNet-B0"])
session_r34 = ort.InferenceSession(MODEL_PATHS["ResNet34"])
class_names = ["BCC", "Normal"]

# Sidebar Upload
with st.sidebar:
    st.header("\U0001F4E4 Upload Skin Image")
    uploaded_files = st.file_uploader(
        "Select one or more images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True, 
        key="uploader"
    )

# Title & Description
st.title("\U0001F4F7 Basal Cell Carcinoma Classification")
st.markdown("Detect **Basal Cell Carcinoma (BCC)** using EfficientNet-B0 and ResNet34.")

# Handle Single or Batch
if uploaded_files:
    if len(uploaded_files) == 1:
        image = Image.open(uploaded_files[0])
        img_col, res_col = st.columns(2)

        with img_col:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with res_col:
            with st.spinner("Running predictions..."):
                start_b = time.time()
                pred_b, conf_b = predict(image, session_bcc)
                dur_b = time.time() - start_b

                start_r = time.time()
                pred_r, conf_r = predict(image, session_r34)
                dur_r = time.time() - start_r

                preds = {
                    "EfficientNet-B0": (pred_b, conf_b, dur_b),
                    "ResNet34": (pred_r, conf_r, dur_r)
                }

                for model, (pred, conf, dur) in preds.items():
                    label = class_names[pred]
                    label_color = "#B00020" if label == "BCC" else "#118C4F"
                    icon = "⚠️" if label == "BCC" else "✅"

                    st.markdown(f"**{model}**")
                    st.markdown(f"""
                    <div style="background-color:#f9f9f9;border-radius:10px;padding:1rem;margin-bottom:1rem;">
                        <div style="font-size:1.5rem;font-weight:bold;color:{label_color};">
                            {icon} {label} ({conf * 100:.3f}%)
                        </div>
                        <div style="font-size:0.85rem;color:#444;">⏱️ {dur:.4f} sec</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        
        st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)    

        # Comparison chart for single image
        st.markdown("### 📊 Model Performance Comparison")

        comparison_single = {
            "Total Time (s)": [dur_b, dur_r],
            "Accuracy (%)": [conf_b * 100, conf_r * 100]
        }
        model_labels = ["EfficientNet-B0", "ResNet34"]

        for metric, values in comparison_single.items():
            st.markdown(f"**{metric}**")
            chart_df = pd.DataFrame({
                "Model": model_labels,
                metric: values
            })

            chart = alt.Chart(chart_df).mark_bar().encode(
                x=alt.X("Model", sort=None),
                y=alt.Y(metric, title=metric),
                color=alt.Color("Model", legend=None)
            ).properties(height=250)

            st.altair_chart(chart, use_container_width=True)

    else:
        with st.spinner("Running batch predictions..."):
            data = []
            total_time_effnet = total_time_resnet = 0
            total_acc_effnet = total_acc_resnet = 0

            for i, file in enumerate(uploaded_files):
                image = Image.open(file)

                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/png;base64,{img_str}" width="100"/>'

                start_b = time.time()
                pred_b, conf_b = predict(image, session_bcc)
                dur_b = time.time() - start_b

                start_r = time.time()
                pred_r, conf_r = predict(image, session_r34)
                dur_r = time.time() - start_r

                total_time_effnet += dur_b
                total_time_resnet += dur_r
                total_acc_effnet += conf_b
                total_acc_resnet += conf_r

                data.append({
                    "#": i + 1,
                    "Image": img_html,
                    "EfficientNet-B0 Label": class_names[pred_b],
                    "EfficientNet-B0 Confidence (%)": f"{conf_b * 100:.3f}",
                    "ResNet34 Label": class_names[pred_r],
                    "ResNet34 Confidence (%)": f"{conf_r * 100:.3f}"
                })

            df = pd.DataFrame(data)
            st.write("### 🔍 Prediction Results")
            st.write(
                df.to_html(escape=False, index=False), 
                unsafe_allow_html=True
            )

            
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)

            avg_time_effnet = total_time_effnet / len(uploaded_files)
            avg_time_resnet = total_time_resnet / len(uploaded_files)
            avg_acc_effnet = total_acc_effnet / len(uploaded_files)
            avg_acc_resnet = total_acc_resnet / len(uploaded_files)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("EfficientNet-B0 Summary")
                st.markdown(f"**Total Time:** {total_time_effnet:.4f} sec")
                st.markdown(f"**Average Time:** {avg_time_effnet:.4f} sec")
            with col2:
                st.subheader("ResNet34 Summary")
                st.markdown(f"**Total Time:** {total_time_resnet:.4f} sec")
                st.markdown(f"**Average Time:** {avg_time_resnet:.4f} sec")

            
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            
            st.markdown("### 📊 Model Performance Comparison")

            comparisons = {
                "Total Time (s)": [total_time_effnet, total_time_resnet],
                "Average Time (s)": [avg_time_effnet, avg_time_resnet],
                "Average Accuracy (%)": [avg_acc_effnet * 100, avg_acc_resnet * 100]
            }
            model_labels = ["EfficientNet-B0", "ResNet34"]

            for metric, values in comparisons.items():
                st.markdown(f"**{metric}**")
                chart_df = pd.DataFrame({
                    "Model": model_labels,
                    metric: values
                })

                chart = alt.Chart(chart_df).mark_bar().encode(
                    x=alt.X("Model", sort=None),
                    y=alt.Y(metric, title=metric),
                    color=alt.Color("Model", legend=None)
                ).properties(height=250)

                st.altair_chart(chart, use_container_width=True)

else:
    image = default_image
    img_col, res_col = st.columns(2)
    with img_col:
        st.image(image, caption="Example BCC Image", use_column_width=True)
    with res_col:
        sample_preds = {
            "EfficientNet-B0": ("BCC", 0.9989, 0.0291),
            "ResNet34": ("BCC", 0.8935, 0.0735)
        }
        for model, (label, conf, dur) in sample_preds.items():
            label_color = "#B00020" if label == "BCC" else "#118C4F"
            icon = "⚠️" if label == "BCC" else "✅"
            st.markdown(f"**{model}**")
            st.markdown(f"""
            <div style="background-color:#f9f9f9;border-radius:10px;padding:1rem;margin-bottom:1rem;">
                <div style="font-size:1.5rem;font-weight:bold;color:{label_color};">
                    {icon} {label} ({conf * 100:.3f}%)
                </div>
                <div style="font-size:0.85rem;color:#444;">⏱️ {dur:.4f} sec</div>
            </div>
            """, unsafe_allow_html=True)
