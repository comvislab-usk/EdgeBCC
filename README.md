<div align="center">

[![Paper](https://img.shields.io/badge/paper-IEEE%20ICONS--IoT%202025-blue.svg)](https://ieeexplore.ieee.org/document/11211337)

# Integration of Edge AI and Streamlit Web Application for Basal Cell Carcinoma Skin Cancer Classification

Wan Hayatun Nisa (Universitas Syiah Kuala), [Kahlil Muchtar (Universitas Syiah Kuala, COMVISLAB USK)](https://comvis.mystrikingly.com/), Yudha Nurdin (Universitas Syiah Kuala), Maya Fitria (Universitas Syiah Kuala), Ahmadiar Ahmadiar (Universitas Syiah Kula), and Novi Maulina (Universitas Syiah Kuala)<br>

</div>

---

<div align="justify">

> **Abstract:** _Basal Cell Carcinoma (BCC) skin cancer is one of the common types of skin cancer and is often not identified in the early stages. BCC can cause further tissue damage if not treated promptly. Early diagnosis is essential to prevent the spread of cancer and increase the chances of more optimal treatment. The purpose of this study is to develop a BCC classification system using the Convolutional Neural Network (CNN) algorithm to provide classification results. In addition, this study incorporates Edge AI technology, utilizing Streamlit as the interface to facilitate user interaction with the skin cancer classification system. With Edge AI, data processing is done locally on the device, which allows for faster classification and reduces dependence on internet connections. The test results show that the developed model can effectively distinguish between normal skin and BCC. The models were evaluated based on accuracy, recall, specificity, F -score, and inference speed. The accuracy for the EfficientNet-B0 architecture is 98.9% and for the ResNet-34 architecture is 100%, respectively._

</div><br>

<p align="center">
  <img style="width: 70%" src="Media/Fig 1.jpg">
</p>

<small>_Fig. 1. Proposed Research Flowchart._</small>

---

## 📊 Data 

Please download the dataset used in this study from the Google Drive links below. The dataset is provided in both Normal and bcc versions, along with predefined training, validation, and testing splits.

  [🔗 Google Drive Link](https://drive.google.com/drive/folders/1BNc1OllmYZQhY79XaSEAdDY-FYBNShLf?usp=sharing)


---

## ⚙️ Hyperparameters

<p align="center"><b>Table 1. Hyperparameter Settings For Model Training</b></p>
<div align="center">
  <small>
    <table >
        <tr style="background-color:#b3b3b3; text-align:center;">
            <th>Parameter</th>
            <th>EfficientNet-B0</th>
            <th>ResNet-34</th>
        </tr>
        <tr>
            <td>
            Epoch
            </td>
            <td>50</td>
            <td>50</td>
        </tr>
        <tr>
            <td>Batch Size</td>
            <td>2</td>
            <td>2</td>
        </tr>
        <tr>
            <td>Loss Function </td>
            <td>Cross Entropy</td>
            <td>Cross Entropy</td>
        </tr>
        <tr>
            <td>Optimizer</td>
            <td>SGD</td>
            <td>SGD</td>
        </tr>
        <tr>
            <td >Learning Rate</td>
            <td>0.01</td>
            <td>0.01</td>
        </tr>
        <tr>
            <td >Momentum</td>
            <td>0.9</td>
            <td>0.9</td>
        </tr>
        </tr>
    </table>
  </small>
</div>

---

### 🚀 Run Streamlit Application

The Streamlit application has been provided in this repository. Follow the steps below to run the real-time defect detection interface.

- **Install Dependencies**

```python
pip install streamlit onnxruntime numpy opencv-python torch pillow pandas altair
```

- **Run Streamlit App**

```python
stramlit run app.py
```

<div style="margin-left: 20px;">
Make sure the trained model files is located in the correct directory.
</div>

---

## 📈 Results

<p align="center"><b>Table 1. Evaluation Metric Results</b></p>
<div align="center">
  <small>
    <table >
        <tr style="background-color:#b3b3b3; text-align:center;">
            <th>Metric</th>
            <th>EfficientNet-B0</th>
            <th>ResNet-34</th>
        </tr>
        <tr style="text-align:center;" >
            <td>Training Time</td>
            <td>147m 17s</td>
            <td>89m 13s</td>
        </tr>
        <tr style="text-align:center;" >
            <td>
            Accuracy
            </td>
            <td>98.9%</td>
            <td>100%</td>
        </tr>
        <tr style="text-align:center;">
            <td >Recall</td>
            <td>100%</td>
            <td>100%</td>
        </tr>
        <tr style="text-align:center;">
            <td >Specificity</td>
            <td>97.8%</td>
            <td>100%</td>
        </tr>
        <tr style="text-align:center;">
            <td>F1-Score</td>
            <td>98.9%</td>
            <td>100%</td>
        </tr>
    </table>
  </small>
</div>
<br>

<p align="center">
  <img style="width: 60%" src="Media/Fig 2.jpg">
</p>

<small>_Fig. 2. Bar chart comparison of Evaluation metrics(accuracy, recall, specificity, and F1-score)._</small>
<br>

---

## 🎨 Qualitative Results

<table align="center">
  <tr>
    <td align="center">
      <img src="Media/Fig 3.jpg" width="90%"><br>
    </td>
    <td align="center">
      <img src="Media/Fig 4.jpg" width="90%"><br>
    </td>
  </tr>
</table>

<small>_Fig. 3.   Interface implemented on the Edge-AI Jetson Orin Device_</small>

---

## 📝 Citation

Please consider citing our paper in your publications if the project helps your research.

```
@inproceedings{WanHayatunNisa2025edgeai,
  title={Integration of Edge AI and Streamlit Web Application for Basal Cell Carcinoma Skin Cancer Classification},
  author={Wan Hayatun Nisa, Kahlil Muchtar, Yudha Nurdin, Maya Fitria, Ahmadiar, Novi Maulinan},
  booktitle={2025 IEEE International Conference on Networking, Intelligent Systems, and IoT (ICONS-IoT)},
  year={2025},
  doi={10.1109/ICONS-IoT65216.2025.11211337}
}
```