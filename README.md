# pothole-detection
Training a computer vision task to detect and segment potholes in images. This model was trained on the [pothole-600](https://sites.google.com/view/pothole-600/dataset) dataset.

# Datasets Used
- [Nienaber 1](https://www.kaggle.com/datasets/felipemuller5/nienaber-potholes-1-simplex)
- [Nienaber 2](https://www.kaggle.com/datasets/felipemuller5/nienaber-potholes-2-complex)
- [Road Damage Detection](https://github.com/sekilab/RoadDamageDetector/)
- [Brazilian NDTI](https://github.com/biankatpas/Cracks-and-Potholes-in-Road-Images-Dataset)

# Setup

Create and Activate Virtual Environment

```	
python3 -m venv venv --system-site-packages
source venv/bin/activate
```

Install required packages

```
pip install -r requirements.txt
```

# Model Testing

Model trained on an augmented dataset containing 6600 images using a Feature Pyramid Network. This model was able to achive an IOU of 0.86

![Image1](./images/output1.png)
![Image2](./images/output2.png)
![Image3](./images/output3.png)
![Image4](./images/output4.png)
