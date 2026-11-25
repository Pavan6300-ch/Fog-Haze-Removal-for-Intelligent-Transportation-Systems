# Fog and Haze Removal for Intelligent Transportation Systems

## 1. Project Overview

Fog and haze reduce visibility in outdoor environments, making it difficult for computer vision systems to detect vehicles accurately. This project aims to improve visibility in hazy images using image dehazing techniques and evaluate the effect of dehazing on vehicle detection.

The project combines the **Dark Channel Prior (DCP)** algorithm for haze removal with a pretrained **YOLOv8 Nano** model for vehicle detection.

## 2. Objectives

- Remove fog and haze from outdoor images.
- Improve image visibility and structural quality.
- Apply YOLOv8 Nano for vehicle detection.
- Compare vehicle detection outputs before and after dehazing.
- Evaluate image quality and computational efficiency.

## 3. Technologies Used

- Python
- OpenCV
- NumPy
- PyTorch
- YOLOv8 Nano
- Scikit-image
- Google Colab

## 4. Proposed Methodology

The project follows these main stages:

1. **Input Image:** Load a hazy outdoor image.
2. **Dark Channel Prior:** Estimate the dark channel of the input image.
3. **Atmospheric Light Estimation:** Estimate atmospheric light using an improved selection strategy.
4. **Transmission Estimation:** Calculate the transmission map.
5. **Guided Filtering:** Refine the transmission map while preserving image edges.
6. **Scene Radiance Recovery:** Recover the dehazed image.
7. **Vehicle Detection:** Apply YOLOv8 Nano to both hazy and dehazed images.
8. **Performance Evaluation:** Compare image quality, detection outputs, and processing time.

## 5. Dehazing Configuration

| Parameter | Value |
|---|---:|
| Algorithm | Dark Channel Prior |
| Atmospheric Light Estimation | V2 |
| Omega | 0.65 |
| Dark Channel Window Size | 15 |
| Transmission Lower Bound | 0.20 |
| Guided Filter Radius | 80 |
| Guided Filter Epsilon | 0.04 |

**Note:** The dehazing algorithm is a traditional image-processing approach, not a trained deep learning model.

## 6. Vehicle Detection

A pretrained YOLOv8 Nano model is used to detect vehicles in hazy and dehazed images.

The evaluated vehicle categories are:

- Cars
- Motorcycles
- Buses
- Trucks

Detection outputs are compared using the number of detected vehicles and their confidence scores.

## 7. Experimental Results

### Image Quality Evaluation

Evaluation was performed on 400 images.

| Metric | Input Hazy Images | Dehazed Images |
|---|---:|---:|
| Average PSNR | 15.9429 dB | 23.1364 dB |
| Average SSIM | 0.8142 | 0.9273 |

Additional observations:

- PSNR improved in 384 out of 400 images.
- SSIM improved in 397 out of 400 images.
- Average PSNR improvement: 7.1935 dB.
- Average SSIM improvement: 0.1131.

### Vehicle Detection Results

| Metric | Hazy Images | Dehazed Images |
|---|---:|---:|
| Total Vehicle Detections | 1,251 | 1,361 |
| Average Detections per Image | 3.1275 | 3.4025 |

The dehazed images produced approximately 8.8% more vehicle detections.

### Computational Efficiency

Benchmarking was performed using Google Colab CPU execution.

| Process | Average Time per Image | FPS |
|---|---:|---:|
| Dehazing | 0.081824 s | 12.22 |
| YOLOv8 Detection | 0.285846 s | 3.50 |
| Combined Pipeline (Estimated) | 0.367669 s | 2.72 |

The combined pipeline speed is estimated from the separately measured average processing times.

## 8. Dataset

The project uses outdoor hazy images and their corresponding clear reference images.

The dataset contains:
- 500 hazy images.
- 492 clear reference images.

The dataset is used for image dehazing and image quality evaluation.

## 9. Project Structure

```text
Fog-Haze-Removal-for-Intelligent-Transportation-Systems/
│
├── src/
│   └── dehazing.py
│
├── config/
│   └── Dehazing_Model_Configuration.json
│
├── models/
│   └── YOLOv8_Vehicle_Detection.pt
│
├── results/
│   ├── Computational_Efficiency.csv
│   ├── YOLO_Detection_Results.csv
│   ├── Vehicle_Confidence_Comparison.csv
│   ├── Vehicle_Class_Comparison.csv
│   ├── Final_400_Image_Metrics.csv
│   ├── Detection_Comparison.csv
│   ├── Final_Project_Summary.txt
│   └── Computational_Efficiency_Summary.txt
│
└── requirements.txt
```

## 10. Limitations

- The dehazing algorithm may not perform equally well under all lighting and atmospheric conditions.
- YOLOv8 detection outputs are model predictions, not verified ground-truth detections.
- The dataset does not provide vehicle bounding-box annotations for calculating detection precision, recall, or mAP.
- The measured computational performance depends on the hardware and execution environment.
- The combined pipeline benchmark is an estimate rather than a direct end-to-end measurement.

## 11. Conclusion

This project demonstrates how traditional image dehazing techniques can be integrated with deep learning-based object detection for intelligent transportation applications.

The experimental results show improvements in image quality metrics and an increase in the number of vehicle detections after dehazing.

The project provides a foundation for exploring visibility enhancement and vehicle perception in adverse weather conditions.

## 12. Future Scope

- Explore advanced deep learning-based dehazing methods.
- Evaluate vehicle detection using annotated datasets.
- Test the system under different weather conditions.
- Optimize the pipeline for faster inference.
- Explore deployment for intelligent transportation applications.
