#### "Performing supervised learning techniques to find out the most effective in accurately classifying breast cancer as malignant or benign"
Credits:
Ahmed Abdelrahman | Andreza dos Santos | Muhammad Hassan | Marivic Tanguin

## Comparing Supervised Machine Learning Algorithms for Breast Cancer Diagnosis

[Data Extraction](https://github.com/Ahmadhha/UTOR_Project4_Team5/blob/main/Data%20Extraction.ipynb)

[Data Visualization](https://public.tableau.com/app/profile/andreza.dos.santos6466/viz/BreastCancer-MeanProject/Story1?publish=yes)

[Machine Learning Analysis](https://github.com/Ahmadhha/UTOR_Project4_Team5/blob/main/Machine%20Learning.ipynb)

[Model Performance Metrics Comparison](https://public.tableau.com/app/profile/mtanguin/viz/ModelPerformanceMetricsComparison/Dashboard1#1)

[Data Source](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?select=data.csv)

[Presentation](https://docs.google.com/presentation/d/1HRPYjpFacVPwD4IIRp076iLtkR5-W-82ltuACiWLI4s/edit?usp=sharing)

## Background
Breast cancer is the most commonly diagnosed cancer among women globally, with an estimated 2.3 million new cases and 685,000 deaths in 2020. Early detection and diagnosis of breast cancer are critical for improving diagnostic outcomes and reducing mortality rates. According to the journal, The Breast, by 2040 the breast cancer burden will increase to more than 3 million new cases per year (an increase of 40%) and more than 1 million deaths per year (an increase of 50%).
Medical imaging techniques such as mammography and ultrasound, as well as histopathological examination of tissue samples obtained through biopsy, are commonly used to diagnose breast cancer.

Machine learning algorithms have shown promise in improving the accuracy and efficiency of breast cancer diagnosis. Several models, including logistic regression, decision trees, random forests, support vector machines, neural networks, and K-Nearest Neighbors (KNN), have been applied to breast cancer diagnosis datasets to predict the presence or absence of malignancy. These models used the ten real-valued features that are computed for each cell nucleus: such as radius (mean of distances from center to points on the perimeter), 
texture (standard deviation of gray-scale values), perimeter,area, smoothness (local variation in radius lengths), compactness (perimeter^2 / area - 1.0), concavity (severity of concave portions of the contour), concave points (number of concave portions of the contour), symmetry and fractal dimension ("coastline approximation" - 1).

Despite the promising results, the development of accurate and reliable machine learning models for breast cancer diagnosis remains a challenging task. Issues such as imbalanced datasets, feature selection, overfitting, and interpretability must be carefully addressed to ensure that the models are effective and clinically relevant. Also, the performance of the models may vary depending on the type and quality of the data used for training and testing.

Therefore, the goal of this study is to develop and evaluate various machine learning models for breast cancer diagnosis using a large, publicly available dataset. Specifically, we aim to compare the performance of logistic regression, decision trees, random forests, support vector machines, neural networks, and KNN on the dataset and determine which model is most effective for predicting breast cancer diagnosis. Also, we will explore various feature selection and performance evaluation techniques to optimize the models and provide insights into the relevant features that contribute to accurate diagnosis. The results of this study could have significant implications for improving the accuracy and efficiency of breast cancer diagnosis and eventually improving diagnostic outcomes.

## Research Question
Among the various supervised learning algorithms to be evaluated in this study using the Breast Cancer Wisconsin dataset, which one performed the most effective in accurately classifying breast cancer as malignant or benign?

## Methods

### Machine Learning Techniques
In this project, we aim to develop a predictive model to diagnose breast masses as either benign or malignant using machine learning algorithms. The dataset contains features related to breast mass characteristics, and we will utilize supervised learning techniques to train and evaluate multiple models for accurate classification.

### Problem Identification
The problem at hand involves binary classification, where the goal is to distinguish between benign and malignant breast masses. By analyzing the dataset, we can build a predictive model that can accurately classify future breast masses based on their characteristics.

### Supervised Learning Algorithms
We will explore six different supervised learning algorithms to determine the optimal approach for this dataset. These algorithms include:

1. Logistic Regression: This algorithm models the relationship between the features and the probability of a breast mass being benign or malignant.

2. Decision Tree: This algorithm uses a hierarchical structure of decisions based on features to classify breast masses.

3. Random Forest: A collection of decision trees that work together to classify breast masses based on different subsets of features.

4. Support Vector Machine (SVM): This algorithm finds a hyperplane that maximally separates the two classes (benign and malignant) in a high-dimensional feature space.

5. Neural Network: A deep learning algorithm that uses an interconnected network of nodes (neurons) to classify breast masses based on learned patterns.

6. K-Nearest Neighbors (KNN): This algorithm classifies breast masses by comparing their features to the k nearest neighbors in the training data.

### Model Evaluation and Selection
Each algorithm will be trained and evaluated using appropriate metrics such as accuracy, precision, recall, and F1-score. The performance of the models will be compared to identify the most optimal algorithm for this particular dataset.

The selected algorithm will be further fine-tuned using techniques such as hyperparameter optimization to maximize its performance. The final model will be saved and deployed for real-time breast mass classification.

The project will be documented in the readme.md file, providing details about the dataset, objectives, methods, and key findings from the machine learning analysis.


## Results



![Performance Model Metrics Comparison](https://user-images.githubusercontent.com/114210481/232283305-462a7c37-43f0-4104-ad06-e3af3e3e8778.jpg)


## Deployment

See this link:

[Project 4: Breast Cancer Diagnosis Deployment](https://github.com/MTanguin/Project_4_Deployment_Test)


## Limitation
This research project solely focused on analyzing the Breast Cancer Wisconsin (Diagnostic) dataset, which contains information on ten specific features of cell nuclei extracted from fine needle aspirate images of breast masses. It is important to note that the dataset's limited scope excludes other possible risk factors and clinical data that could impact breast cancer diagnosis. Furthermore, this study solely evaluate the performance of the selected supervised learning algorithms and may not consider alternative algorithms or techniques that may prove more effective for diagnosing breast cancer.


## Conclusion
The evaluation results indicate that all of the models exhibit high accuracy rates, with Random Forest achieving the highest accuracy at 96.49%, followed by Logistic Regression and SVM with 95.61%. Furthermore, the precision scores for all models are also high, with Random Forest attaining the highest precision score of 96.73%, closely followed by SVM and Logistic Regression with 96.05%.

In terms of recall, Random Forest attained the highest score at 95.81%, while SVM and Logistic Regression recorded 94.64%. The F1-score, which measures the trade-off between precision and recall, is highest for Random Forest at 96.23%, with the Optimized Neural Network following closely at 95.39%.

Overall, the Random Forest model performed the best among the models with high scores across all metrics. However, the differences in performance between the models were relatively small, and other factors such as computational resources and interpretability may influence the choice of model to use.

## Recommendation
Based on the evaluation results, the Random Forest model appears to be the best-performing model among the options provided, exhibiting high accuracy, precision, recall, and F1-score. Therefore, it is recommended that the Random Forest model be considered for deployment in the relevant application.

However, it is important to note that the choice of model to use ultimately depends on various factors such as the specific requirements of the problem, available computational resources, interpretability needs, and other relevant considerations like datasets size. The choice of machine learning algorithm can also impact how well the model performs on larger datasets. For example, some algorithms such as decision trees and linear regression may not scale well to very large datasets, while others such as random forests and neural networks may be better suited for larger datasets.

Therefore, it is important to carefully consider all relevant factors when making a final decision on which model to use.

### Disclaimer
The Breast Cancer Wisconsin (Diagnostic) dataset is made available for academic purposes only. The dataset contains information on characteristics of cell nuclei present in fine needle aspirate (FNA) images of breast masses. The dataset is intended to be used for research and educational purposes only and not for gaining commercial benefits. The dataset does not contain any sensitive personal information of individuals. The dataset is provided as is, without any warranty, express or implied, including without limitation, warranties of merchantability or fitness for a particular purpose. Users are responsible for ensuring that their use of the dataset complies with all applicable laws and regulations. The creators and distributors of the dataset are not liable for any damages arising from the use of the dataset.



Sources: 

[WHO](https://www.iarc.who.int/news-events/current-and-future-burden-of-breast-cancer-global-statistics-for-2020-and-2040/)

[www.cancer.gov](https://www.cancer.gov/types/breast/patient/breast-screening-pdq)

[kaggle.com](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
