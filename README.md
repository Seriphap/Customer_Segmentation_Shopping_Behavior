# 👨‍👨‍👦‍👦 Customer Segmentation on Shopping Behavior

## 🏷️ Topic

This project focuses on **customer segmentation using shopping behavior data**. The main goal is to help businesses and marketers identify distinct groups of customers based on their purchasing patterns and demographic features, enabling targeted marketing strategies and improved customer understanding.

---

## 🚀 Streamlit App
https://customersegmentationshoppingbehavior-jbxpkequvka6h9kmxrj9gt.streamlit.app/
An interactive **Streamlit web application** is provided, allowing users to:
- Upload their own shopping datasets (or use the provided example)
- Explore the raw data
- Apply clustering algorithms (K-medoids/PAM with weighted Hamming distance)
- Visualize results with radar and sunburst charts
- Experiment with segmentation by adjusting the importance (weight) of shopping behavior vs. demographic attributes

---

## 🗂️ App Structure

```
.
├── main.py                  # Streamlit app entry point
├── modules/
│   ├── dataset.py           # Dataset viewer module
│   └── Clustering.py        # Main clustering and visualization logic
├── data/
│   └── 8_Shopping.csv       # Example shopping dataset
```

- `main.py`: Handles app layout, sidebar, file upload, and navigation between "Dataset" and "Clustering" sections.
- `modules/dataset.py`: Displays the uploaded or default dataset.
- `modules/Clustering.py`: Runs the full clustering pipeline and visualizations.

---

## 🐍 Python Libraries Used

- `streamlit` - for interactive web app
- `pandas`, `numpy` - data manipulation
- `plotly` - interactive charts (radar, sunburst)
- `pyclustering` (kmedoids) - clustering algorithm
- `random` - random state for reproducibility

---

## 📊 Visualizations

- **Radar Chart**: Shows average purchasing behavior in each cluster across product categories.
- **Sunburst Chart**: Visualizes the demographic composition (gender, age, marital status, children, working status) within each cluster.
- **Silhouette Score Plot**: Helps determine the optimal number of clusters by plotting silhouette scores for k=2 to k=10.

---

## 🧠 Model & Clustering Methodology

- **Feature Groups:**  
  - *Basket features:* Customer Behavior is binary purchase indicators: 0/1 (e.g., Readymade, Frozen foods, Alcohol, etc.)
  - *Demographic features:* Gender, Age, Marital Status, Children, Working status

- **Encoding:**  
  - Categorical demographic features are encoded numerically.

- **Weighted Hamming Distance:**  
  - Users can adjust the weight between basket vs. demographic features to emphasize different aspects of segmentation.

- **Clustering Algorithm:**  
  - K-medoids (PAM algorithm) is used with the custom weighted Hamming distance.
  - The number of clusters (k) is user-selectable; silhouette scores are used as a guide.

---

## 🔍 Insights & What You Can Discover

By using this app, you can:
- **Identify distinct customer segments** based on both what people buy and who they are.
- **Understand the purchasing patterns** of each segment via radar plots (e.g., one group buys lots of alcohol and snacks, another focuses on fresh foods, etc.).
- **Explore the demographic makeup** of each cluster with sunburst charts (e.g., Cluster A is mostly young, single, working men; Cluster B is married women with children, etc.).
- **Test hypotheses interactively** by adjusting weights: see how segmentation changes when you focus more on basket vs. demographic features.
- **Export and further analyze** the clustered data for targeted marketing.

---

## 📥 How to Use

1. **Launch the app locally**:  
   ```
   pip install -r requirements.txt
   streamlit run main.py
   ```

2. **Upload your CSV dataset** (or use the default example).
3. **Navigate** between "Dataset" and "Clustering" in the sidebar.
4. **Adjust weights** and number of clusters (k) as needed.
5. **Interpret the visualizations** to generate actionable insights for segmentation.

---

## 💡 Example Use Cases

- Marketing teams identifying high-value customer groups
- Retailers tailoring promotions to specific segments
- Product managers understanding customer diversity

---

## 📝 Note

- Please ensure your uploaded dataset matches the structure of the example dataset for best results.
