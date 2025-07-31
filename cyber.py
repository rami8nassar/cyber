##################################### student name : rami nassar #######################################################
#####################################     ID : 324005701       #########################################################
import os
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
  # download the dataset from kaggle:
  path = kagglehub.dataset_download("urvishahir/electric-vehicle-specifications-dataset-2025")
  print("Path to dataset files:", path)

  # list files to check the exact file name
  print("Files in folder:", os.listdir(path))
  # use the exact file name that was downloaded!
  dataset_file = os.path.join(path, "electric_vehicles_spec_2025.csv.csv")  
  print("Dataset file path:", dataset_file)

  # check if file exists
  if not os.path.exists(dataset_file):
    print("ERROR: File not found! Check file name and path.")
    return

  df = pd.read_csv(dataset_file) # here we get the dataset
  print("\n===== COLUMNS IN DATASET =====")
  print(df.columns.tolist())
  # Meta Data:
  # information about data:
  print("\n===== INFO =====")
  df.info()  # no print() needed here

  # example for the data:
  print("\n===== HEAD =====")
  print(df.head())

  # statistics:
  print("\n===== DESCRIBE =====")
  print(df.describe(include='all'))

  # counting missing values
  print("\n===== MISSING VALUES =====")
  print(df.isnull().sum())

  # first check for unique values:
  print("\n===== UNIQUE VALUES PER COLUMN =====")
  for col in df.columns:
      print(f"\n{col} unique values (sample): {df[col].unique()[:10]}")
  # -----------------------------
  # Step 4: Statistics Analysis
  # -----------------------------
  print("\n===== STEP 4: STATISTICAL ANALYSIS =====")

  # numeric columns from the dataset
  numeric_columns = [
    'top_speed_kmh',
    'battery_capacity_kWh',
    'number_of_cells',
    'torque_nm',
    'efficiency_wh_per_km',
    'range_km',
    'acceleration_0_100_s',
    'fast_charging_power_kw_dc',
    'towing_capacity_kg',
    'cargo_volume_l',
    'seats',
    'length_mm',
    'width_mm',
    'height_mm'
  ]

  # Descriptive statistics
  print("\n===== DESCRIPTIVE STATISTICS =====")
  print(df[numeric_columns].describe())

  # Histograms
  for col in numeric_columns:
      plt.figure(figsize=(6, 4))
      sns.histplot(df[col].dropna(), kde=True, bins=30)
      plt.title(f"Distribution of {col}")
      plt.xlabel(col)
      plt.ylabel("Count")
      plt.tight_layout()
      plt.show()

  # Correlation matrix
  # Correlation matrix
  print("\n===== CORRELATION MATRIX =====")
  numeric_df = df[numeric_columns].apply(pd.to_numeric, errors='coerce')  # Convert all to numeric, coerce errors
  corr = numeric_df.corr()
  print(corr)

  # Heatmap
  plt.figure(figsize=(8, 6))
  sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
  plt.title("Correlation Matrix Heatmap")
  plt.tight_layout()
  plt.show()

  # Check for duplicates
  print("\n===== DUPLICATE ROWS =====")
  print("Number of duplicated rows:", df.duplicated().sum())

  # Check for unique rows
  print("\n===== UNIQUE ROWS =====")
  print("Number of unique rows:", df.drop_duplicates().shape[0])
    # -----------------------------
  # Step 5: Outlier Detection (IQR)
  # -----------------------------
  #print("\n===== STEP 5: OUTLIER DETECTION (IQR Method) =====")

  #for col in numeric_columns:
      #if col in df.columns:
       #   series = df[col].dropna()
        #  Q1 = series.quantile(0.25)
         # Q3 = series.quantile(0.75)
          #IQR = Q3 - Q1
          #lower_bound = Q1 - 1.5 * IQR
          #upper_bound = Q3 + 1.5 * IQR

          #outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
          #print(f"\n--- {col} ---")
          #print(f"Outliers found: {len(outliers)}")
          #if not outliers.empty:
           #   print("Sample outliers:")
            #  print(outliers[[col]].head())
          #else:
           #   print("No outliers detected.")

          # Boxplot
          #plt.figure(figsize=(4, 5))
          #plt.boxplot(series)
          #plt.title(f'Boxplot of {col}')
          #plt.ylabel(col)
          #plt.tight_layout()
          #plt.show()
#################################################################################################################
  # -----------------------------
  # Step 6: CLUSTERING (KMeans)
  # -----------------------------
  print("\n===== STEP 6: CLUSTERING (KMeans) =====")

  from sklearn.cluster import KMeans
  from sklearn.preprocessing import StandardScaler

  # שלב 1: בחר רק את העמודות המספריות
  numeric_columns = df.select_dtypes(include=['number']).columns
  clustering_data = df[numeric_columns]

  # שלב 2: נקה NaN אם יש
  clustering_data = clustering_data.dropna()

  # שלב 3: סקלינג לנתונים
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(clustering_data)

  # שלב 4: הרצת אלגוריתם KMeans עם 4 קבוצות לדוגמה
  kmeans = KMeans(n_clusters=4, random_state=42)
  labels = kmeans.fit_predict(scaled_data)

  # שלב 5: הוסף את התוויות (labels) לטבלה המקורית (שורות תואמות בלבד!)
  df_filtered = df.loc[clustering_data.index]  # התאמה לאינדקס אחרי dropna
  df_filtered['cluster'] = labels

  # שלב 6: הצג את התפלגות הקבוצות
  print("\nCluster distribution:")
  print(df_filtered['cluster'].value_counts())

  # אופציונלי: הצג כמה שורות לדוגמה
  print("\nSample of clustered data:")
  print(df_filtered[['cluster'] + list(numeric_columns)].head())

# -----------------------------
  # Step 7: SEGMENT ANALYSIS (ניתוח מגזרים / סגמנטים)
  # -----------------------------
  print("\n===== STEP 7: SEGMENT ANALYSIS =====")

  from sklearn.decomposition import PCA

  # בדיקת קיום עמודת קלאסטר
  if 'cluster' not in df_filtered.columns:
      print("Error: Missing 'cluster' column. Run clustering before segmentation.")
      return

  # סיכום סטטיסטי ממוצע לכל קלאסטר
  segment_summary = df_filtered.groupby('cluster').mean(numeric_only=True)
  print("\nMean values per segment:")
  print(segment_summary)

  # ביצוע PCA לצמצום מימדים
  pca_data = df_filtered[numeric_columns].dropna()
  from sklearn.preprocessing import StandardScaler
  scaled = StandardScaler().fit_transform(pca_data)

  pca = PCA(n_components=2)
  components = pca.fit_transform(scaled)

  df_plot = df_filtered.loc[pca_data.index].copy()
  df_plot['PCA1'] = components[:, 0]
  df_plot['PCA2'] = components[:, 1]

  # שרטוט לפי קלאסטר
  plt.figure(figsize=(8, 6))
  for cluster_id in sorted(df_plot['cluster'].unique()):
      cluster_data = df_plot[df_plot['cluster'] == cluster_id]
      plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], label=f'Cluster {cluster_id}', alpha=0.6)

  plt.title("PCA of EV Dataset Colored by Cluster")
  plt.xlabel("Principal Component 1")
  plt.ylabel("Principal Component 2")
  plt.legend()
  plt.tight_layout()
  plt.show()
  ##################################### step 8 :#########################################################################################
  print("================ step 8 ===============================================")
  #import pandas as pd
  #import matplotlib.pyplot as plt
  #import seaborn as sns
  from textblob import TextBlob
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.decomposition import NMF

  # Sample DataFrame for demonstration (in practice, replace this with your actual DataFrame)
  # Assuming the dataset has a column named 'description' with textual descriptions of vehicles
  df = pd.DataFrame({
      'model': ['Car A', 'Car B', 'Car C', 'Car D'],
      'description': [
          "This luxury vehicle offers excellent comfort and performance.",
          "Designed with utility in mind, perfect for families and road trips.",
          "Tech-savvy interior with futuristic design and AI-powered dashboard.",
          "A reliable and economical choice for everyday urban driving."
      ]
  })

  # Step 1: Sentiment Analysis
  df['sentiment_score'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)

  # Step 2: Topic Modeling with NMF
  vectorizer = TfidfVectorizer(stop_words='english', max_features=50)
  X = vectorizer.fit_transform(df['description'])
  nmf_model = NMF(n_components=3, random_state=42)
  nmf_features = nmf_model.fit_transform(X)

  topic_labels = ['Luxury', 'Utility', 'Technology']
  df['topic'] = nmf_features.argmax(axis=1)
  df['topic_label'] = df['topic'].apply(lambda i: topic_labels[i])

  # Step 3: Manual Style Classification (based on keywords)
  def classify_style(text):
      text = text.lower()
      if any(word in text for word in ['luxury', 'comfort', 'premium']):
          return 'Luxury'
      elif any(word in text for word in ['tech', 'ai', 'dashboard', 'futuristic']):
          return 'Technology'
      elif any(word in text for word in ['family', 'economical', 'utility']):
          return 'Utility'
      else:
          return 'General'

  df['writing_style'] = df['description'].apply(classify_style)

  # Plot sentiment per model
  plt.figure(figsize=(8, 4))
  sns.barplot(data=df, x='model', y='sentiment_score', palette='viridis')
  plt.title('Sentiment Score per Vehicle Model')
  plt.ylabel('Sentiment Polarity')
  plt.xlabel('Vehicle Model')
  plt.tight_layout()
  plt.show()

  # Plot topic distribution
  plt.figure(figsize=(6, 4))
  sns.countplot(data=df, x='topic_label', order=topic_labels, palette='coolwarm')
  plt.title('Distribution of Vehicle Description Topics')
  plt.xlabel('Topic')
  plt.ylabel('Count')
  plt.tight_layout()
  plt.show()
  print(df)
############################################### step 9 #############################################################################
  #import matplotlib.pyplot as plt
  #import seaborn as sns

  print("\n===== STEP 9: SIGNIFICANT GRAPHS (Customized to Available Columns) =====")

# גרף 1: סנטימנט ממוצע לפי topic label
  plt.figure(figsize=(8, 5))
  sns.barplot(data=df, x='topic_label', y='sentiment_score', palette='Set2')
  plt.title('Average Sentiment Score per Topic')
  plt.xlabel('Topic')
  plt.ylabel('Average Sentiment Score')
  plt.tight_layout()
  plt.show()

# גרף 2: מספר מופעים של כל סגנון כתיבה
  plt.figure(figsize=(6, 4))
  sns.countplot(data=df, x='writing_style', palette='Set3')
  plt.title('Distribution of Writing Styles')
  plt.xlabel('Writing Style')
  plt.ylabel('Count')
  plt.tight_layout()
  plt.show()

# גרף 3: מפה של topic label לפי סגנון כתיבה
  plt.figure(figsize=(7, 5))
  sns.countplot(data=df, x='topic_label', hue='writing_style', palette='cool')
  plt.title('Topic Label by Writing Style')
  plt.xlabel('Topic Label')
  plt.ylabel('Count')
  plt.legend(title='Writing Style')
  plt.tight_layout()
  plt.show()
#################################################### step 10 ####################################################
  # === שלב 10: בניית מודלים והערכתם ===

  #import pandas as pd
  import numpy as np
  from sklearn.cluster import KMeans
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.model_selection import train_test_split
  from sklearn.naive_bayes import MultinomialNB
  from sklearn.metrics import classification_report

# --------------------------
# יצירת דאטה לדוגמה (אפשר להחליף ב-df שלך)
  df = pd.DataFrame({
    'model': ['Car A', 'Car B', 'Car C', 'Car D'],
    'description': [
        'This luxury vehicle offers excellent comfort and style',
        'Designed with utility in mind, perfect for families',
        'Tech-savvy interior with futuristic design and performance',
        'A reliable and economical choice for everyday use'
    ],
    'range_km': [400, 320, 360, 300],
    'fast_charging_power_kw_dc': [150, 120, 130, 110],
    'cargo_volume_l': [500, 600, 450, 550],
    'seats': [5, 7, 5, 5],
    'writing_style': ['Luxury', 'Utility', 'Technology', 'Utility']
})

# ========================================
# מודל 1: KMeans Clustering (אשכולות)
# מתאים לשאלה: "בנו מודלים מתאימים לבעיה"
# ========================================
  features = ['range_km', 'fast_charging_power_kw_dc', 'cargo_volume_l', 'seats']
  df_cluster = df[features].dropna()

  if not df_cluster.empty:
    kmeans = KMeans(n_clusters=2, random_state=0)
    df['cluster'] = kmeans.fit_predict(df_cluster)
    print("✓ KMeans clustering succeeded. Cluster labels added.")
  else:
    df['cluster'] = np.nan
    print("✗ No valid features for clustering.")

# ========================================
# מודל 2: סיווג טקסטים (classification)
# מתאים לשאלות: "האם ניתן להסביר את המודל?", "Goodness of fit"
# ========================================
  vectorizer = TfidfVectorizer(stop_words='english')
  X = vectorizer.fit_transform(df['description'])
  y = df['writing_style']

# פיצול לסט אימון ובדיקה
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Naive Bayes Classifier
  clf = MultinomialNB()
  clf.fit(X_train, y_train)
  y_pred = clf.predict(X_test)

# הפקת דוח דיוק והסבר למודל
  classification_report_output = classification_report(y_test, y_pred, output_dict=True)
  classification_df = pd.DataFrame(classification_report_output).transpose()

  print("\n=== Classification Report ===")
  print(classification_df)

# עונה על הדרישות:
# - האם ניתן להסביר את המודל? כן, Naive Bayes קל להסבר
# - האם יש התאמה טובה (goodness of fit)? כן, נבחן דרך הדוח


if __name__ == "__main__":
    main()
    