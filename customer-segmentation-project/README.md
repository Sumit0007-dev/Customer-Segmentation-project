# Customer Segmentation using K-Means Clustering

## Project structure
```
customer-segmentation-project/
├── data/
│   └── mall_customers.csv          # raw dataset
├── src/
│   └── customer_segmentation.py    # full pipeline
├── outputs/                        # generated on run: plots + result CSVs
├── requirements.txt
├── .gitignore
└── README.md
```

## How to run
```bash
git clone <your-repo-url>
cd customer-segmentation-project
pip install -r requirements.txt
python src/customer_segmentation.py
```
Outputs (`elbow_plot.png`, `customer_segments.png`, `cluster_summary.csv`,
`mall_customers_with_clusters.csv`) are written to `outputs/`.

## One-line summary
Grouped mall customers into 5 behavioural segments based on **Annual Income**
and **Spending Score**, using unsupervised learning (K-Means) — no labels
were available or needed.

## Files
| File | What it is |
|---|---|
| `mall_customers.csv` | Raw dataset (200 customers: ID, Gender, Age, Income, Spending Score) |
| `customer_segmentation.py` | Full pipeline: load → scale → elbow method → K-Means → visualize |
| `elbow_plot.png` | Chart used to pick the number of clusters (k) |
| `customer_segments.png` | Final scatter plot of the 5 clusters + centroids |
| `cluster_summary.csv` | Average Age/Income/Spending per cluster |
| `mall_customers_with_clusters.csv` | Original data with a `Cluster` column added |

## How to explain it in the interview (walk through in this order)

**1. The problem**
"I wanted to group customers by purchasing behaviour without any predefined
labels — that rules out classification and points to clustering."

**2. Feature choice**
"I used Annual Income and Spending Score because together they describe
*who has money* and *who spends it* — the two axes that actually separate
customer types."

**3. Why scale the data**
"K-Means uses Euclidean distance, so if Income (range ~15–140) and Spending
Score (range ~1–100) aren't on the same scale, Income would dominate the
distance calculation. I used `StandardScaler` to put both on a comparable
scale before clustering."

**4. Why K-Means, and why k=5**
"K-Means is simple, fast, and works well when clusters are roughly
spherical — which fits this kind of income/spending data. I used the
**Elbow Method**: plotted inertia (within-cluster sum of squared distances)
against different values of k, and looked for the point where adding more
clusters stops giving a meaningful drop. That was around k=5."

**5. What the clusters mean (this is the part that shows understanding, not
just running code)** — actual numbers from this run:

| Cluster | Income | Spending | Interpretation |
|---|---|---|---|
| 0 | High (~89k) | Low (~22) | High earners, cautious spenders — "careful affluent" |
| 1 | High (~91k) | High (~84) | High earners, big spenders — target for premium offers |
| 2 | Mid (~53k) | Mid (~47) | Average, unremarkable segment |
| 3 | Low (~25k) | High (~81) | Low income, high spending — impulsive / aspirational buyers |
| 4 | Low (~25k) | Low (~18) | Low income, low spending — price-sensitive, low engagement |

**6. Business takeaway**
"A business could target Cluster 1 with loyalty/premium programs, and try to
convert Cluster 0 (money but not spending) with better engagement, while
Cluster 4 probably isn't worth heavy marketing spend."

**7. Limitations (say this proactively — it shows maturity)**
- K-Means assumes roughly round, evenly-sized clusters; real customer data
  may not always fit this assumption as cleanly.
- The choice of k from the elbow method is somewhat subjective — could
  validate further with **Silhouette Score**.
- Only 2 features used here; a real deployment might include recency/
  frequency/monetary (RFM) features for richer segmentation.

## Likely follow-up questions and short answers
- **"Why not hierarchical clustering?"** → K-Means scales better to larger
  datasets; hierarchical clustering is more useful for smaller datasets
  where you want a dendrogram / don't want to pre-specify k.
- **"How do you evaluate a clustering result without ground truth?"** →
  Inertia (elbow method) for choosing k, and Silhouette Score to check
  how well-separated the clusters are.
- **"What does `random_state=42` do?"** → Makes K-Means' random centroid
  initialization reproducible, so results are consistent across runs.
- **"Why `n_init=10`?"** → K-Means can converge to a local minimum depending
  on initial centroid placement; running it 10 times and keeping the best
  result reduces that risk.
