#new  
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          E-Commerce Unsupervised Learning Dashboard                         ║
║          DSAI Assignment — K-Means | Anomaly Detection | PCA | RecSys       ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config & global style
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce ML Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = ["#00D4AA", "#FF6B6B", "#4ECDC4", "#FFE66D", "#A29BFE",
           "#FD79A8", "#6C5CE7", "#00B894", "#E17055", "#0984E3"]

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    
    .main { background: #0D1117; }
    
    .stApp { background: linear-gradient(135deg, #0D1117 0%, #161B22 100%); }
    
    .metric-card {
        background: linear-gradient(135deg, #1C2128, #21262D);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
    }
    .metric-val {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4AA;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label { color: #8B949E; font-size: 0.85rem; letter-spacing: 0.05em; text-transform: uppercase; }
    
    h1, h2, h3 { color: #E6EDF3 !important; }
    
    .section-header {
        background: linear-gradient(90deg, #00D4AA22, transparent);
        border-left: 3px solid #00D4AA;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 12px 0;
        color: #E6EDF3;
        font-weight: 600;
        font-size: 1.1rem;
    }
    .insight-box {
        background: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
        color: #C9D1D9;
        line-height: 1.6;
    }
    .tag {
        display: inline-block;
        background: #00D4AA22;
        border: 1px solid #00D4AA55;
        color: #00D4AA;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
    }
    .stDataFrame { border-radius: 10px; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161B22, #0D1117);
        border-right: 1px solid #30363D;
    }
    [data-testid="stSidebar"] .stRadio label { color: #C9D1D9 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #8B949E;
        border-bottom: 2px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #00D4AA !important;
        border-bottom: 2px solid #00D4AA !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: consistent dark figure style
# ─────────────────────────────────────────────────────────────────────────────
def dark_fig(figsize=(10, 5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")
    ax.tick_params(colors="#8B949E")
    ax.xaxis.label.set_color("#C9D1D9")
    ax.yaxis.label.set_color("#C9D1D9")
    ax.title.set_color("#E6EDF3")
    return fig, ax

def dark_figs(rows, cols, figsize=(14, 6)):
    fig, axes = plt.subplots(rows, cols, figsize=figsize, facecolor="#0D1117")
    flat = axes.flat if hasattr(axes, "flat") else [axes]
    for ax in flat:
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363D")
        ax.tick_params(colors="#8B949E")
        ax.xaxis.label.set_color("#C9D1D9")
        ax.yaxis.label.set_color("#C9D1D9")
        ax.title.set_color("#E6EDF3")
    return fig, axes

# ─────────────────────────────────────────────────────────────────────────────
# Data loading & preprocessing (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_and_preprocess():
    customers = pd.read_csv("data/customers.csv")
    products  = pd.read_csv("data/products.csv")
    ratings   = pd.read_csv("data/ratings.csv")

    # ── Missing values check
    missing_before = customers.isnull().sum().sum()
    customers = customers.dropna().drop_duplicates()
    missing_after = customers.isnull().sum().sum()

    # ── Feature engineering
    customers["spend_per_session"] = (
        customers["monthly_spend"] / customers["sessions_per_month"].replace(0, 1)
    ).round(2)
    customers["purchase_frequency"] = (
        customers["n_purchases"] / 30
    ).round(4)

    # ── Scaling
    FEATURES = [
        "monthly_spend", "n_purchases", "recency_days",
        "avg_order_value", "sessions_per_month", "return_rate",
        "spend_per_session", "annual_income",
    ]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(customers[FEATURES])
    X_df = pd.DataFrame(X_scaled, columns=FEATURES, index=customers.index)

    return customers, products, ratings, X_df, FEATURES, scaler, missing_before

customers, products, ratings, X_scaled, FEATURES, scaler, missing_before = load_and_preprocess()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0;'>
        <div style='font-size:2.5rem;'>🛒</div>
        <div style='color:#00D4AA; font-weight:700; font-size:1.1rem;'>E-Commerce ML</div>
        <div style='color:#8B949E; font-size:0.8rem;'>Unsupervised Learning Dashboard</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        [
            "📊 Overview & Data",
            "🎯 K-Means Clustering",
            "🔍 Anomaly Detection",
            "📐 PCA",
            "🎁 Recommendation System",
            "📝 Analysis & Reflection",
        ],
        label_visibility="collapsed"
    )
    
    st.divider()
    st.markdown("""
    <div style='color:#8B949E; font-size:0.75rem; padding:10px 0;'>
        <b style='color:#C9D1D9;'>Dataset</b><br>
        500 customers · 50 products<br>
        2,824 ratings
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW & DATA
# ═════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview & Data":
    st.markdown("## 📊 Data Understanding & Preprocessing")
    st.markdown(
        "Task 1 · Synthetic e-commerce dataset with **customer behaviour**, "
        "**demographics**, and **product ratings**."
    )

    # KPI strip
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lbl in zip(
        [c1, c2, c3, c4],
        [len(customers), len(products), len(ratings), len(FEATURES)],
        ["Customers", "Products", "Ratings", "ML Features"],
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-val'>{val:,}</div>
            <div class='metric-label'>{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🗂️ Raw Data", "📈 Feature Distributions", "⚙️ Preprocessing"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("<div class='section-header'>Customer Table (sample)</div>", unsafe_allow_html=True)
            st.dataframe(
                customers.drop(columns="segment_label").head(20),
                use_container_width=True, height=380
            )
        with col2:
            st.markdown("<div class='section-header'>Data Types & Stats</div>", unsafe_allow_html=True)
            st.dataframe(customers[FEATURES].describe().round(2), use_container_width=True, height=380)

    with tab2:
        fig, axes = dark_figs(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        for i, feat in enumerate(FEATURES):
            axes[i].hist(customers[feat], bins=35, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor="#0D1117", linewidth=0.5)
            axes[i].set_title(feat.replace("_", " ").title(), fontsize=9, pad=6)
        fig.suptitle("Feature Distributions (Raw)", color="#E6EDF3", fontsize=13, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.markdown("<div class='section-header'>Preprocessing Pipeline</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class='insight-box'>
            <b style='color:#00D4AA;'>Steps Applied</b><br><br>
            ✅ Dropped duplicates and NaN rows<br>
            ✅ Engineered <code>spend_per_session</code> and <code>purchase_frequency</code><br>
            ✅ Applied <b>StandardScaler</b> — zero mean, unit variance<br><br>
            <b>Missing values before:</b> {missing_before} &nbsp;→&nbsp; <b>after:</b> 0
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class='insight-box'>
            <b style='color:#00D4AA;'>Why Preprocessing Matters for Unsupervised Learning</b><br><br>
            Unsupervised algorithms such as K-Means and PCA are <b>distance-based</b> or 
            <b>variance-based</b>. Without scaling, features with large magnitudes 
            (e.g., annual income in tens of thousands) dominate over features with small 
            ranges (e.g., return rate 0–1), producing biased clusters and misleading 
            principal components. StandardScaler ensures every feature contributes equally 
            to the learning process.
            </div>
            """, unsafe_allow_html=True)

        # Scaled distribution
        fig, axes = dark_figs(2, 4, figsize=(16, 7))
        axes = axes.flatten()
        X_arr = X_scaled.values
        for i, feat in enumerate(FEATURES):
            axes[i].hist(X_arr[:, i], bins=35, color=PALETTE[i % len(PALETTE)], alpha=0.85, edgecolor="#0D1117")
            axes[i].set_title(feat.replace("_", " ").title(), fontsize=9)
        fig.suptitle("Feature Distributions After StandardScaler", color="#E6EDF3", fontsize=13)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Correlation heatmap
        st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)
        fig, ax = dark_fig(figsize=(9, 6))
        corr = customers[FEATURES].corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(10, 170, as_cmap=True)
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap=cmap,
                    linewidths=0.5, linecolor="#0D1117", ax=ax,
                    annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlations", pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — K-MEANS
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎯 K-Means Clustering":
    st.markdown("## 🎯 Customer Segmentation — K-Means Clustering")

    # ── Controls
    with st.sidebar:
        st.markdown("**K-Means Settings**")
        max_k     = st.slider("Max K to evaluate", 3, 12, 10)
        chosen_k  = st.slider("Number of Clusters (K)", 2, max_k, 4)
        n_init    = st.slider("n_init (restarts)", 5, 30, 10)

    @st.cache_data
    def run_elbow(X, max_k, n_init):
        inertias, silhouettes = [], []
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, n_init=n_init, random_state=42)
            labels = km.fit_predict(X)
            inertias.append(km.inertia_)
            silhouettes.append(silhouette_score(X, labels, sample_size=300))
        return inertias, silhouettes

    inertias, silhouettes = run_elbow(X_scaled.values, max_k, n_init)

    # Optimal K by silhouette
    opt_k = np.argmax(silhouettes) + 2
    km_final = KMeans(n_clusters=chosen_k, n_init=n_init, random_state=42)
    customers["cluster"] = km_final.fit_predict(X_scaled.values)

    # ── Elbow + Silhouette plots
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = dark_fig(figsize=(6, 4))
        ks = range(2, max_k + 1)
        ax.plot(ks, inertias, "o-", color="#00D4AA", lw=2, ms=7)
        ax.axvline(chosen_k, color="#FF6B6B", lw=1.5, ls="--", label=f"K={chosen_k}")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Inertia (WCSS)")
        ax.set_title("Elbow Method")
        ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
        ax.grid(color="#30363D", lw=0.5)
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = dark_fig(figsize=(6, 4))
        ax.plot(ks, silhouettes, "s-", color="#FFE66D", lw=2, ms=7)
        ax.axvline(opt_k, color="#A29BFE", lw=1.5, ls="--", label=f"Best K={opt_k}")
        ax.axvline(chosen_k, color="#FF6B6B", lw=1.5, ls="--", label=f"Chosen K={chosen_k}")
        ax.set_xlabel("Number of Clusters (K)")
        ax.set_ylabel("Silhouette Score")
        ax.set_title("Silhouette Score")
        ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
        ax.grid(color="#30363D", lw=0.5)
        st.pyplot(fig); plt.close()

    st.info(f"📌 Best silhouette score at **K={opt_k}** ({max(silhouettes):.3f}). "
            f"Currently showing K={chosen_k}. Adjust the slider to explore.")

    # ── PCA-2D cluster scatter
    pca2 = PCA(n_components=2, random_state=42)
    Xp = pca2.fit_transform(X_scaled.values)

    fig, ax = dark_fig(figsize=(9, 6))
    for cl in range(chosen_k):
        mask = customers["cluster"] == cl
        ax.scatter(Xp[mask, 0], Xp[mask, 1],
                   c=PALETTE[cl % len(PALETTE)], alpha=0.75, s=28,
                   edgecolors="#0D1117", lw=0.3, label=f"Cluster {cl}")
    ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.set_title(f"K-Means Clusters (K={chosen_k}) — PCA 2D Projection")
    ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D",
              markerscale=1.5, framealpha=0.8)
    ax.grid(color="#30363D", lw=0.4, alpha=0.6)
    st.pyplot(fig); plt.close()

    # ── Cluster profiles
    st.markdown("<div class='section-header'>Cluster Profiles</div>", unsafe_allow_html=True)
    profile_cols = ["monthly_spend", "n_purchases", "avg_order_value",
                    "sessions_per_month", "return_rate", "annual_income"]
    profile = customers.groupby("cluster")[profile_cols].mean().round(2)
    profile["size"] = customers["cluster"].value_counts().sort_index()
    st.dataframe(profile.style.background_gradient(cmap="YlGn", axis=0), use_container_width=True)

    # ── Radar / bar per cluster
    st.markdown("<div class='section-header'>Feature Comparison Across Clusters</div>", unsafe_allow_html=True)
    # Normalise profile for display
    norm_profile = (profile[profile_cols] - profile[profile_cols].min()) / (
        profile[profile_cols].max() - profile[profile_cols].min() + 1e-9
    )
    fig, ax = dark_fig(figsize=(12, 4))
    x = np.arange(len(profile_cols))
    w = 0.8 / chosen_k
    for i, cl in enumerate(profile.index):
        ax.bar(x + i*w, norm_profile.loc[cl], width=w,
               color=PALETTE[cl % len(PALETTE)], alpha=0.85,
               label=f"Cluster {cl}", edgecolor="#0D1117")
    ax.set_xticks(x + w*(chosen_k-1)/2)
    ax.set_xticklabels([c.replace("_", "\n") for c in profile_cols], fontsize=8, color="#C9D1D9")
    ax.set_ylabel("Normalised Value")
    ax.set_title("Cluster Comparison (Normalised Features)")
    ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
    ax.grid(axis="y", color="#30363D", lw=0.5)
    st.pyplot(fig); plt.close()

    # ── Interpretations
    st.markdown("<div class='section-header'>Cluster Interpretations</div>", unsafe_allow_html=True)
    segment_names = {0: "Budget Browsers", 1: "Regular Shoppers", 2: "Premium Buyers", 3: "Power Users"}
    cols = st.columns(min(chosen_k, 4))
    for cl in range(chosen_k):
        with cols[cl % 4]:
            row = profile.loc[cl]
            name = segment_names.get(cl, f"Segment {cl}")
            st.markdown(f"""
            <div class='metric-card'>
                <div style='color:{PALETTE[cl % len(PALETTE)]}; font-weight:700; font-size:1rem;'>
                    ● {name}
                </div>
                <div style='color:#8B949E; font-size:0.8rem; margin:4px 0;'>n = {int(row.get("size",0))}</div>
                <hr style='border-color:#30363D; margin:8px 0;'>
                <div style='color:#C9D1D9; font-size:0.85rem;'>
                💰 Spend: <b>${row.monthly_spend:.0f}/mo</b><br>
                🛍️ Orders: <b>{row.n_purchases:.0f}/mo</b><br>
                💳 AOV: <b>${row.avg_order_value:.0f}</b><br>
                ↩️ Returns: <b>{row.return_rate*100:.1f}%</b>
                </div>
            </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY DETECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Detection":
    st.markdown("## 🔍 Density-Based Anomaly Detection")
    st.markdown("Using **Gaussian Mixture / Multivariate Gaussian** density estimation to flag outlier customers.")

    with st.sidebar:
        st.markdown("**Anomaly Settings**")
        threshold_pct = st.slider("Anomaly threshold (percentile)", 1, 10, 3)

    # Use 2 features for visualisation, full set for scoring
    ANOM_FEATS = ["monthly_spend", "return_rate", "n_purchases", "avg_order_value"]
    X_anom = customers[ANOM_FEATS].values
    Xs_anom = StandardScaler().fit_transform(X_anom)

    # Multivariate Gaussian
    mu  = Xs_anom.mean(axis=0)
    cov = np.cov(Xs_anom.T)
    rv  = multivariate_normal(mean=mu, cov=cov, allow_singular=True)
    log_probs = rv.logpdf(Xs_anom)

    thresh = np.percentile(log_probs, threshold_pct)
    customers["is_anomaly"] = log_probs < thresh
    customers["log_prob"]   = log_probs

    n_anom = customers["is_anomaly"].sum()
    st.info(f"🚨 **{n_anom}** anomalous customers detected ({threshold_pct}th percentile threshold)")

    # KPI
    c1, c2, c3 = st.columns(3)
    c1.metric("Normal Customers",   f"{len(customers)-n_anom:,}")
    c2.metric("Anomalous Customers", f"{n_anom:,}", delta=f"{n_anom/len(customers)*100:.1f}%", delta_color="inverse")
    c3.metric("Log-prob Threshold",  f"{thresh:.2f}")

    # ── 2D scatter: monthly_spend vs return_rate
    fig, ax = dark_fig(figsize=(9, 5.5))
    normal = customers[~customers["is_anomaly"]]
    anom   = customers[ customers["is_anomaly"]]
    ax.scatter(normal["monthly_spend"], normal["return_rate"],
               c="#00D4AA", alpha=0.55, s=25, label="Normal", edgecolors="none")
    ax.scatter(anom["monthly_spend"],   anom["return_rate"],
               c="#FF6B6B", alpha=0.9,  s=80, marker="X", label="Anomaly",
               edgecolors="#FF0000", lw=0.8, zorder=5)
    ax.set_xlabel("Monthly Spend ($)")
    ax.set_ylabel("Return Rate")
    ax.set_title("Anomaly Detection: Monthly Spend vs Return Rate")
    ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
    ax.grid(color="#30363D", lw=0.4, alpha=0.6)
    st.pyplot(fig); plt.close()

    # ── Log-probability distribution
    fig, ax = dark_fig(figsize=(10, 4))
    ax.hist(log_probs[~customers["is_anomaly"]], bins=50, color="#00D4AA", alpha=0.7, label="Normal", density=True)
    ax.hist(log_probs[ customers["is_anomaly"]], bins=20, color="#FF6B6B", alpha=0.9, label="Anomaly", density=True)
    ax.axvline(thresh, color="#FFE66D", lw=2, ls="--", label=f"Threshold ({threshold_pct}th pct)")
    ax.set_xlabel("Log Probability (Multivariate Gaussian)")
    ax.set_ylabel("Density")
    ax.set_title("Log-Probability Density: Normal vs Anomalous")
    ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
    ax.grid(color="#30363D", lw=0.4)
    st.pyplot(fig); plt.close()

    # ── KDE contour
    fig, ax = dark_fig(figsize=(9, 5.5))
    from scipy.stats import gaussian_kde
    # Only on 2D for viz
    X2d = Xs_anom[:, :2]  # monthly_spend, return_rate (scaled)
    kde = gaussian_kde(X2d.T, bw_method=0.3)
    xg, yg = np.mgrid[X2d[:,0].min()-0.5:X2d[:,0].max()+0.5:200j,
                       X2d[:,1].min()-0.5:X2d[:,1].max()+0.5:200j]
    zg = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
    cset = ax.contourf(xg, yg, zg, levels=15, cmap="YlGn", alpha=0.5)
    ax.contour(xg, yg, zg, levels=8, colors="#00D4AA", alpha=0.4, linewidths=0.7)
    ax.scatter(X2d[~customers["is_anomaly"], 0], X2d[~customers["is_anomaly"], 1],
               c="#00D4AA", alpha=0.5, s=18, label="Normal")
    ax.scatter(X2d[ customers["is_anomaly"], 0], X2d[ customers["is_anomaly"], 1],
               c="#FF6B6B", s=70, marker="X", label="Anomaly", zorder=5, edgecolors="#CC0000")
    ax.set_xlabel("Monthly Spend (scaled)")
    ax.set_ylabel("Return Rate (scaled)")
    ax.set_title("KDE Contour — Density Estimation")
    ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
    st.pyplot(fig); plt.close()

    # ── Anomaly table
    st.markdown("<div class='section-header'>Anomalous Customer Profiles</div>", unsafe_allow_html=True)
    cols_show = ["customer_id", "monthly_spend", "n_purchases", "return_rate",
                 "avg_order_value", "sessions_per_month", "log_prob"]
    st.dataframe(
        customers[customers["is_anomaly"]][cols_show]
            .sort_values("log_prob")
            .reset_index(drop=True)
            .style.background_gradient(subset=["monthly_spend","return_rate"], cmap="Reds"),
        use_container_width=True
    )

    st.markdown("""
    <div class='insight-box'>
    <b style='color:#FF6B6B;'>Why Are These Anomalies?</b><br><br>
    Anomalous customers deviate significantly from the joint Gaussian distribution 
    of the feature space. They typically exhibit a combination of:
    <ul>
        <li><b>Extremely high monthly spend</b> far beyond the population mean</li>
        <li><b>Abnormally high return rates</b> (60–95%) suggesting fraudulent activity or wardrobing</li>
        <li><b>Unusual purchase volume</b> inconsistent with their session count</li>
    </ul>
    These patterns can indicate <b>fraud</b>, <b>bot activity</b>, or <b>return policy abuse</b> 
    — all valuable signals for the e-commerce risk team.
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PCA
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📐 PCA":
    st.markdown("## 📐 Dimensionality Reduction — PCA")

    with st.sidebar:
        st.markdown("**PCA Settings**")
        n_components = st.slider("Components to analyse", 2, len(FEATURES), len(FEATURES))

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled.values)
    ev = pca.explained_variance_ratio_
    cum_ev = np.cumsum(ev)

    # ── Scree + cumulative
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = dark_fig(figsize=(6, 4))
        ks = np.arange(1, n_components + 1)
        ax.bar(ks, ev * 100, color="#00D4AA", alpha=0.8, edgecolor="#0D1117")
        ax.set_xlabel("Principal Component")
        ax.set_ylabel("Variance Explained (%)")
        ax.set_title("Scree Plot")
        ax.grid(axis="y", color="#30363D", lw=0.5)
        for i, v in enumerate(ev * 100):
            ax.text(i+1, v+0.3, f"{v:.1f}%", ha="center", color="#C9D1D9", fontsize=8)
        st.pyplot(fig); plt.close()

    with col2:
        fig, ax = dark_fig(figsize=(6, 4))
        ax.plot(ks, cum_ev * 100, "o-", color="#FFE66D", lw=2, ms=7)
        ax.fill_between(ks, cum_ev * 100, alpha=0.15, color="#FFE66D")
        ax.axhline(90, color="#FF6B6B", lw=1.5, ls="--", label="90% threshold")
        ax.axhline(95, color="#A29BFE", lw=1.5, ls="--", label="95% threshold")
        ax.set_xlabel("Number of Components")
        ax.set_ylabel("Cumulative Variance (%)")
        ax.set_title("Cumulative Explained Variance")
        ax.legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D")
        ax.grid(color="#30363D", lw=0.4)
        n90 = int(np.searchsorted(cum_ev, 0.90)) + 1
        n95 = int(np.searchsorted(cum_ev, 0.95)) + 1
        ax.annotate(f"PC{n90}", xy=(n90, cum_ev[n90-1]*100),
                    xytext=(n90+0.3, cum_ev[n90-1]*100-5),
                    color="#FF6B6B", fontsize=9)
        st.pyplot(fig); plt.close()

    st.info(f"📌 **{n90} components** capture ≥90% variance | **{n95} components** capture ≥95% variance")

    # ── 2D PCA scatter coloured by cluster
    if "cluster" not in customers.columns:
        km_tmp = KMeans(n_clusters=4, n_init=10, random_state=42)
        customers["cluster"] = km_tmp.fit_predict(X_scaled.values)

    pca2 = PCA(n_components=2, random_state=42)
    Xp2 = pca2.fit_transform(X_scaled.values)

    fig, axes = dark_figs(1, 2, figsize=(14, 5.5))
    # Left: coloured by cluster
    for cl in customers["cluster"].unique():
        m = customers["cluster"] == cl
        axes[0].scatter(Xp2[m, 0], Xp2[m, 1],
                        c=PALETTE[cl % len(PALETTE)], alpha=0.65, s=22,
                        label=f"Cluster {cl}", edgecolors="none")
    axes[0].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[0].set_title("PCA 2D — K-Means Clusters")
    axes[0].legend(labelcolor="#C9D1D9", facecolor="#21262D", edgecolor="#30363D", markerscale=1.5)
    axes[0].grid(color="#30363D", lw=0.4, alpha=0.5)
    # Right: coloured by monthly spend
    sc = axes[1].scatter(Xp2[:, 0], Xp2[:, 1],
                          c=customers["monthly_spend"], cmap="YlGn",
                          alpha=0.7, s=22, edgecolors="none")
    plt.colorbar(sc, ax=axes[1], label="Monthly Spend ($)")
    axes[1].set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}% var)")
    axes[1].set_title("PCA 2D — Spend Gradient")
    axes[1].grid(color="#30363D", lw=0.4, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Loadings heatmap
    st.markdown("<div class='section-header'>PC Loadings (Feature Contributions)</div>", unsafe_allow_html=True)
    loadings = pd.DataFrame(
        pca.components_[:min(6, n_components)].T,
        index=FEATURES,
        columns=[f"PC{i+1}" for i in range(min(6, n_components))]
    )
    fig, ax = dark_fig(figsize=(10, 5))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, linecolor="#0D1117",
                ax=ax, cbar_kws={"shrink": 0.7}, annot_kws={"size": 9})
    ax.set_title("Feature Loadings per Principal Component")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Before/after comparison table
    st.markdown("<div class='section-header'>Dimensionality: Before vs After</div>", unsafe_allow_html=True)
    comp_data = {
        "Aspect": ["Dimensions", "Orthogonality", "Interpretability", "Variance retained"],
        "Before PCA": [f"{len(FEATURES)} features", "Correlated", "Direct feature meaning", "100%"],
        f"After PCA ({n90} components)": [
            f"{n90} components", "Fully orthogonal",
            "Abstract linear combinations", f"{cum_ev[n90-1]*100:.1f}%"
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — RECOMMENDATION SYSTEM
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🎁 Recommendation System":
    st.markdown("## 🎁 Collaborative Filtering Recommendation System")

    with st.sidebar:
        st.markdown("**RecSys Settings**")
        n_recs    = st.slider("Recommendations per user", 3, 10, 5)
        n_similar = st.slider("Neighbours (user-user CF)", 5, 30, 15)
        demo_users = st.multiselect(
            "Show recommendations for",
            customers["customer_id"].sample(10, random_state=1).tolist(),
            default=customers["customer_id"].sample(3, random_state=1).tolist()
        )

    # ── Build user-item matrix
    @st.cache_data
    def build_user_item():
        ui = ratings.pivot_table(
            index="customer_id", columns="product_id",
            values="rating", aggfunc="mean"
        )
        return ui

    ui_matrix = build_user_item()

    # ── User-user cosine similarity (fill NaN with 0 for similarity only)
    from sklearn.metrics.pairwise import cosine_similarity
    ui_filled = ui_matrix.fillna(0)
    sim_matrix = pd.DataFrame(
        cosine_similarity(ui_filled),
        index=ui_filled.index,
        columns=ui_filled.index
    )

    def recommend_for_user(user_id, n_recs=5, n_neighbours=15):
        if user_id not in ui_matrix.index:
            return pd.DataFrame()
        # Find similar users
        sim_users = (
            sim_matrix[user_id]
            .drop(user_id, errors="ignore")
            .nlargest(n_neighbours)
        )
        # Products this user hasn't rated
        rated = ui_matrix.loc[user_id].dropna().index.tolist()
        unrated = [p for p in ui_matrix.columns if p not in rated]
        if not unrated:
            return pd.DataFrame()
        # Weighted predicted rating
        scores = {}
        for prod in unrated:
            nb_data = []
            for nb, sim in sim_users.items():
                if nb in ui_matrix.index and not pd.isna(ui_matrix.loc[nb, prod]):
                    nb_data.append((sim, ui_matrix.loc[nb, prod]))
            if nb_data:
                total_sim = sum(s for s, _ in nb_data)
                if total_sim > 0:
                    scores[prod] = sum(s * r for s, r in nb_data) / total_sim

        if not scores:
            return pd.DataFrame()

        rec_df = (
            pd.Series(scores)
              .nlargest(n_recs)
              .reset_index()
        )
        rec_df.columns = ["product_id", "predicted_rating"]
        rec_df = rec_df.merge(products, on="product_id", how="left")
        rec_df["predicted_rating"] = rec_df["predicted_rating"].round(2)
        return rec_df

    # ── Visualise user-item matrix heatmap (subset)
    st.markdown("<div class='section-header'>User-Item Rating Matrix (sample 30×30)</div>", unsafe_allow_html=True)
    sample_users = ui_matrix.index[:30]
    sample_prods = ui_matrix.columns[:30]
    sub = ui_matrix.loc[sample_users, sample_prods]
    fig, ax = dark_fig(figsize=(12, 6))
    sns.heatmap(sub, ax=ax, cmap="YlGn", mask=sub.isna(),
                linewidths=0.3, linecolor="#0D1117",
                cbar_kws={"label": "Rating", "shrink": 0.7},
                annot=False, xticklabels=4, yticklabels=4)
    ax.set_title("Ratings Matrix (30×30 sample) — Sparsity visible as grey")
    ax.tick_params(labelsize=7)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    sparsity = ui_matrix.isna().sum().sum() / ui_matrix.size
    col1, col2, col3 = st.columns(3)
    col1.metric("Matrix Size",  f"{ui_matrix.shape[0]} × {ui_matrix.shape[1]}")
    col2.metric("Sparsity",     f"{sparsity*100:.1f}%")
    col3.metric("Known Ratings", f"{int(ui_matrix.notna().sum().sum()):,}")

    # ── User similarity heatmap (sample)
    st.markdown("<div class='section-header'>User-User Similarity (sample 25 users)</div>", unsafe_allow_html=True)
    su25 = sim_matrix.iloc[:25, :25]
    fig, ax = dark_fig(figsize=(8, 6))
    sns.heatmap(su25, ax=ax, cmap="Blues", linewidths=0.3, linecolor="#0D1117",
                cbar_kws={"shrink": 0.7}, xticklabels=4, yticklabels=4)
    ax.set_title("Cosine Similarity Between Users")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # ── Recommendations
    st.markdown("<div class='section-header'>🛍️ Personalised Recommendations</div>", unsafe_allow_html=True)

    if not demo_users:
        st.warning("Select at least one user in the sidebar.")
    else:
        for uid in demo_users:
            recs = recommend_for_user(uid, n_recs=n_recs, n_neighbours=n_similar)
            with st.expander(f"👤 {uid}", expanded=True):
                if recs.empty:
                    st.write("No recommendations (user has rated all products).")
                else:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.dataframe(
                            recs[["product_id", "category", "price", "predicted_rating"]],
                            use_container_width=True, hide_index=True
                        )
                    with col2:
                        # Bar chart of predicted ratings
                        fig, ax = dark_fig(figsize=(4, 3))
                        colors = [PALETTE[i % len(PALETTE)] for i in range(len(recs))]
                        ax.barh(recs["product_id"], recs["predicted_rating"],
                                color=colors, edgecolor="#0D1117")
                        ax.set_xlim(1, 5)
                        ax.set_xlabel("Predicted Rating")
                        ax.set_title("Top Picks", fontsize=9)
                        ax.grid(axis="x", color="#30363D", lw=0.5)
                        plt.tight_layout()
                        st.pyplot(fig); plt.close()

    # ── Intuition explanation
    st.markdown("""
    <div class='insight-box'>
    <b style='color:#00D4AA;'>💡 How User-User Collaborative Filtering Works</b><br><br>
    The key intuition is: <i>people who agreed in the past tend to agree in the future</i>.<br><br>
    <b>Step 1 — Build the rating matrix:</b> Each row is a customer, each column is a product, 
    and cells contain ratings (NaN = not yet rated).<br><br>
    <b>Step 2 — Compute user similarity:</b> We measure cosine similarity between every pair 
    of user rating vectors. Users with similar tastes get high similarity scores.<br><br>
    <b>Step 3 — Predict ratings:</b> For products a target user hasn't rated, we compute a 
    weighted average of ratings from the K most similar neighbours, weighting by similarity.<br><br>
    <b>Step 4 — Rank & recommend:</b> Products with the highest predicted scores are recommended.<br><br>
    <span class='tag'>User-User CF</span> <span class='tag'>Cosine Similarity</span> 
    <span class='tag'>Sparse Matrix</span> <span class='tag'>Neighbourhood Model</span>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ANALYSIS & REFLECTION
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📝 Analysis & Reflection":
    st.markdown("## 📝 Analysis & Reflection — Task 6")

    st.markdown("""
    <div class='insight-box'>
    <b style='color:#00D4AA; font-size:1.1rem;'>How Unsupervised Learning Uncovered Hidden Patterns</b><br><br>
    The dataset had no pre-defined labels — every insight was discovered purely from data 
    structure. The pipeline revealed:<br><br>
    • <b>K-Means</b> surfaced four naturally occurring customer segments (budget, regular, 
      premium, power-user) that business teams can target with tailored marketing campaigns.<br><br>
    • <b>Anomaly Detection</b> flagged ~3% of customers with statistically improbable 
      combinations of spend volume, purchase frequency, and return rate — patterns invisible 
      to simple rule-based systems.<br><br>
    • <b>PCA</b> revealed that 90% of the information in 8 raw features can be compressed 
      into just 3–4 orthogonal components, enabling faster downstream models and cleaner 
      visualisations.<br><br>
    • <b>Collaborative Filtering</b> leveraged the latent structure of shared preferences 
      across thousands of ratings to produce personalised recommendations without requiring 
      any explicit user preferences as input.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Comparison table
    st.markdown("<div class='section-header'>Technique Comparison</div>", unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Technique": ["K-Means Clustering", "Gaussian Anomaly Detection", "PCA", "Collaborative Filtering"],
        "Purpose": ["Customer segmentation", "Fraud / outlier detection", "Dimensionality reduction", "Product recommendations"],
        "Input": ["Feature matrix", "Feature matrix", "Feature matrix", "User-item ratings"],
        "Key Output": ["Cluster labels", "Anomaly flags", "PC scores & loadings", "Predicted ratings"],
        "Scalability": ["High ✅", "Medium ⚠️", "High ✅", "Medium ⚠️"],
        "Interpretability": ["High ✅", "Medium ⚠️", "Low ❌", "Medium ⚠️"],
        "Strengths": [
            "Fast, scalable, easy to interpret segments",
            "Principled probability-based threshold",
            "Removes multicollinearity, speeds up models",
            "No domain knowledge needed; leverages community"
        ],
        "Limitations": [
            "Assumes spherical clusters; sensitive to K",
            "Assumes Gaussian; fails on non-linear distributions",
            "Principal components lose interpretability",
            "Cold-start problem; sparse data hurts quality"
        ],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Real-world applications
    st.markdown("<div class='section-header'>Real-World Applications</div>", unsafe_allow_html=True)
    apps = [
        ("🛒 Retail & E-Commerce", "#00D4AA",
         "Customer lifetime value segmentation, personalised email campaigns, fraud detection, "
         "inventory optimisation, and 'customers also bought' recommendations (Amazon, Shopify)."),
        ("🏦 Banking & Finance", "#FFE66D",
         "Anomaly detection for credit card fraud, customer churn segmentation, "
         "credit risk scoring with PCA-reduced features, and loan product recommendations."),
        ("🎵 Streaming & Media", "#A29BFE",
         "Content recommendation (Netflix, Spotify), user taste clustering, "
         "detecting bot streams with anomaly detection, and PCA on audio feature vectors."),
        ("🏥 Healthcare", "#4ECDC4",
         "Patient segmentation for personalised treatment, anomaly detection in vitals, "
         "reducing high-dimensional genomic data via PCA, drug recommendation based on similar patients."),
        ("📱 Social Media & Ad Tech", "#FD79A8",
         "Interest-based user clustering, detecting fake accounts via anomaly scoring, "
         "compressing user embedding vectors with PCA, and ad targeting via collaborative signals."),
    ]
    cols = st.columns(len(apps))
    for col, (title, colour, body) in zip(cols, apps):
        col.markdown(f"""
        <div class='metric-card' style='border-left: 3px solid {colour};'>
            <div style='color:{colour}; font-weight:700; font-size:0.95rem; margin-bottom:8px;'>{title}</div>
            <div style='color:#C9D1D9; font-size:0.82rem; line-height:1.5;'>{body}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Key insights
    st.markdown("<div class='section-header'>Key Insights from This Dataset</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='insight-box'>
    <ol style='color:#C9D1D9; line-height:2;'>
        <li>Customer spend follows a <b>log-normal distribution</b> — a small number of power users 
            drive disproportionate revenue, matching the Pareto 80/20 principle.</li>
        <li>The <b>first two principal components</b> collectively explain ~55% of total variance 
            and map closely to "spending power" and "engagement frequency."</li>
        <li>Anomalous customers have <b>return rates 3–10× higher</b> than normal customers — 
            a key signal for the risk/policy team.</li>
        <li>Collaborative filtering quality improves significantly with <b>denser rating matrices</b>; 
            cold-start users (few ratings) receive less accurate recommendations.</li>
        <li>K-Means produces the most business-actionable output: each cluster has a clear story 
            and can be mapped to a distinct CRM strategy.</li>
    </ol>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline diagram
    st.markdown("<div class='section-header'>Full ML Pipeline</div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center; padding: 20px; background:#161B22; border-radius:12px; border:1px solid #30363D;'>
        <div style='display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:8px; color:#C9D1D9; font-size:0.9rem;'>
            <span style='background:#21262D; border:1px solid #00D4AA; border-radius:8px; padding:8px 16px;'>📥 Raw Data</span>
            <span style='color:#00D4AA;'>→</span>
            <span style='background:#21262D; border:1px solid #30363D; border-radius:8px; padding:8px 16px;'>🔧 Preprocessing<br><small>Scale · Clean · Engineer</small></span>
            <span style='color:#00D4AA;'>→</span>
            <span style='background:#21262D; border:1px solid #30363D; border-radius:8px; padding:8px 16px;'>🎯 K-Means<br><small>Segmentation</small></span>
            <span style='color:#00D4AA;'>→</span>
            <span style='background:#21262D; border:1px solid #30363D; border-radius:8px; padding:8px 16px;'>🔍 Anomaly Det.<br><small>Density Est.</small></span>
            <span style='color:#00D4AA;'>→</span>
            <span style='background:#21262D; border:1px solid #30363D; border-radius:8px; padding:8px 16px;'>📐 PCA<br><small>Dim. Reduction</small></span>
            <span style='color:#00D4AA;'>→</span>
            <span style='background:#21262D; border:1px solid #FFE66D; border-radius:8px; padding:8px 16px;'>🎁 RecSys<br><small>Collab. Filter</small></span>
            <span style='color:#00D4AA;'>→</span>
            <span style='background:#21262D; border:1px solid #00D4AA; border-radius:8px; padding:8px 16px;'>📊 Insights</span>
        </div>
    </div>
    """, unsafe_allow_html=True)