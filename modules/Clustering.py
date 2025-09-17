import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pyclustering.cluster.kmedoids import kmedoids
import random
from io import BytesIO



def run(df1):

    # 1) Load data
    df = df1.copy()
    BASKET = ['Readymade','Frozenfoods','Alcohol','FreshVegetables','Milk',
            'Bakerygoods','Freshmeat','Toiletries','Snacks','Tinnedgoods']
    DEMO   = ['GENDER','Age','MARITAL','CHILDREN','WORKING']
    for c in BASKET: df[c] = df[c].astype(float).round().astype(int)
    for c in DEMO:   df[c] = df[c].astype(str).str.strip()


     # 2) Encode for Hamming
    enc = pd.DataFrame(index=df.index)
    for c in BASKET: enc[c] = df[c].astype(int)
    for c in DEMO:   enc[c] = pd.Categorical(df[c]).codes
    X = enc.values
    st.markdown(f"### Encode for Hamming: \n{X}")
    st.dataframe(X)

    # 3) Weights: BASKET/DEMO
    st.markdown('#### Select Weight for BASKET (%)')
    col1, col2, cal3 = st.columns(3)
    with col1:
        basket_weight = st.slider("", min_value=0, max_value=100, value=50, step=10)
    demo_weight = 100 - basket_weight
    BLOCK_WEIGHTS = {'basket': basket_weight / 100, 'demo': demo_weight / 100}
    w = np.array([BLOCK_WEIGHTS['basket']/len(BASKET) if c in BASKET else BLOCK_WEIGHTS['demo']/len(DEMO) for c in enc.columns], float)
    st.markdown(f"Encode for Hamming: \n{w}")

    # 4) Weighted Hamming distance
    def hamming_weighted_matrix(X, w, chunk=256):
        n, p = X.shape
        D = np.zeros((n, n), float)
        for i in range(0, n, chunk):
            Xi = X[i:i+chunk]
            for j in range(0, n, 1024):
                Xj = X[j:j+1024]
                mism = (Xi[:,None,:] != Xj[None,:,:]).astype(float)
                D[i:i+Xi.shape[0], j:j+Xj.shape[0]] = np.tensordot(mism, w, axes=([2],[0]))
        D = (D + D.T)/2; np.fill_diagonal(D, 0.0)
        return D
    D = hamming_weighted_matrix(X, w)
    st.markdown(f"### Weighted Hamming distance: \n{D}")
    st.dataframe(D)

    # 5) K-medoids (PAM)
    rng = np.random.default_rng(42)
    def kmedoids(D, k, max_iter=60, n_init=5):
        n = D.shape[0]
        best = None
        for _ in range(n_init):
            med = [int(rng.integers(0,n))]
            dmin = D[med[0]].copy()
            while len(med) < k:
                probs = dmin**2; s = probs.sum()
                med.append(int(rng.choice(n, p=probs/s))) if s > 0 else med.append(int(rng.choice([i for i in range(n) if i not in med])))
                dmin = np.minimum(dmin, D[med[-1]])
            med = np.array(med, int)
            labels = np.argmin(D[med], axis=0); cost = D[med].min(axis=0).sum()
            improved = True
            while improved:
                improved = False
                non_med = [i for i in range(n) if i not in med]
                for mi, m in enumerate(med.copy()):
                    best_delta, swap = 0.0, None
                    for h in non_med:
                        new_med = med.copy(); new_med[mi] = h
                        new_cost = D[new_med].min(axis=0).sum()
                        delta = new_cost - cost
                        if delta < best_delta: best_delta, swap = float(delta), h
                    if swap is not None:
                        med[mi] = swap
                        labels = np.argmin(D[med], axis=0); cost = D[med].min(axis=0).sum()
                        improved = True
            if (best is None) or (cost < best['cost']): best = {'medoids': med, 'labels': labels, 'cost': float(cost)}
        return best

    # 6) Silhouette score calculation
    def silhouette_precomputed(D, labels):
        labels = np.asarray(labels); n = len(labels); s = np.zeros(n)
        for i in range(n):
            same = (labels == labels[i])
            a = D[i, same].sum() / (same.sum() - 1) if same.sum() > 1 else 0.0
            b = min(D[i, labels == l].mean() for l in np.unique(labels) if l != labels[i])
            s[i] = 0 if max(a, b) == 0 else (b - a) / max(a, b)
        return s.mean()
    
    sil_score = []
    for k_val in range(2,11):
        model = kmedoids(D, k_val)
        labels = model['labels']
        silhouette_avg = silhouette_precomputed(D, labels)
        silhouette_avg = round(float(silhouette_avg), 2)
        sil_score.append((k_val, silhouette_avg))
        #st.markdown(f"{k_val}:{silhouette_avg}")
    st.markdown(f"### Silhouette score calculation from k = 2 to 10 : \n{sil_score}")
    df_sil = pd.DataFrame(sil_score, columns=["k", "Silhouette Score"])
    col1, col2, cal3 = st.columns(3)
    with col1:
        st.dataframe(df_sil)

    fig_sil = go.Figure()
    fig_sil.add_trace(go.Scatter(x=df_sil["k"], y=df_sil["Silhouette Score"],
                            mode='lines+markers',name='Silhouette Score'))
    fig_sil.update_layout(title="Silhouette Score by Number of Clusters (k)",
                    xaxis_title="Number of Clusters (k)",
                    yaxis_title="Silhouette Score",
                    template="plotly_white",
                    yaxis=dict(range=[0, 1])
                    )
    st.plotly_chart(fig_sil)

    # 7) เลือก k สำหรับ final clustering
    k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=3)
    model = kmedoids(D, k)
    labels = model['labels']
 
    # 9) label each cluster
    st.subheader("𝄜 Data table with labeling each cluster")
    df_out = df.copy(); df_out['cluster'] = labels
    st.dataframe(df_out)


    # 10) customer behavior insight and visualization
    st.subheader("𝄜 Summary Basket_rates table each cluster")
    basket_rates = df_out.groupby('cluster')[BASKET].mean().round(2)
    st.dataframe(basket_rates)
    cluster_summary = df_out.groupby('cluster').size().to_frame('n_cluster')
    cluster_summary['percent'] = (cluster_summary['n_cluster'] / len(df_out) * 100).round(2)
    st.dataframe(cluster_summary)

    # คำนวณค่าเฉลี่ยของแต่ละกลุ่ม
    cluster_means = basket_rates
    n_clusters = k
    shopping_cols = [col for col in basket_rates.columns]

    # กำหนดสีสำหรับแต่ละ Cluster
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # ===== Radar Chart รวม =====
    fig_combined = go.Figure()
    for i in range(n_clusters):
        fig_combined.add_trace(go.Scatterpolar(
            r=cluster_means.iloc[i].tolist(),
            theta=shopping_cols,
            fill='toself',
            #name=f'Cluster {i}',
            name=f"Cluster {i} : {cluster_summary['percent'].loc[i]:.2f}%",
            line=dict(color=colors[i])
        ))

    fig_combined.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])),
        showlegend=True,
        title="Radar Chart Summary: Purchasing behavior of each group"
    )

    st.subheader("📊 Radar Chart : All Clusters")
    st.plotly_chart(fig_combined)

    # ===== Radar Chart แยกแต่ละกลุ่ม =====
    st.subheader("📈 Radar Chart : Separate behavior by Clusters")

    figs = []
    for i in range(n_clusters):
        fig_single = go.Figure()
        fig_single.add_trace(go.Scatterpolar(
            r=cluster_means.iloc[i].tolist() + [cluster_means.iloc[i].tolist()[0]],
            theta=shopping_cols + [shopping_cols[0]],
            fill='toself',
            name=f'Cluster {i}',
            line=dict(color=colors[i])
        ))
        fig_single.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1])),
            showlegend=False,
            title=f"Cluster {i} : {cluster_summary['percent'].loc[i]:.2f}%"
        )
        figs.append(fig_single)
    # แสดงกราฟใน 3 คอลัมน์
    cols = st.columns(3)
    for i in range(n_clusters):
        with cols[i % 3]:
            st.plotly_chart(figs[i], use_container_width=True)

    # 11) Demo insight and visualization
    st.subheader("𝄜 Summary Demo_ratios table each cluster")
    demo_ratios = (df_out.groupby('cluster')[DEMO].value_counts(normalize=True)
                .rename('ratio').reset_index().round(3))
    st.dataframe(demo_ratios)

    # ===== Sunburst Chart แยกตาม Cluster =====
    st.subheader("🌞 Sunburst Chart — Separate demographic by Cluster")

    # ยืนยันว่ามีคอลัมน์ที่ต้องใช้ครบ
    required_cols = ['cluster', 'GENDER', 'Age', 'MARITAL', 'CHILDREN', 'WORKING', 'ratio']
    missing = [c for c in required_cols if c not in demo_ratios.columns]
    if missing:
        st.error(f"Sunburst ต้องการคอลัมน์: {required_cols} แต่ขาด {missing}")
    else:
        #levels = ['GENDER','Age','WORKING','CHILDREN','MARITAL']
        levels = ['WORKING','CHILDREN','GENDER','Age','MARITAL']

        def build_sunburst_nodes(df_c, cluster_id, lvls):
            """
            สร้างโครงสร้าง nodes สำหรับ go.Sunburst
            - root = Cluster {id}
            - ใช้ branchvalues='total' (ค่าของ node = ผลรวมลูก)
            """
            node_map = {}  # id -> dict(label, parent, value)
            total = float(df_c['ratio'].sum())  # ส่วนใหญ่ ~ 1.0 ภายในคลัสเตอร์

            # root node
            root_id = f"cluster={cluster_id}"
            node_map[root_id] = {
                'label': f"Cluster {cluster_id}",
                'parent': "",
                'value': total
            }

            # สร้างโหนดแต่ละชั้นจากกลุ่มค่า
            for L in range(len(lvls)):
                group_cols = lvls[:L+1]
                gb = (df_c.groupby(group_cols, dropna=False)['ratio']
                            .sum()
                            .reset_index())
                for _, row in gb.iterrows():
                    node_id   = "|".join([f"{col}={row[col]}" for col in group_cols])
                    parent_id = root_id if L == 0 else "|".join([f"{col}={row[col]}" for col in lvls[:L]])
                    node_map[node_id] = {
                        'label': str(row[group_cols[-1]]),
                        'parent': parent_id,
                        'value': float(row['ratio'])
                    }

            # แปลงเป็น arrays ตามลำดับ insert
            ids     = list(node_map.keys())
            labels  = [node_map[i]['label']  for i in ids]
            parents = [node_map[i]['parent'] for i in ids]
            values  = [node_map[i]['value']  for i in ids]
            return ids, labels, parents, values

        # แสดงกราฟใน 3 คอลัมน์ (เหมือนส่วน Radar ที่คุณทำไว้)
        cols = st.columns(3)
        figs_sb = []

        # ปรับ label ให้มีความหมายมากขึ้น
        demo_ratios['CHILDREN'] = demo_ratios['CHILDREN'].map({'Yes': 'CHILD(Y)', 'No': 'CHILD(N)'}).fillna(demo_ratios['CHILDREN'])
        demo_ratios['WORKING'] = demo_ratios['WORKING'].map({'Yes': 'WORK(Y)', 'No': 'WORK(N)'}).fillna(demo_ratios['WORKING'])

        # ใช้ n_clusters = k ที่คุณประกาศไว้ก่อนหน้า
        for i in range(n_clusters):
            df_c = demo_ratios[demo_ratios['cluster'] == i].copy()
            if df_c.empty:
                continue
            ids, labels, parents, values = build_sunburst_nodes(df_c, i, levels)

            # ดึง % คลัสเตอร์เพื่อใส่ใน title (ถ้ามีใน cluster_summary)
            title_pct = ""
            if (i in cluster_summary.index) and ('percent' in cluster_summary.columns):
                title_pct = f": {cluster_summary['percent'].loc[i]:.2f}%"

            fig_sb = go.Figure(
                go.Sunburst(
                    ids=ids,
                    labels=labels,
                    parents=parents,
                    values=values,
                    marker=dict(colors=values,  # ใช้ ratio เป็นค่าที่ใช้ไฮไลต์
                        colorscale='Blues',#'Viridis', 'Reds'
                    ),
                    branchvalues='total',
                    hovertemplate="ระดับ: %{label}<br>สัดส่วน: %{value:.2%}<extra></extra>",# โชว์เป็น % (เนื่องจาก values เป็น ratio ของคลัสเตอร์)
                    maxdepth=None,  # จำกัดความลึกได้ เช่น 3
                )
        
            )
            
            fig_sb.update_layout(
                title=f"Cluster {i} {title_pct}",
                margin=dict(l=10, r=10, t=40, b=10),
                uniformtext=dict(minsize=10, mode='hide')
            )
            figs_sb.append(fig_sb)

        # Render เป็น 3 คอลัมน์
        for idx, fig in enumerate(figs_sb):
            with cols[idx % 3]:
                st.plotly_chart(fig, use_container_width=True)
