import pandas as pd
import plotly.express as px
import umap
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

client = MongoClient("mongodb://localhost:27017")
db = client["scientific_writting"]
collection = db["resumos"]

docs = list(collection.find())

# 3) Extrair frases (objetivos + conclusões)
data = []
for doc in docs:
    for obj in doc.get("objetivos", []):
        data.append({
            "id_doc": doc["_id"],
            "tipo": "objetivo",
            "texto": obj.get("construcoes_linguisticas", ""),
            "categoria": obj.get("categoria", "")
        })
    for concl in doc.get("conclusoes", []):
        data.append({
            "id_doc": doc["_id"],
            "tipo": "conclusao",
            "texto": concl.get("construcoes_linguisticas", ""),
            "categoria": concl.get("categoria", "")
        })

df = pd.DataFrame(data)
df = df[df["texto"].str.strip() != ""]  # remove vazios

# 4) Gerar embeddings (frases -> vetores)
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["texto"].tolist(), show_progress_bar=True)

# 5) Redução dimensional com UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
embedding_2d = umap_model.fit_transform(embeddings)

df["x"] = embedding_2d[:,0]
df["y"] = embedding_2d[:,1]

# 6) Clusterização com KMeans
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(embeddings)

# 7) Visualização interativa com Plotly
fig = px.scatter(
    df,
    x="x", y="y",
    color="cluster",
    hover_data=["tipo", "categoria", "texto"],
    symbol="tipo",
    title="Clusters de Objetivos e Conclusões"
)
fig.show()

# 1) Calcular matriz de similaridade
similarity_matrix = cosine_similarity(embeddings)

# 2) Criar rótulos curtos para visualização
labels = [f"{row.tipo.upper()}-{row.categoria}-{row.id_doc}" for row in df.itertuples()]

# 3) Gerar heatmap interativo
fig = px.imshow(
    similarity_matrix,
    x=labels,
    y=labels,
    color_continuous_scale="Viridis",
    title="Matriz de Similaridade entre Objetivos e Conclusões"
)

fig.update_xaxes(side="top", tickangle=45)
fig.show()
