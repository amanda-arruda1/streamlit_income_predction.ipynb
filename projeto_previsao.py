import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import plotly.express as px

st.set_page_config(page_title="Previsão de Renda")

st.title('📊 Análise e Previsão de Renda')
st.markdown('''
Este aplicativo permite analisar dados de renda e prever valores com base em modelos de machine learning.
- **Análise Exploratória**: Visualize estatísticas e distribuições dos dados.
- **Modelagem**: Treine e avalie modelos de árvore de decisão e random forest.
- **Resultados**: Compare o desempenho dos modelos.
''')

st.divider()

uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type="csv")
if uploaded_file is not None:
    renda = pd.read_csv(uploaded_file)
else:
    st.warning("Por favor, carregue um arquivo CSV.")
    st.stop()


colunas_irrelevantes = [col for col in renda.columns if 'Unnamed' in col or 'id_cliente' in col]
renda = renda.drop(columns=colunas_irrelevantes)


tab1, tab2, tab3, tab4 = st.tabs(["Análise Exploratória", "Modelagem", "Resultados", "Comentários"])


with tab1:
    st.subheader("Análise Exploratória")
    
    st.write("**Primeiras linhas do dataset:**")
    st.write(renda.head())

    st.write("**Estatísticas descritivas:**")
    st.write(renda.describe())

    st.write("**Valores faltantes por coluna:**")
    st.write(renda.isnull().sum())

    st.write("**Distribuição das variáveis numéricas (Plotly):**")
    st.write('''
    Os histogramas abaixo mostram a distribuição das variáveis numéricas. Eles ajudam a identificar:
    - **Assimetria**: Se os dados estão concentrados à esquerda ou à direita.
    - **Outliers**: Valores extremos que podem distorcer a análise.
    - **Tendência Central**: Onde a maioria dos dados está concentrada.
    ''')
    colunas_numericas = renda.select_dtypes(include=['int64', 'float64']).columns
    for coluna in colunas_numericas:
        fig = px.histogram(renda, x=coluna, nbins=30, title=f'Distribuição de {coluna}')
        st.plotly_chart(fig)


    st.write("**Matriz de Correlação:**")
    st.write('''
    A matriz de correlação mostra a relação linear entre as variáveis numéricas. Valores próximos de **1** indicam uma correlação positiva forte, enquanto valores próximos de **-1** indicam uma correlação negativa forte. Valores próximos de **0** indicam que não há correlação.
    ''')
    colunas_numericas = renda.select_dtypes(include=['int64', 'float64']).columns
    corr = renda[colunas_numericas].corr()  
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title='Matriz de Correlação')

    fig.update_layout(xaxis=dict(tickangle=-90))

    st.plotly_chart(fig)


colunas_relevantes = [
    'sexo', 'posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos', 'tipo_renda',
    'educacao', 'estado_civil', 'tipo_residencia', 'idade', 'tempo_emprego',
    'qt_pessoas_residencia', 'renda'
]
renda = renda[colunas_relevantes]


renda['tempo_emprego'] = renda['tempo_emprego'].fillna(renda['tempo_emprego'].median())


renda['renda'] = renda['renda'].clip(
    lower=renda['renda'].quantile(0.05),
    upper=renda['renda'].quantile(0.95)
)


renda['idade_ao_quadrado'] = renda['idade'] ** 2
renda['renda_per_capita'] = renda['renda'] / renda['qt_pessoas_residencia']


colunas_categoricas = renda.select_dtypes(include=['object']).columns.tolist()
colunas_numericas = renda.select_dtypes(include=['int64', 'float64']).columns.tolist()
colunas_numericas.remove('renda')  

preprocessador = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), colunas_numericas),
        ('cat', OneHotEncoder(), colunas_categoricas)
    ]
)

X = renda.drop('renda', axis=1)
y = renda['renda']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


with tab2:
    st.subheader("Modelagem")

    st.write("### Árvore de Decisão")
    max_depth = st.slider('Profundidade máxima da árvore', 1, 10, 3)
    min_samples_split = st.slider('Mínimo de amostras para dividir um nó', 2, 50, 20)
    min_samples_leaf = st.slider('Mínimo de amostras em uma folha', 1, 20, 10)

    modelo_arvore = Pipeline(steps=[
        ('preprocessador', preprocessador),
        ('regressor', DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        ))
    ])
    modelo_arvore.fit(X_train, y_train)

    if st.checkbox('Mostrar árvore de decisão'):
        st.write("Visualização da árvore de decisão:")
        dot_data = export_graphviz(
            modelo_arvore.named_steps['regressor'],
            out_file=None,
            feature_names=colunas_numericas + list(modelo_arvore.named_steps['preprocessador'].named_transformers_['cat'].get_feature_names_out(colunas_categoricas)),
            filled=True,
            rounded=True,
            special_characters=True
        )
        st.graphviz_chart(dot_data)

    st.write("### Random Forest")
    n_estimators = st.slider('Número de árvores', 10, 200, 100)
    max_depth_rf = st.slider('Profundidade máxima da árvore (RF)', 1, 10, 5)

    modelo_rf = Pipeline(steps=[
        ('preprocessador', preprocessador),
        ('regressor', RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth_rf,
            random_state=42
        ))
    ])
    modelo_rf.fit(X_train, y_train)

with tab3:
    st.subheader("Resultados")

    st.write("### Avaliação da Árvore de Decisão")
    y_pred = modelo_arvore.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)  
    rmse = mse ** 0.5  
    r2 = r2_score(y_test, y_pred)
    st.write(f'**RMSE (Árvore de Decisão):** {rmse:.3f}')
    st.write(f'**R² (Árvore de Decisão):** {r2:.3f}')
    st.write('''
    - **RMSE (Root Mean Squared Error)**: Mede a diferença média entre os valores previstos e os reais. Quanto menor, melhor.
    - **R² (Coeficiente de Determinação)**: Indica a proporção da variância dos dados que é explicada pelo modelo. Valores próximos de **1** indicam um bom ajuste.
    ''')

    st.divider()

   
    st.write("### Avaliação do Random Forest")
    y_pred_rf = modelo_rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)  
    rmse_rf = mse_rf ** 0.5  
    r2_rf = r2_score(y_test, y_pred_rf)
    st.write(f'**RMSE (Random Forest):** {rmse_rf:.3f}')
    st.write(f'**R² (Random Forest):** {r2_rf:.3f}')
    st.write('''
    - O **Random Forest** geralmente tem um desempenho melhor que a Árvore de Decisão, pois combina várias árvores para reduzir o overfitting.
    - Comparando os dois modelos, o Random Forest tende a ter um **RMSE menor** e um **R² maior**, indicando um melhor ajuste aos dados.
    ''')

    st.divider()

    st.write("### Comparação dos Modelos")
    st.write(f"**RMSE (Árvore de Decisão):** {rmse:.3f}")
    st.write(f"**R² (Árvore de Decisão):** {r2:.3f}")
    st.write(f"**RMSE (Random Forest):** {rmse_rf:.3f}")
    st.write(f"**R² (Random Forest):** {r2_rf:.3f}")
    st.write('''
    - O **Random Forest** superou a Árvore de Decisão em termos de RMSE e R².
    - Isso ocorre porque o Random Forest é um método de ensemble, que combina várias árvores para melhorar a precisão e a generalização.
    ''')


with tab4:
    st.subheader("Comentários")
    st.write("Deixe seu comentário sobre as análises ou o aplicativo:")
    
    comentario = st.text_area("Escreva seu comentário aqui:", height=150)
    
    if st.button("Enviar Comentário"):
        if comentario.strip() == "":
            st.warning("Por favor, escreva um comentário antes de enviar.")
        else:
            st.success("Comentário enviado com sucesso!")
            st.write("Seu comentário foi:", comentario)