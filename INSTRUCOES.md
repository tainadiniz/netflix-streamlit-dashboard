# Netflix Exec Dashboard — Reed Hastings

Projeto completo em **5 etapas** focado em **Data Storytelling** para decisões executivas.

## Como rodar
1. Crie um ambiente virtual (opcional) e instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
2. (Opcional) Configure o Kaggle para baixar automaticamente o dataset:
   - Gere o `kaggle.json` no Kaggle (Account → Create New API Token).
   - Salve em `~/.kaggle/kaggle.json` (ou `%USERPROFILE%\.kaggle\kaggle.json` no Windows).
3. Execute o app:
   ```bash
   streamlit run dashboard_netflix.py
   ```

## Dados
- **Principal**: `shivamb/netflix-shows` (arquivo: `netflix_titles.csv`).
- **Avaliações**: crie um `data/ratings.csv` com colunas `title` e `score` (0–10).
  - O app também entende colunas como `imdb_score`, `tmdb_score`, `averageRating` e normaliza se vierem em 0–100.

## Filtros
- País (multiselect)
- Gênero (multiselect)
- Ano de lançamento (range)
- Faixa de avaliação (se `ratings.csv` estiver presente)

## Visualizações
- **Mapa coroplético**: distribuição de títulos por país
- **Barras horizontais**: gêneros mais frequentes
- **Linha temporal**: lançamentos por ano
- **Heatmap**: País × Gênero
- **Nuvem de palavras**: descrições
- **Top 10 por avaliação / Histograma / Scatter**: habilitados quando `ratings.csv` estiver presente
