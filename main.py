from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Loading the dataset
movies_data = pd.read_csv('movie_dataset.csv') #use your path dataset extracted from
# :https://www.kaggle.com/datasets/utkarshx27/movies-dataset

# Preprocessing pdata (example using overview for recommendations)
movies_data['overview'] = movies_data['overview'].fillna('')
movies_data['cast'] = movies_data['cast'].fillna('')  # Ensure cast column is filled
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_data['overview'])

# Calculating cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    query = request.form['query']
    indices = pd.Series(movies_data.index, index=movies_data['original_title']).drop_duplicates()
    idx = indices.get(query)

    if idx is not None:
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]  # Get top 10 recommendations

        # Getting movie indices and similarity scores
        movie_indices = [i[0] for i in sim_scores]
        similarity_scores = [i[1] for i in sim_scores]  # Extract similarity scores

        results = movies_data.iloc[movie_indices]
        results['similarity'] = similarity_scores

        # Extracting cast information
        results['cast'] = results['cast'].apply(lambda x: x.split(',')[:3])  # Get the first 3 cast members for display
    else:
        results = pd.DataFrame(columns=movies_data.columns)  # Return empty DataFrame with correct columns

    return render_template('resultspage.html', query=query, results=results)


if __name__ == '__main__':
    app.run(debug=True)
