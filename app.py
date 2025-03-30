from flask import Flask, render_template, request, session
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from textblob import TextBlob
import matplotlib.pyplot as plt
import io
import base64
import requests
from googlesearch import search 

app = Flask(__name__)
app.secret_key = "your_secret_key"

model = keras.models.load_model("model.h5", compile=False)
scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/overview')
def overview():
    return render_template('overview.html')

@app.route('/tools_used')
def tools_used():
    return render_template('tools_used.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_name = request.form['stock'].strip().upper()
    start_date = '2020-01-01'
    end_date = '2025-12-31'

    try:
        df = yf.download(stock_name + ".NS", start=start_date, end=end_date, progress=False)

        if df.empty or 'Close' not in df.columns:
            return render_template('error.html', message="Invalid stock ticker or no data available.")

        data = df[['Close']]
        dataset = data.values

        if dataset.shape[0] < 60:
            return render_template('error.html', message="Not enough historical data for prediction.")

        scaled_data = scaler.fit_transform(dataset)
        last_60_days = scaled_data[-60:]
        X_test = np.array([last_60_days])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        future_predictions = []
        input_sequence = last_60_days.copy()

        for _ in range(5):
            pred_price = model.predict(np.array([input_sequence]))
            future_price = scaler.inverse_transform(pred_price)[0][0]
            future_predictions.append(float(future_price))
            input_sequence = np.append(input_sequence[1:], pred_price, axis=0)

        session['stock'] = stock_name
        session['predictions'] = [float(p) for p in future_predictions]

        # Generate sentiment analysis data
        news_headlines, sentiment_scores, sentiment_chart = perform_sentiment_analysis(stock_name)
        
        session['news_headlines'] = news_headlines
        session['sentiment_scores'] = sentiment_scores
        session['sentiment_chart'] = sentiment_chart

        return render_template('result1.html', stock=stock_name, pred=future_predictions[0], 
                               news_headlines=news_headlines, sentiment_scores=sentiment_scores, 
                               sentiment_chart=sentiment_chart)

    except Exception as e:
        return render_template('error.html', message=f"Error fetching stock data: {str(e)}")
# @app.route('/result/<int:day>')
# def result(day):
#     if 'predictions' not in session:
#         return render_template('error.html', message="No prediction data found. Please search again.")

#     stock = session.get('stock', 'Unknown')
#     predictions = session['predictions']
#     sentiment_chart = session.get('sentiment_chart', None)

#     if 1 <= day <= 5:
#         return render_template(f'result{day}.html', stock=stock, pred=predictions[day - 1])
#     else:
#         return render_template('error.html', message="Invalid day selected.")
@app.route('/result/<int:day>')
def result(day):
    if 'predictions' not in session or 'news_headlines' not in session or 'sentiment_scores' not in session or 'sentiment_chart' not in session:
         return render_template('error.html', message="No prediction data found. Please search again.")
    
    stock = session.get('stock', 'Unknown')
    predictions = session['predictions']
    news_headlines = session['news_headlines']
    sentiment_scores = session['sentiment_scores']
    sentiment_chart = session['sentiment_chart']

    if 1 <= day <= 5:
        return render_template(f'result{day}.html', stock=stock, pred=predictions[day - 1], 
                               news_headlines=news_headlines, sentiment_scores=sentiment_scores,
                               sentiment_chart=sentiment_chart)
    else:
        return render_template('error.html', message="Invalid day selected.")


def perform_sentiment_analysis(stock_name):
    query = f"{stock_name} stock news site:moneycontrol.com OR site:economictimes.indiatimes.com"
    urls = [url for url in search(query, num_results=5)]

    news_parsed = []
    for url in urls:
        try:
            request = Request(url=url, headers={'user-agent': 'Mozilla/5.0'})
            response = urlopen(request)
            html = BeautifulSoup(response, 'html.parser')

            paragraphs = html.find_all('p')
            for p in paragraphs:
                text = p.get_text()
                if len(text) > 30:
                    news_parsed.append(text)
        except Exception as e:
            print(f"Error fetching news from {url}: {e}")

    sentiments = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for headline in news_parsed:
        analysis = TextBlob(headline)
        if analysis.sentiment.polarity > 0:
            sentiments["Positive"] += 1
        elif analysis.sentiment.polarity < 0:
            sentiments["Negative"] += 1
        else:
            sentiments["Neutral"] += 1

    if sum(sentiments.values()) == 0:
        return news_parsed, sentiments, None

    
    labels = sentiments.keys()
    sizes = sentiments.values()
    colors = ['#4CAF50', '#f44336', '#9E9E9E']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(sizes, labels=labels, colors=colors, startangle=90, wedgeprops={'edgecolor': 'white'}, autopct='%1.1f%%')
    # Draw a circle at the center to make it a donut chart
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)
    
    plt.title(f' {stock_name}')
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    chart_url = base64.b64encode(img.getvalue()).decode('utf8')
    plt.close()
    
   

    return news_parsed, sentiments, chart_url


if __name__ == '__main__':
    app.run(debug=True)
