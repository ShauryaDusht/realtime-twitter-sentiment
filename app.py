import streamlit as st
import tweepy
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from textblob import TextBlob
from datetime import datetime

# Download NLTK resources
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Function to clean tweets
def clean_tweet(tweet):
    # Remove URLs, mentions, hashtags, and special characters
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'@\w+', '', tweet)
    tweet = re.sub(r'#\w+', '', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    tweet = re.sub(r'[^\w\s]', '', tweet)
    return tweet.strip()

# VADER sentiment analysis function
def get_vader_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    
    # Determine sentiment based on compound score
    if sentiment_scores['compound'] >= 0.05:
        return 'Positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# TextBlob sentiment analysis function
def get_textblob_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return 'Positive'
    elif analysis.sentiment.polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

# Function to fetch tweets using Twitter API v2
def fetch_tweets_v2(client, keyword, count=100):
    try:
        # Fetch tweets using v2 endpoint
        tweets = client.search_recent_tweets(
            query=f"{keyword} -is:retweet lang:en",
            max_results=min(count, 100),  # API v2 has 100 tweet limit per request
            tweet_fields=["created_at", "public_metrics", "author_id", "text"]
        )
        
        # Extract tweet data
        tweet_data = []
        
        if not hasattr(tweets, 'data'):
            st.warning("No tweets found with the given keyword.")
            return pd.DataFrame()
            
        for tweet in tweets.data:
            tweet_info = {
                'id': tweet.id,
                'created_at': tweet.created_at,
                'author_id': tweet.author_id,
                'text': tweet.text,
                'clean_text': clean_tweet(tweet.text),
                'likes': tweet.public_metrics['like_count'],
                'retweets': tweet.public_metrics['retweet_count'],
                'replies': tweet.public_metrics['reply_count']
            }
            tweet_data.append(tweet_info)
        
        return pd.DataFrame(tweet_data)
    
    except tweepy.TweepyException as e:
        st.error(f"Error fetching tweets: {e}")
        return pd.DataFrame()

# Function to analyze sentiment and update DataFrame
def analyze_sentiment(df):
    if df.empty:
        return df
    
    # Apply sentiment analysis to each tweet
    df['vader_sentiment'] = df['clean_text'].apply(get_vader_sentiment)
    df['textblob_sentiment'] = df['clean_text'].apply(get_textblob_sentiment)
    
    # Calculate sentiment scores
    df['vader_score'] = df['clean_text'].apply(lambda x: SentimentIntensityAnalyzer().polarity_scores(x)['compound'])
    df['textblob_score'] = df['clean_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    
    return df

# Main function to run the app
def main():
    st.set_page_config(
        page_title="Real-Time Twitter Sentiment Analysis",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Real-Time Twitter Sentiment Analysis")
    st.markdown("Analyze sentiment of recent tweets based on keywords")
    
    # Sidebar - API credentials input
    st.sidebar.header("Twitter API v2 Credentials")
    st.sidebar.markdown("""
    **Note:** This app uses Twitter API v2 which requires a Bearer Token.
    [Learn more about Twitter API v2](https://developer.twitter.com/en/docs/twitter-api/getting-started/getting-access-to-the-twitter-api)
    """)
    bearer_token = st.sidebar.text_input("Bearer Token", type="password")
    
    # Sidebar - Search parameters
    st.sidebar.header("Search Parameters")
    keyword = st.sidebar.text_input("Enter keyword to search",'#IPL')
    tweet_count = st.sidebar.slider("Number of tweets to analyze", 10, 100, 50)
    
    # Initialize session state
    if 'analyzed_df' not in st.session_state:
        st.session_state.analyzed_df = pd.DataFrame()
    if 'last_update' not in st.session_state:
        st.session_state.last_update = None

    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Search and analyze tweets")
        analyze_button = st.button("Analyze Tweets")
        
        if analyze_button and bearer_token:
            try:
                # Set up API client with v2 endpoints
                client = tweepy.Client(bearer_token=bearer_token)
                
                with st.spinner(f"Fetching and analyzing tweets for '{keyword}'..."):
                    # Fetch tweets using v2 endpoint
                    df = fetch_tweets_v2(client, keyword, count=tweet_count)
                    
                    if not df.empty:
                        # Analyze sentiment
                        st.session_state.analyzed_df = analyze_sentiment(df)
                        st.session_state.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.success(f"Successfully analyzed {len(df)} tweets!")
                    else:
                        st.warning("No tweets found with the given keyword. Try a different keyword.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.error("If you're seeing authentication errors, please verify your Bearer Token.")
        
        elif analyze_button:
            st.warning("Please enter your Twitter API Bearer Token.")
    
    with col2:
        if st.session_state.last_update:
            st.info(f"Last updated: {st.session_state.last_update}")
    
    # Display results if we have analyzed data
    if not st.session_state.analyzed_df.empty:
        df = st.session_state.analyzed_df
         
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["Summary", "Visualization", "Tweets List"])
        
        with tab1:
            # Summary statistics
            st.subheader("Sentiment Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                # VADER sentiment counts
                vader_counts = df['vader_sentiment'].value_counts()
                st.metric("Total Tweets", len(df))
                st.metric("Positive Tweets (VADER)", 
                         vader_counts.get('Positive', 0),
                         delta=f"{vader_counts.get('Positive', 0)/len(df)*100:.1f}%")
                st.metric("Negative Tweets (VADER)", 
                         vader_counts.get('Negative', 0),
                         delta=f"{vader_counts.get('Negative', 0)/len(df)*100:.1f}%")
                st.metric("Neutral Tweets (VADER)", 
                         vader_counts.get('Neutral', 0),
                         delta=f"{vader_counts.get('Neutral', 0)/len(df)*100:.1f}%")
            
            with col2:
                # TextBlob sentiment counts
                textblob_counts = df['textblob_sentiment'].value_counts()
                st.metric("Positive Tweets (TextBlob)", 
                         textblob_counts.get('Positive', 0),
                         delta=f"{textblob_counts.get('Positive', 0)/len(df)*100:.1f}%")
                st.metric("Negative Tweets (TextBlob)", 
                         textblob_counts.get('Negative', 0),
                         delta=f"{textblob_counts.get('Negative', 0)/len(df)*100:.1f}%")
                st.metric("Neutral Tweets (TextBlob)", 
                         textblob_counts.get('Neutral', 0),
                         delta=f"{textblob_counts.get('Neutral', 0)/len(df)*100:.1f}%")
        
        with tab2:
            st.subheader("Sentiment Visualization")
            col1, col2 = st.columns(2)
            
            with col1:
                # VADER sentiment pie chart
                fig1 = px.pie(
                    values=df['vader_sentiment'].value_counts().values,
                    names=df['vader_sentiment'].value_counts().index,
                    title="VADER Sentiment Distribution",
                    color=df['vader_sentiment'].value_counts().index,
                    color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
                )
                st.plotly_chart(fig1)
            
            with col2:
                # TextBlob sentiment pie chart
                fig2 = px.pie(
                    values=df['textblob_sentiment'].value_counts().values,
                    names=df['textblob_sentiment'].value_counts().index,
                    title="TextBlob Sentiment Distribution",
                    color=df['textblob_sentiment'].value_counts().index,
                    color_discrete_map={'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
                )
                st.plotly_chart(fig2)
            
            # Sentiment score distribution
            fig3 = px.histogram(
                df, x="vader_score", 
                nbins=20,
                title="Distribution of VADER Sentiment Scores",
                labels={"vader_score": "Sentiment Score", "count": "Number of Tweets"},
                color_discrete_sequence=['skyblue']
            )
            st.plotly_chart(fig3)
        
        with tab3:
            st.subheader("Tweets List")
            
            # Filter options
            sentiment_filter = st.multiselect(
                "Filter by sentiment (VADER)",
                options=["Positive", "Negative", "Neutral"],
                default=["Positive", "Negative", "Neutral"]
            )
            
            # Apply filters
            filtered_df = df[df['vader_sentiment'].isin(sentiment_filter)]
            
            # Search box
            search_term = st.text_input("Search in tweets", "")
            if search_term:
                filtered_df = filtered_df[filtered_df['text'].str.contains(search_term, case=False)]
            
            # Display tweets
            for i, row in filtered_df.iterrows():
                sentiment_color = "green" if row['vader_sentiment'] == "Positive" else "red" if row['vader_sentiment'] == "Negative" else "gray"
                
                st.markdown(f"""
                <div style="border:1px solid #ddd; padding:10px; margin-bottom:10px; border-left:5px solid {sentiment_color};">
                    <p><strong>Author ID: {row['author_id']}</strong> • {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>{row['text']}</p>
                    <p>
                        <span style="color:{sentiment_color};"><strong>{row['vader_sentiment']}</strong></span> 
                        (VADER Score: {row['vader_score']:.3f}, TextBlob Score: {row['textblob_score']:.3f})
                        • ❤️ {row['likes']} • 🔄 {row['retweets']} • 💬 {row['replies']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download data option
            st.download_button(
                label="Download Full Analysis Data (CSV)",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name=f"twitter_sentiment_{keyword}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()