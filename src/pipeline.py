import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_data(ticker):
    df = yf.download(ticker, period="10y")
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    return df.dropna()

def build_features(df):
    df["r"] = df["Close"].pct_change()
    df["vol"] = df["r"].rolling(5).std()
    df = df.dropna()
    X = df[["r","vol"]]
    y = df["Target"]
    return X, y

def train_model(X, y):
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=.2,shuffle=False)
    m = RandomForestClassifier(n_estimators=200, max_depth=5)
    m.fit(Xtr,ytr)
    print("Test acc", m.score(Xte,yte))
    return m
