# index.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# 가상의 데이터 생성 (실제 데이터를 사용해야 함)
data = {
}

df = pd.DataFrame(data)

# Features와 Labels 설정
X = df.drop('price', axis=1)
y = df['price']

# 학습 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 모델 평가 (MSE 예시)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# 모델 저장
joblib.dump(model, 'airbnb_price_prediction_model.joblib')

# 예측 함수 정의


def predict_price(features):
    # 모델 로드
    loaded_model = joblib.load('airbnb_price_prediction_model.joblib')

    # 예측
    prediction = loaded_model.predict([features])

    return prediction[0]


# 가상의 특정 방 정보로 가격 예측
sample_features = {
    '방의위치': 3,
    '방의크기': 75,
    '편의시설': 3,
    '계절': 2
}

predicted_price = predict_price(sample_features)
print(f'예상 가격: ${predicted_price:.2f}')
