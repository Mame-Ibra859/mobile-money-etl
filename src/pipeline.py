from extract import extract
from transform import transform
from detect_fraud import detect_fraud


def main():
    df = extract("data/raw/PS_20174392719_1491204439457_log.csv")
    df = transform(df)
    df = detect_fraud(df)
    df.to_csv("data/processed/fraud_predictions.csv", index=False)


if __name__ == "__main__":
    main()
