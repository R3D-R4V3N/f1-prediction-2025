from race_predictor import predict_race

if __name__ == "__main__":
    results = predict_race("Chinese Grand Prix")
    print(results[["Driver", "Team", "Grid", "Final_Position"]])
