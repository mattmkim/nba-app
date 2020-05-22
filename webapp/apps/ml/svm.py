import joblib
import pandas as pd

class SVM:
    def __init__(self):
        path_to_artifacts = "../../research/"
        self.model = joblib.load(path_to_artifacts + "sv_classifier.joblib")

    def preprocessing(self, input):
        input = pd.DataFrame(input, index=[0])
        input.fillna(self.values_fill_missing)

        return input

    def predict(self, input):
        return self.model.predict_proba(input)

    def postprocessing(self, input):
        return {"probability_no": input[0], "probability_yes": input[1], "status": "OK"}

    def compute_pred(self, input):
        try:
            input = self.preprocessing(input)
            prediction = self.predict(input)[0]
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "error", "message": str(e)}

        return prediction