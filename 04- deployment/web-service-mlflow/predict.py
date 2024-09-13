import pickle
from flask import Flask, request, jsonify

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import mlflow.pyfunc
import xgboost as xgb


MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("nyc-taxi-expertiment")
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

model_name = "nyc-taxi-regressor"
champion_model_version = client.get_model_version_by_alias(model_name, "champion").version
model_version = champion_model_version
run_id = client.get_model_version_by_alias(model_name, "champion").run_id

path = client.download_artifacts(run_id=run_id, path='dict_vectorizor.bin')
print(f"downloading the dict victorizer and the model from {path}")
with open(path, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)
# To run on gunicorn: gunicorn --bind=0.0.0.0:9696 predict:app
app = Flask('duration-prediction')
def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    X =  xgb.DMatrix(X)
    print('Transforming the features with DV!')
    preds = model.predict(X)
    return preds[0]

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)
    result = {
        'duration': pred,
        'model_version': champion_model_version,
        'run_id': run_id
    }
    return jsonify(str(result))

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696) 