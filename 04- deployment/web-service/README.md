## Deploying a model as a web-service

* Creating a virtual environment with Pipenv -- pipenv install and pipenv shell
* Creating a script for predictiong 
* Putting the script into a Flask app
* Packaging the app to Docker

```bash
gunicorn --bind=0.0.0.0:9696 predict:app
```

```bash
docker build -t ride-duration-prediction-service:v1 .
```

```bash
docker run -it --rm -p 9696:9696  ride-duration-prediction-service:v1
```