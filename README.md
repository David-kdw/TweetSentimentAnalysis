# Tweet Sentiment Analysis

Ce projet comprend un modèle BERT personnalisé pour l'analyse de sentiment des tweets, avec une démonstration à l'aide de Gradio et une API web à l'aide de FastAPI.

## Structure du projet

git clone https://github.com/David-kdw/TweetSentimentAnalysis.git
cd TweetSentimentAnalysis
pip install -r requirements.txt

## Entraînement du modèle
> Exécutez le script train_model.py pour entraîner et sauvegarder le modèle :
python train_model.py
> il vous est proposé de télécharger les poids du modèle entrainé au lien suivant https://drive.google.com/file/d/1Q3OTkE4M2kIPIZWH5viIwG3gZQZHJiJl/view?usp=drive_link

## Démonstration avec Gradio
> Exécutez le script gradio_demo.py pour lancer la démonstration interactive

python gradio_demo.py

## API avec FastAPI
> Exécutez le script fastapi_app.py pour lancer l'API :

python fastapi_app.py

L'API sera disponible sur http://127.0.0.1:8989.

