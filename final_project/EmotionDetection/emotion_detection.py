import requests
import json

WATSON_NLP_URL = "https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict"
HEADERS = {"grpc-metadata-mm-model-id": "emotion_aggregated-workflow_lang_en_stock"}

def emotion_detector(text_to_analyze):
    payload = {"raw_document": {"text": text_to_analyze}}

    try:
        response = requests.post(WATSON_NLP_URL, headers=HEADERS, json=payload)
        response.raise_for_status()
        
        response_json = response.json()
        emotions = response_json.get("emotion_predictions", [{}])[0].get("emotion", {})

        anger = emotions.get("anger", 0.0)
        disgust = emotions.get("disgust", 0.0)
        fear = emotions.get("fear", 0.0)
        joy = emotions.get("joy", 0.0)
        sadness = emotions.get("sadness", 0.0)

        emotion_scores = {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness
        }
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        return {
            "anger": anger,
            "disgust": disgust,
            "fear": fear,
            "joy": joy,
            "sadness": sadness,
            "dominant_emotion": dominant_emotion
        }

    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

if __name__ == "__main__":
    test_text = "I am so happy I am doing this."
    result = emotion_detector(test_text)
    print(result)
