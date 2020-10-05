import pickle
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras

app = Flask(__name__)
model = keras.models.load_model("mymodel")
tokenizer=pickle.load(open('transform.pkl','rb'))

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/predict',methods=['POST'])
def predict():
    seed_text = request.form['message']
    next_words = 20
  
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=11-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return render_template('page.html',prediction_text=seed_text)



if __name__ == "__main__":
    app.run(debug=True)