from flask import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
data = pd.read_csv("conv.csv")

@app.route("/", methods=["GET", "POST"])
def home():
    chat = ""

    if request.method == "POST":
        old_chat = request.form.get("chat", "")
        qts = request.form.get("qts", "").strip().lower()

        texts = [qts] + data["question"].str.lower().tolist()

        cv = CountVectorizer()
        vector = cv.fit_transform(texts)

        cs = cosine_similarity(vector)
        score = cs[0][1:]
        data["score"] = score * 100

        result = data.sort_values(by="score", ascending=False)
        result = result[result.score > 10]

        if len(result) == 0:
            msg = "Sorry 😔 I don't understand."
        else:
            msg = result.head(1)["answer"].values[0]

        new_chat = "You: " + qts + "\nBot: " + msg
        chat = old_chat + "\n\n" + new_chat

        return render_template("home.html", msg=msg, chat=chat.strip())
    return render_template("home.html", chat="")

