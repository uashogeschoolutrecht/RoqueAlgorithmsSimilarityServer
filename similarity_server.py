import model
import json
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
resources = {r"/api/*": {"origins": "*"}}
app.config["CORS_HEADERS"] = "Content-Type"
app.config['JSON_SORT_KEYS'] = False


@app.route('/similarity', methods=['POST'])
def get_similarity():
    """
        Function to get the similarities between the original article and the found articles

        Parameters:
            original_article (string): the original article
            found_article (string): the found article
            original_language (string): the language of the orginal article
            found_language (string): the language of the found article
            type (string): the type of similarity to get {cosine or euclidean}

        Returns:
            string: the similarity score as a string, because flask doesnt want to return a float..
    """
    print(request.form["found_articles"])
    # the parameters can be accessed through `request.form[{parameter name}]`
    original_article = {"article": request.form["original_article"],
                        "language": request.form["original_language"] if request.form[
                                                                             "original_language"] != "UNKOWN" else 'dutch'}
    found_articles = []
    for article in json.loads(request.form["found_articles"]):
      found_articles.append({
            "article": article["article"],
            "language": article["language"] if article["language"] and article["language"] != "UNKNOWN" else "dutch",
            "url": article["url"]
        })
    
    similarities = model.get_similarities(request.form["type"], original_article, found_articles)

    return dict({"similarities" :similarities})

if __name__ == '__main__':
    app.run(host="0.0.0.0")