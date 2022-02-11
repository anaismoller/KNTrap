import pandas as pd
from flask import Flask
from pathlib import Path
from flask import request, redirect
from flask import render_template

app = Flask(__name__)


@app.route("/")
def upload_image():

    # THIS IS THE LINE TO CHANGE WITH THE FILE YOU WANT
    df = pd.read_csv("../Fink_outputs/selected_20220211.csv")

    list_to_proc = df.id.tolist()

    list_to_proc = [
        f"{f}.difflc.forced.png"
        for f in list_to_proc
        if not (
            Path("annotated") / (f"{f}.difflc.forced.png".replace(".png", ".txt"))
        ).exists()
    ]

    print(list_to_proc)

    # If no valid image file was uploaded, show the file upload form:
    return render_template("results.html", results=list_to_proc,)


@app.route("/form", methods=["POST"])
def my_form_post():
    # text = request.form["text"]
    # processed_text = text.upper()
    for k, v in request.form.items():
        img_name = f"static/{k}"
        txt_name = f"annotated/{k.replace('.png', '.txt')}"
        tags = request.form.getlist("tags")
        tags_str = " ".join(tags)
        print(tags)

        with open(txt_name, "w") as f:
            f.write(f"tags: {tags_str}\n")
            f.write(f"comment: {v}")

    return redirect(request.referrer)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
