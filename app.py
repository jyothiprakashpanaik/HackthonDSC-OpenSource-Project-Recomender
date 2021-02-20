
import pandas as pd
from flask import Flask, request, render_template
from models.model import nltk_model_predict
import json

app = Flask(__name__)

@app.route("/")
def hello():
	return render_template('index.html')

@app.route("/predict",methods=["POST","GET"])
def predict():
	if request.method == "POST":
		lang = request.form.getlist("lang")
		# print(lang)
		fields = request.form.getlist("fields")
		# print(fields)
		lang += fields
		# print(lang)
		model_input = pd.DataFrame([lang])
		# print(model_input)
		model_output = nltk_model_predict(model_input)[0]

		owner_repo_name = []
		repo_description = []
		github_repo_url = []
		count_of_stars = []
		primary_language_name = []
		license_name = []
		repo_created_day = []

		for output in model_output:
			owner_repo_name.append(output["owner_repo_name"])
			github_repo_url.append(output["github_repo_url"])
			repo_description.append(output["repo_description"])
			count_of_stars.append(output["count_of_stars"])
			primary_language_name.append(output["primary_language_name"])
			license_name.append(output["license_name"])
			repo_created_day.append(output["repo_created_day"])

		# print(owner_repo_name)
		# print(github_repo_url)
		# print(repo_description)
		# print(count_of_stars)
		# print(primary_language_name)
		# print(license_name)
		# print(repo_created_day)

		myList = zip(owner_repo_name,github_repo_url,repo_description,count_of_stars,primary_language_name,license_name,repo_created_day)

		return render_template("result.html",context=myList)

	return ("404")

if __name__ == "__main__":
	app.run(debug=True)