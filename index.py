"""Main Web Demo Program"""
from flask import Flask, render_template, flash, request
from wtforms import Form, TextField, SelectField, validators
from naive_bayes_final import nb_main
from vectorspace import vecspace_main
from decisiontree import dectree_main

# Flask App config
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

# List of system choice for dropdown menu
SYSTEMS = [('VECSPACE', 'Vector Space'), ('DECTREE',
                                          'Decision Tree'), 
                                          ('NAIVEBAYES', 'Naive Bayes')]

# Class for forms and user input choices
class ReusableForm(Form):
    lyrics_input = TextField('Search', validators=[
        validators.required()], render_kw={"placeholder": "Enter your lyrics..."})
    systems = SelectField(label='System', choices=SYSTEMS)


@app.route("/", methods=['GET', 'POST'])
def main():
    """Main App Function to be Run"""
    form = ReusableForm(request.form)

    ##Print any errors
    print(form.errors)
    if request.method == 'POST':
        if form.validate():
            # Get user input from page
            lyrics = request.form['lyrics_input']
            system_choice = request.form['systems']

            # Empty artist string to be returned
            output_artist = str()

            # Run appropriate system based on user choice
            if system_choice == 'NAIVEBAYES':
                output_artist = nb_main(lyrics)
            elif system_choice == 'DECTREE':
                output_artist = dectree_main(lyrics)
            elif system_choice == 'VECSPACE':
                user_file = open("output.txt", "w")
                user_file.write(lyrics)
                user_file.close()
                output_artist = vecspace_main()
            flash('Your lyrics is most similar to: ' + output_artist)

    return render_template('index.html', form=form)


if __name__ == "__main__":
    app.run()
