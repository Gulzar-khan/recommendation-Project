# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, render_template,request
import pickle
import numpy as numpy
import pandas as pd


top_selling_df = pickle.load(open('top_selling.pkl', 'rb'))
top_books = pickle.load(open('top_books.pkl', 'rb'))

collab_user1 = pickle.load(open('collab_user1.pkl', 'rb'))


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('index.html',
        book_name =list(top_selling_df['Book-Title'].values),
        author_name=list(top_selling_df['Book-Author'].values),
        image=list(top_selling_df['Image-URL-M'].values),
        year=list(top_selling_df['Year-Of-Publication'].values),
        Publisher_name=list(top_selling_df['Publisher'].values),
        Times_Book_Bought=list(top_selling_df['Times_Book_Bought'].values),
                            title=list(top_books['Book-Title'].values),
                           Avg_Rating=list(top_books['Avg_Rating'].values),
                           Score=list(top_books['Score'].values),
                           image2=list(top_books['Image-URL-M'].values),
                           )
@app.route('/collab')
def collab_ui():
    return render_template('collab.html')

@app.route('/collab_1',methods=["post"])
def collab():
    user_input=request.form.get('user_input')
    df_user = collab_user1[collab_user1['user_id']==int(user_input)]
    # print(df_user.head(5))
    # df_pred = df_user[df_user['pred_rating'].notna()].sort_values('pred_rating', ascending=False).head(5)
    #
    df_rating = df_user[df_user['pred_rating'].notna()].sort_values('book_rating', ascending=False).head(15)
    df_rating = df_rating.drop(['user_id', 'Total_No_Of_Users_Rated', 'book_rating', 'Avg_Rating'], axis=1)
    data = []heroku login
    for i in range(len(df_rating) + 1):
        data.extend(df_rating.iloc[i:i + 1].values.tolist())
    print(data)
    return render_template('collab.html',data=data)

# render_template('collab.html',
#                            book_name1 = list(df_rating['book_title'].values),
#                             pred_rating=list(df_rating['pred_rating'].values),
#                             image2=list(df_rating['Image-URL-M'].values),
#                             ISBN=list(df_rating['ISBN'].values)
#                            )
if __name__ == '__main__':
    app.run(debug=True)





# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('Gulzar')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
