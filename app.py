# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from flask import Flask, render_template,request
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')
import string
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer



top_selling_df=pickle.load(open('top_selling.pkl','rb'))
top_books = pickle.load(open('top_books.pkl', 'rb'))
# for collaborative
collab_user1 = pickle.load(open('collab_user1.pkl', 'rb'))

us_canada_book_title = pickle.load(open('us_canada_book_title.pkl', 'rb'))
corr = pickle.load(open('corr.pkl', 'rb'))
books_with_image = pickle.load(open('books_with_image.pkl', 'rb'))

#content base
title_matrix = pickle.load(open('title_matrix.pkl', 'rb'))
features = pickle.load(open('features.pkl', 'rb'))
book4 = pickle.load(open('book4.pkl', 'rb'))

# search base
book_with_tags_image = pickle.load(open('book_with_tags_image.pkl', 'rb'))








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
    df_rating = df_user[df_user['pred_rating'].notna()].sort_values('book_rating', ascending=False).head(20)
    df_rating = df_rating.drop(['user_id', 'Total_No_Of_Users_Rated', 'book_rating', 'Avg_Rating'], axis=1)
    data = []
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


@app.route('/collab2')
def collab_ui2():
    return render_template('collab2.html')
@app.route('/collab_2',methods=["post"])
def collab2():
    us_canada_book_list=list(us_canada_book_title)
    user_input=request.form.get('user_input2')
    indexx= us_canada_book_list.index(user_input)
    corr_coffey_hands = corr[indexx]
    similar_items = list(us_canada_book_title[(corr_coffey_hands < 1.0) & (corr_coffey_hands > 0.9)])

    data1 = []
    for i in similar_items:
        item = []
        temp_df = books_with_image[books_with_image['Book-Title'] == i]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author']))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title']))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Publisher']))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M']))
        data1.append(item)
    return render_template('collab2.html', data=data1)




@app.route('/content')
def collab_ui3():
    return render_template('content.html')


@app.route('/content_1', methods=["POST"])
def content():
    user_input = request.form.get('user_input3')
    title_id_arr = books_with_image[books_with_image['Book-Title'] == user_input].index.values
    title_id = 0
    for i in title_id_arr:
        title_id += int(i)
    # generate sim matrix
    sim_matrix = cosine_similarity(title_matrix, title_matrix)
    # features = vectorizer.get_feature_names()
    top_n = 20
    top_n_idx = np.flip(np.argsort(sim_matrix[title_id]), axis=0)[0:top_n]
    top_n_sim_values = sim_matrix[title_id, top_n_idx]

    # find top n with values > 0
    top_n_idx = top_n_idx[top_n_sim_values > 0]
    scores = top_n_sim_values[top_n_sim_values > 0]

    # find features from the vectorized matrix
    sim_books_idx = books_with_image['Book-Title'].iloc[top_n_idx].index
    words = []
    for book_idx in sim_books_idx:
        try:
            feature_array = np.squeeze(title_matrix[book_idx].toarray())
        except:
            feature_array = np.squeeze(title_matrix[book_idx])
        idx = np.where(feature_array > 0)
        words.append([" , ".join([features[i] for i in idx[0]])])

        # collate results
    res = pd.DataFrame({
        "sim_books": book4['Book-Title'].iloc[top_n_idx].values, "words": words,
        "scores": scores},
        columns=["sim_books", "scores", "words"])  # for print book name which we want to show similar with,
    # in this df use this as first col-"book_title" : books_title['title'].iloc[title_id], and add one more column
    similar_items = res.drop_duplicates(subset=['sim_books']).reset_index()
    temp_df = similar_items.merge(books_with_image, left_on='sim_books', right_on='Book-Title')
    temp_df = temp_df.drop_duplicates('Book-Title').drop(['Book-Title'], axis=1)
    data3 = []
    for i in temp_df.values.tolist():
        data3.append(i)
    return render_template('content.html', data=data3)




@app.route('/search')
def collab_ui5():
    return render_template('search.html')
@app.route('/search_1',methods=["post"])

# def stemming(text):
#     # '''a function which stems each word in the given text'''
#     text = [stemmer.stem(word) for word in text.split()]
#     return " ".join(text)
def search():
    user_input=request.form.get('user_input4')
    text=user_input
    sw = stopwords.words('english')
    # displaying the stopwords
    # np.array(sw)
    # Making necessory function for applying on tags column for better results
    '''a function for removing punctuation'''
    # replacing the punctuations with no space,
    # which in effect deletes the punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    # return the text stripped of punctuation marks
    text = text.translate(translator)
    '''a function for removing the stopword'''
    # removing the stop words and lowercasing the selected words
    text = [word.lower() for word in text.split() if word.lower() not in sw]
    # joining the list of words with space separator
    text = " ".join(text)
    # create an object of stemming function
    stemmer = SnowballStemmer(language='english')
    '''a function which stems each word in the given text'''
    text = [stemmer.stem(word) for word in text.split()]
    res = " ".join(text)

    mask = book_with_tags_image['tags'].str.contains(res, case=False, na=False)
    recom_df=book_with_tags_image[mask]
    data4 = []
    for i in recom_df.values.tolist():
        data4.append(i)
    return render_template('search.html', data=data4)
# render_template('search.html', data=user_input)








@app.route('/about')
def collab_ui4():
    return render_template('about.html')













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
