from flask import Flask, render_template,request,redirect, url_for, flash
#from flask_uploads import UploadSet, configure_uploads,DOCUMENTS
import os
import rake
import TextRank
import multi_senti_func
import multi_prod_func
import pandas as pd
from werkzeug.utils import secure_filename
from sentiment import *
from amazon_review_crawler import *

uploadFolder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
import summary_LSA
import preprocessing
import text_rank_summary

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
allowedTypes = set(['xlsx'])
app = Flask(__name__, template_folder=tmpl_dir)
app.secret_key="my_secret_key"
#docs=UploadSet('datafiles',DOCUMENTS)
app.config['UPLOAD_FOLDER']=uploadFolder
#configure_uploads(app,docs)

def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowedTypes

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/keyphrase_extraction")
def keyphrase_extraction():
    return render_template("keyphrase.html")
@app.route("/keyphrase_extraction/textrank")
def keyphrase_extraction_textrank():
    return render_template("textrank.html")

@app.route("/summarization")
def text_summarization():
    return render_template("summarization.html")

@app.route("/summarization/textrank")
def text_summarization_textrank():
    return render_template("summarization_textrank.html")

@app.route("/get_summarization", methods = ['POST'])
def get_summarization():
    text = request.form['text']
    max_length_of_summary = request.form["max_length_of_summary"]
    number_of_concept = request.form["number_of_concept"]
    is_tfidf = request.form["is_tfidf"]
    split_long_sentence = request.form["split_long_sentence"]
    lsa_result = summary_LSA.summarize(text,l=int(max_length_of_summary),k=int(number_of_concept),tfidf=is_tfidf,to_split_length=int(split_long_sentence))
    context = dict()
    context['summarization'] = lsa_result
    return render_template("summarization_result.html", **context)

@app.route("/get_summarization_textrank", methods = ['POST'])
def get_summarization_textrank():
    text = request.form["text"]
    textRank_result = text_rank_summary.extract_sentences(text)
    context = dict()
    context["summarization"] = textRank_result
    return render_template("summarization_text_rank_result.html", **context)

@app.route('/upload',methods=['GET','POST'])
def upload_file():
  if request.method == 'POST':
    print(request.files)
    f = request.files['file']
    if f and allowed_file(f.filename):
      f.save(secure_filename(f.filename))
      return render_template('upload.html', results = 'file uploaded successfully.')
    else:
      return render_template('upload.html', results = 'file not uploaded.')






#nitesh
@app.route("/sentiment_analysis_english")
def sentiment_analysis():
    return render_template("sentiment.html")

@app.route("/get_sentiment_score",methods=['POST'])
def get_sentiment_score():
    text = request.form['text']
    title = request.form['title']
    #print("Get Form data")
    t_score, s_score = get_sentiment(text, title)
    context = dict() #input back
    context['text'] = text #result
    context['title'] = title
    context['text_score']= t_score
    context['summary_score']= s_score
    return render_template("sentiment_score.html", **context)


def saveReviews(product_name, reviews):
    try:
        df1 = pd.DataFrame(reviews)
        writer = pd.ExcelWriter(product_name + '.xlsx')
        df1.to_excel(writer)
        writer.save()
    except:
        print("Error saving file")

def getAggregatedScores(reviews):
    agg_title_score = 0
    agg_text_score = 0
    agg_hybrid_score = 0
    total = 0

    for review in reviews:
        agg_title_score += review['title_score']
        agg_text_score += review['text_score']
        agg_hybrid_score += review['hybrid_score']

        total += 1

    agg_title_score /= total
    agg_text_score /= total
    agg_hybrid_score /= total

    return agg_title_score, agg_text_score, agg_hybrid_score


@app.route("/upload_sentiment_bulk", methods=['POST'])
def upload_sentiment_bulk():
    if request.method == 'POST':
        f = request.files['file']
        if f and allowed_file(f.filename):
          f.save(secure_filename(f.filename))
  
    product_id = request.form['product_id']  
    df = pd.read_excel(f.filename)

    columns = {'reviewText':'text','overall':'stars','summary':'title'}
    df1=df.loc[df.productID==product_id,columns.keys()]
    df1.rename(index=str, columns=columns,inplace=True)
    reviews=list(df1.T.to_dict().values())[:20]
    
    context=dict()

    if len(reviews) == 0:
        context['error'] = 2
        return render_template("bulk_error.html",**context)

    reviews = get_sentiment_bulk(reviews)

    agg_title_score, agg_text_score, agg_hybrid_score = getAggregatedScores(reviews)

    context['product_id']=product_id
    context['reviews']=reviews

    context['agg_title_score'] = agg_title_score
    context['agg_text_score'] = agg_text_score
    context['agg_hybrid_score'] = agg_hybrid_score

    return render_template("upload_bulk_sentiment_result.html",**context)


@app.route("/get_crawler_sentiment_score", methods=['POST'])
def get_crawler_sentiment_score():
    product_url=request.form['product_url']
    num_reviews=int(request.form['num_reviews'])
    reviews, product_id, product_name, message = extractReviews(product_url, num_reviews)
    reviews = reviews[:num_reviews]

    context=dict()

    if len(reviews) == 0:
        print("No reviews fetched")
        context['error'] = 1
        return render_template("bulk_error.html",**context)

    saveReviews(product_name, reviews)

    reviews = get_sentiment_bulk(reviews)
    agg_title_score, agg_text_score, agg_hybrid_score = getAggregatedScores(reviews)

    context['message']=message
    context['product_name']=product_name
    context['reviews']=reviews
    
    context['agg_title_score'] = agg_title_score
    context['agg_text_score'] = agg_text_score
    context['agg_hybrid_score'] = agg_hybrid_score

 
    return render_template("bulk_sentiment_result.html",**context)


#nitesh
@app.route("/crawler")
def crawler():
    return render_template("crawler.html")

@app.route("/get_amazon_file", methods=['POST'])
def get_file():
    product_url=request.form['product_url']
    num_reviews=int(request.form['num_reviews'])
    reviews, product_id, product_name, message = extractReviews(product_url, num_reviews)
    reviews = reviews[:num_reviews]

    # saveReviews(product_name, reviews)
    context=dict()
    context['message']=message
    context['product_name']=product_name
    context['reviews']=reviews
    return render_template("crawler_result.html",**context)


#yunsi
@app.route("/sentiment_analysis_multilingual")
def multilingual_analysis():
    return render_template("multi_senti.html")
#yunsi
@app.route("/get_senti_infor",methods=['POST'])
def get_senti_infor():
    text = request.form['text']
    title = request.form['title']
    #print("Get Form data")
    trans_text = multi_senti_func.translation_to_eng(text)
    trans_title = multi_senti_func.translation_to_eng(title)
    text_token = multi_senti_func.tokenization(trans_text)
    title_token = multi_senti_func.tokenization(trans_title)
    text_pos = multi_senti_func.pos_mark(text_token)
    title_pos = multi_senti_func.pos_mark(title_token)
    wn_text_senti_score = multi_senti_func.senti_score(text_pos)
    wn_title_senti_score = multi_senti_func.senti_score(title_pos)
    vander_text_senti_score = multi_senti_func.vader_senti_score(trans_text)
    vander_title_senti_score = multi_senti_func.vader_senti_score(trans_title)
    adjusted_score = multi_senti_func.adjusted_score(wn_text_senti_score,wn_title_senti_score,vander_text_senti_score,vander_title_senti_score)
    context = dict() #input back
    context['text'] = text #result
    context['title'] = title
    context['trans_text'] = trans_text #result
    context['trans_title'] = trans_title
    context['wn_text_senti_score']= wn_text_senti_score
    context['wn_title_senti_score']= wn_title_senti_score
    context['vander_text_senti_score']= vander_text_senti_score
    context['vander_title_senti_score']= vander_title_senti_score
    context['adjusted_score']= adjusted_score
    #print("Context: ",context)
    return render_template("multi_senti_score.html", **context)

#yunsi
@app.route("/survey_senti",methods=['POST'])
def survey_senti():
    before_sur = request.form['before_sur']
    after_sur = request.form['after_sur']
    #print("Get Form data")
    trans_bef = multi_senti_func.translation_to_eng(before_sur)
    trans_aft = multi_senti_func.translation_to_eng(after_sur)
    bef_token = multi_senti_func.tokenization(trans_bef)
    aft_token = multi_senti_func.tokenization(trans_aft)
    bef_pos = multi_senti_func.pos_mark(bef_token)
    aft_pos = multi_senti_func.pos_mark(aft_token)
    wn_bef_senti_score = multi_senti_func.senti_score(bef_pos)
    wn_aft_senti_score = multi_senti_func.senti_score(aft_pos)
    context = dict() #input back
    context['before_sur'] = before_sur #result
    context['after_sur'] = after_sur
    context['trans_bef'] = trans_bef #result
    context['trans_aft'] = trans_aft
    context['wn_bef_senti_score']= wn_bef_senti_score
    context['wn_aft_senti_score']= wn_aft_senti_score
    #print("Context: ",context)
    return render_template("multi_survey_score.html", **context)

#yunsi
@app.route("/product_senti",methods=['POST'])
def product_senti():
    filenames = request.form['name']
    df = pd.read_excel(filenames)
    #print("Get Form data")
    text_t = multi_prod_func.translation_to_eng(df['text'])
    title_t = multi_prod_func.translation_to_eng(df['title'])
    text_token = multi_prod_func.tokenization(text_t)
    title_token = multi_prod_func.tokenization(title_t)
    text_tag = multi_prod_func.pos_mark(text_token)
    title_tag = multi_prod_func.pos_mark(title_token)
    text_tag = multi_prod_func.pos_mark(text_token)
    title_tag = multi_prod_func.pos_mark(title_token)
    text_sentiment = multi_prod_func.senti_score(text_tag)
    title_sentiment = multi_prod_func.senti_score(title_tag)
    df['text_Sentiment_Score'] = text_sentiment
    df['title_Sentiment_Score'] = title_sentiment
    vader_text_score = multi_prod_func.vader_senti_score(text_t)
    vader_title_score = multi_prod_func.vader_senti_score(title_t)
    df['Vader_text_score'] = vader_text_score
    df['Vader_title_score'] = vader_title_score
    product_score = multi_prod_func.adj_score_avg(df)
    context = dict() #input back
    context['product_score'] = product_score #result
    #print("Context: ",context)
    return render_template("multi_product.html", **context)

@app.route('/get_keyphrases',methods=['POST'])
def get_keyphrases():
    stoppath = "SmartStoplist.txt"
    filename=request.form['name']
    surveys=pd.read_excel(filename,header=0)
    col_name=request.form['question']
    filter_by=request.form['filter_by']
    text=""
    #col=surveys[col_name]
    product_id=request.form['product_id']
    if product_id!='':
        df=surveys.loc[surveys[filter_by]==product_id]
        col=df[col_name]
    else:
        col=surveys[col_name]
    for i in range(len(col)):
        text=text+" "+col[i]
    print(text)
    min_char_length=request.form['min_char_length']
    min_words_length=request.form['min_words_length']
    max_words_length=request.form['max_words_length']
    min_keyword_frequency=request.form['min_keyword_frequency']
    trade_off=request.form['trade_off']
    top_n=request.form['top_n']
    rake_object = rake.Rake(stoppath,int(min_char_length),int(min_words_length),int(max_words_length),int(min_keyword_frequency),1,3,2)
    keywords_score, keywords_counts, stem_counts = rake_object.run(text, float(trade_off),int(top_n))
    context = dict()
    context['keywords_score'] = keywords_score
    context['keywords_counts'] = keywords_counts
    context['stem_counts']= stem_counts
    return render_template("keyphrase_result.html", **context)

@app.route('/get_keyphrases_textrank',methods=['POST'])
def get_keyphrases_textrank():
    filename=request.form['textrank_name']
    top_n=request.form['top_n_textrank']
    surveys = pd.read_excel(filename, header=0)
    col_name = request.form['textrank_question']
    text = ""
    col = surveys[col_name]
    for i in range(len(col)):
        text = text + " " + col[i]
    top_keywords=TextRank.extractKeyphrases(text,int(top_n))
    context=dict()
    context['keywords']=top_keywords
    return render_template("keyword_textrank.html",**context)

if __name__=="__main__":
    app.run(debug=True)
