# ====================== Import libraries ====================== #
# !pip install SpeechRecognition
# !pip install pyaudio
# General libraries
import streamlit as st
import speech_recognition as sr
import os
import pandas as pd
import numpy as np
# Gemsim & Cosine Similarity
from gensim import corpora, models, similarities



# ====================== Definitions and functions ====================== #

DataPath                  = './Data/'
# --- For Content-based Filtering ---
ProcessedFileName         = 'Products_ThoiTrangNam_raw_processed.csv'
FinalFileName             = 'Products_ThoiTrangNam_raw_final.csv'
GemsimDictName            = 'gensim_dictionary.dict'
GensimTfidfName           = 'gensim_tfidf.tfidf'
GemsimModelName           = 'gensim_model.model'
FinalFilePath             = os.path.join(DataPath, FinalFileName)
ProcessedFilePath         = os.path.join(DataPath, ProcessedFileName)
RECS_NUM                  = 10
DEF_SIMILARITY_THRESHOLD  = 0.4

USER_ITEM_RECS_NUM        = 5
DEF_RATING_THRESHOLD      = 3.0

USER_ITEM_HIST_NUM        = 20
TOP_USER_WITH_RATING_NUM  = 100

# --- For Collaborative Filtering ---
ProductRatingFileName     = 'Products_ThoiTrangNam_rating_processed.csv'
UserRecFileName           = 'UsrRecMatrix_.csv'
ItemRecFileName           = 'ItemRecMatrix_.csv'
UserRecFilePath           = os.path.join(DataPath, UserRecFileName)
ItemRecFilePath           = os.path.join(DataPath, ItemRecFileName)
ProductRatingFilePath     = os.path.join(DataPath, ProductRatingFileName)


# ====================== Text processing ====================== #
SPECIAL_WORDS = ['không', 'chẳng', 'chả']
import vnmese_txt_preprocess_lib as vtp
preprocess_lib = vtp.PreprocessLib()

def text_preprocessing(text):
  text = preprocess_lib.process_text(text)
  text = preprocess_lib.covert_unicode(text)
  text = preprocess_lib.process_postag_thesea(text)
  text = preprocess_lib.process_special_word(text, SPECIAL_WORDS)
  text = preprocess_lib.remove_stopword(text)
  return text


# ====================== Load data ====================== #
_INPUT  = ['product_id', 'product_name', 'image', 'link', 'product_name_description_processed']
gemsim_dict  = corpora.Dictionary.load(GemsimDictName)
gemsim_tfidf = models.TfidfModel.load(GensimTfidfName)
gemsim_model = similarities.SparseMatrixSimilarity.load(GemsimModelName)
data         = pd.read_csv(FinalFilePath, encoding='utf8')
df           = data[_INPUT]

# --- For Collaborative Filtering ---
"""
@todo: Increase performance without loading data as global variables
@ref: [Srteamlit Optimize Performance](https://docs.streamlit.io/library/api-reference/performance)
"""
df_user   = pd.read_csv(UserRecFilePath, encoding='utf8', header=0)
df_item   = pd.read_csv(ItemRecFilePath, encoding='utf8', header=0)
df_rating = pd.read_csv(ProductRatingFilePath, encoding='utf8', header=0, sep='\t')
# --- Process df_item ---
df_item = df_item.sort_values(by=['product_id'])
df_item_id_name = pd.merge(df_item, df, on='product_id', how='inner')
df_item_id_name['id_name'] = df_item_id_name['product_id'].astype(str) + ' - ' + df_item_id_name['product_name']
df_item_id_name = df_item_id_name[['product_id', 'product_name', 'id_name']]
# --- Process df_user ---
df_user = df_user.sort_values(by=['user_id'])
df_user_id_name = pd.merge(df_user, df_rating, on='user_id', how='inner')
df_user_id_name['id_name'] = df_user_id_name['user_id'].astype(str) + ' - ' + df_user_id_name['user']
df_user_id_name = df_user_id_name[['user_id', 'user', 'id_name']]



# ====================== General Functions ====================== #
def make_clickable(link):
  # target _blank to open new window
  # extract clickable text to display for your link
  # text = link.split('=')[1]
  text = 'link'
  return f'<a target="_blank" href="{link}">{text}</a>'

def takecomand():
  text = ''
  r = sr.Recognizer()
  with sr.Microphone() as source:
    st.write("Tell me your product's ID or description ...")
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)
    try:
      text = r.recognize_google(audio, language="vi-VI")
      st.write("Your input :", text)
    except:
      st.write("Please say again ..")
    return text

# ====================== Product Recommendations ====================== #

# Create class for product recommendations
class ProductRecommendations:
  def __init__(self):
    # Initialize to load data will slow down the app
    # self.gemsim_dict  = corpora.Dictionary.load(FinalFilePath)
    # self.gemsim_tfidf = models.TfidfModel.load(GensimTfidfPath)
    # self.gemsim_model = similarities.SparseMatrixSimilarity.load(GemsimModelPath)
    # self.data         = pd.read_csv(FinalFilePath, encoding='utf8')
    # self.df           = self.data[_INPUT]
    pass
    
  def recommend_products(self, desc_, recs_num=RECS_NUM, threshold=DEF_SIMILARITY_THRESHOLD):
    """_summary_
    Parameters
    ----------
    Arguments:
        desc_     (str): Product ID or description
        recs_num  (int): Number of recommendations
    -------
    Returns:
    list
        List of recommended products
    """
    # Input product_id or description
    input_text  = desc_
    # dict_       = self.gemsim_dict
    # tfidf_      = self.gemsim_tfidf
    # index_      = self.gemsim_model
    # df          = self.df
    dict_ = gemsim_dict
    tfidf_ = gemsim_tfidf
    index_ = gemsim_model
    
    with st.spinner('Searching ...'):
      # Check if input_text is empty
      if input_text == '':
        st.error('Please enter product ID or description')
        # Return empty dataframe
        return pd.DataFrame(columns=['product_id', 'similarity', ])

      # Check if input is product_id or description
      if input_text.isdigit():
        # Input is product_id
        product_id = int(input_text)
        # Check if product_id exists
        if product_id not in df.product_id.values:
          st.error(f'Product ID {product_id} does not exist')
          # Return empty dataframe
          return pd.DataFrame(columns=['product_id', 'similarity', ])
        else:
          # Get product description
          product_description = df[df.product_id == product_id].product_name_description_processed.values[0]
      else:
        # Input is product description
        product_description = input_text

      # Preprocess input text
      processed_description = text_preprocessing(product_description)
      if not input_text.isdigit():
        st.success('Input description after preprocessing: {}'.format(processed_description))
      # Convert to bag of words
      corpus_ = dict_.doc2bow(processed_description.split())
      # Calculate TF-IDF
      corpus_tfidf_ = tfidf_[corpus_]
      # Calculate similarity
      sims = index_[corpus_tfidf_]
      # Sort similarity
      sims = sorted(enumerate(sims), key=lambda item: -item[1])
      # Only get similarity >= threshold
      sims = [s for s in sims if s[1] >= threshold]
      # Get top similar products
      top_similar_products = sims[:recs_num]
    
    # Return top similar products inform of dataframe of product_id, similarity
    df_ = pd.DataFrame(columns=['product_id', 'similarity', ])
    for i, (product_id, similarity) in enumerate(top_similar_products):
      df_ = pd.concat([df_, pd.DataFrame({'product_id': [df.iloc[product_id].product_id], 'similarity': [similarity]})], ignore_index=True)
    return df_
  

  def get_product_info(self, df_, on_):
    """_summary_
    Parameters
    ----------
    Arguments:
        df  (dataframe): Dataframe contains column to merge
        on  (str): Column name to merge
    -------
    Returns:
    list
        Product name, image, link, description
    """
    df_ = pd.merge(df_, df, on=on_)
    return df_
  

  def get_product_info_(self, product_id):
    """_summary_
    Parameters
    ----------
    Arguments:
        product_id  (int): Product ID
    -------
    Returns:
    list
        Product name, image, link, description
    """
    # Check if product_id exists
    if product_id not in df.product_id.values:
      st.error(f'Product ID {product_id} does not exist')
      return None
    else:
      df_ = df[df.product_id == product_id]      
      return df_
    

  """The cached result is returned instead of re-computing the result."""
  @st.cache_data()
  def get_product_id_name_list(_self):
    """ Get list of product_id and product_name
    Parameters
    ----------
    Arguments:
    -------
    Returns:
    list
        List of product_id and product_name
    """
    # Return list of product_id and product_name in a same line
    df['id_name'] = df['product_id'].astype(str) + ' - ' + df['product_name']
    return df['id_name']
  
  
  def get_product_id_name_list_(_self, item_id):
    """ Get list of product_id and product_name
    Parameters
    ----------
    Arguments:
        item_id  (int): Item ID
    -------
    Returns:
    list
        List of product_id and product_name
    """
    # Check if item_id exists
    if item_id not in df.product_id.values:
      return None
    # Return list of product_id and product_name in a same line
    df_ = df[df.product_id == item_id]
    df_['id_name'] = df_['product_id'].astype(str) + ' - ' + df_['product_name']
    return df_['id_name']
  

  def get_top_user_rated_items(self, user_id, num_items=USER_ITEM_HIST_NUM):
    """ Get top user rated items
    Parameters
    ----------
    Arguments:
        user_id  (int): User ID
        num_items  (int): Number of items to return
    -------
    Returns:
    list
        List of user ratings information
    """
    df_rating_ = df_rating[df_rating.user_id == user_id].sort_values(by='rating', ascending=False)[:num_items]
    # get product_name
    df_rating_['product_name'] = df_rating_['product_id'].apply(lambda x: df[df.product_id == x]['product_name'].values[0])
    # get link
    df_rating_['link'] = df_rating_['product_id'].apply(lambda x: df[df.product_id == x]['link'].values[0])
    return df_rating_
  

  def get_top_user_with_rating(self, num_users=TOP_USER_WITH_RATING_NUM):
    """ Get top users with rating
    Parameters
    ----------
    Arguments:
        num_users  (int): Number of users to return
    -------
    Returns:
    list
        List of user ratings information
    """
    # get top users who rated most items, return user_id, user_name and number of ratings in a dataframe
    df_rating_ = df_rating.groupby('user_id').count().sort_values(by='rating', ascending=False)[:num_users]
    df_rating_ = df_rating_.reset_index()
    df_rating_ = df_rating_.rename(columns={'rating': 'num_ratings'}) 
    # get user_name
    df_rating_['user'] = df_rating_['user_id'].apply(lambda x: df_rating[df_rating.user_id == x]['user'].values[0])
    return df_rating_[['user_id', 'user', 'num_ratings']]
    # return df_rating_[['user_id', 'num_ratings']]
  

  """The cached result is returned instead of re-computing the result."""
  @st.cache_data()  
  def get_all_user_ids(_self):
    """ Get list of user_ids
    Parameters
    ----------
    Arguments:
    -------
    Returns:
    list
        List of user_ids
    """
    userIds = df_user['user_id'].unique()
    userIds.sort()
    return userIds
  

  """The cached result is returned instead of re-computing the result."""
  @st.cache_data()
  def get_all_user_ids_names(_self):
    """ Get list of user_ids and user_names based on df_user (collaborative filtering)
    Parameters
    ----------
    Arguments:
    -------
    Returns:
    list
        List of user_ids and user_names
    """
    return df_user_id_name['id_name'].unique()
  
  
  """The cached result is returned instead of re-computing the result."""
  @st.cache_data()
  def get_all_item_ids(_self):
    """ Get list of item_ids
    Parameters
    ----------
    Arguments:
    -------
    Returns:
    list
        List of item_ids
    """
    itemIds = df_item['product_id'].unique()
    itemIds.sort()
    return itemIds
  

  """The cached result is returned instead of re-computing the result."""
  @st.cache_data()
  def get_all_item_ids_names(_self):
    """ Get list of item_ids and item_names based on df_item (collaborative filtering)
    Parameters
    ----------
    Arguments:
    -------
    Returns:
    list
        List of item_ids and item_names
    """
    return df_item_id_name['id_name'].unique()


  def get_rec_user_items(self, user_id, recs_num=USER_ITEM_RECS_NUM, threshold=DEF_RATING_THRESHOLD):
    """ Get list of recommended items for a user
    Parameters
    ----------
    Arguments:
        user_id  (int): User ID
        recs_num (int): Number of recommendations
        threshold (float): Minimum rating to recommend
    -------
    Returns:
    list
        List of recommended items
    """
    # Check if user_id exists
    if user_id not in df_user['user_id'].unique():
      st.error(f'User ID {user_id} does not exist')
      return
    else:
      # Get list of recommended items
      df_ = df_user[df_user.user_id == user_id].sort_values(by='rating', ascending=False)
      df_ = df_[df_.rating >= threshold]
      df_ = df_[:recs_num]
      return df_
    

  def get_rec_item_users(self, product_id, recs_num=USER_ITEM_RECS_NUM, threshold=DEF_RATING_THRESHOLD):
    """ Get list of recommended users for an item
    Parameters
    ----------
    Arguments:
        product_id  (int): Product ID
        recs_num (int): Number of recommendations
        threshold (float): Minimum rating to recommend
    -------
    Returns:
    list
        List of recommended users
    """
    # Check if product_id exists
    if product_id not in df_item['product_id'].unique():
      st.error(f'Product ID {product_id} does not exist')
      return None
    else:
      # Get list of recommended users
      df_ = df_item[df_item.product_id == product_id].sort_values(by='rating', ascending=False)
      df_ = df_[df_.rating >= threshold]
      df_ = df_[:recs_num]
      df_['user'] = df_['user_id'].apply(lambda x: df_rating[df_rating.user_id == x]['user'].values[0])
      # --- For each user_id, get top 5 reated items of that user in df_rating ---
      df_rating_ = pd.DataFrame()
      for user_ in df_['user_id'].unique():
        df_rating_top = df_rating[df_rating.user_id == user_].sort_values(by='rating', ascending=False)
        df_rating_    = pd.concat([df_rating_, df_rating_top[:5]])
      # get product_name
      df_rating_['product_name'] = df_rating_['product_id'].apply(lambda x: df[df.product_id == x]['product_name'].values[0])
      # get link and format as html link
      df_rating_['link'] = df_rating_['product_id'].apply(lambda x: df[df.product_id == x]['link'].values[0])

      return df_, df_rating_