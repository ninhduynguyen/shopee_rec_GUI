



# ====================== Import libraries ====================== #

import os
import streamlit as st
import pandas as pd
import utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.display.float_format = '{:.2f}'.format


# ====================== Definitions and functions ====================== #
# --- Data path ---
ImgPath = './Picture/'
DataPath                  = './Data/'
ProductFileName           = 'Products_ThoiTrangNam_raw.csv'
ProductRatingFileName     = 'Products_ThoiTrangNam_rating_raw.csv'
BrandImg                  = os.path.join(ImgPath, "Shopee_Logo.png")
RecommendationImg         = os.path.join(ImgPath, "Recommendation_system.png")
CollaUserItemImg          = os.path.join(ImgPath, "CollaborativeFiltering_UserBased_ItemBased.png")
ContentBasedImg           = os.path.join(ImgPath, "content_based_filtering.png")
No_Image_Available        = os.path.join(ImgPath, "No_Image_Available.jpg")
Pie_Chart                 = os.path.join(ImgPath, "pie_chart.png")
Sub_Category              = os.path.join(ImgPath, "sub_category.png")
Rating                    = os.path.join(ImgPath, "Rating.png")
ProductFilePath           = os.path.join(DataPath, ProductFileName)
ProductRatingFilePath     = os.path.join(DataPath, ProductRatingFileName)
# --- For GUI ---
BussinessObjective        = "Business Objective"
ContentBasedFiltering     = "Content-based Filtering"
CollaborativeFiltering    = "Collaborative Filtering"
menu_                     = [BussinessObjective, ContentBasedFiltering, CollaborativeFiltering]
FilterProdDesc            = "Product description"
FilterProdLst             = "Product list"
UserBasedFilter           = "User-based Filtering"
ItemBasedFilter           = "Item-based Filtering"
DEF_SIMILARITY_THRESHOLD  = 0.4
DEF_RATING_THRESHOLD      = 3.0


# ====================== Declarations ====================== #

pr_ = utils.ProductRecommendations()

item_list_userbase = pd.read_csv('item_list_userbase.csv')
item_list_itembase =  pd.read_csv('item_list_itembase.csv')

item_list_userbase = item_list_userbase['0'].tolist()
item_list_itembase = item_list_itembase['0'].tolist()

# ====================== Streamlit GUI & Process ====================== #
def product_info_display(row):
  """Display product info in a grid
  Parameters
  ----------
  row : pandas.core.series.Series
      A row of the dataframe

  Returns
  -------
    None
  """ 
  # --- display product image ---
  if row.image != None:
    # Check if image can be displayed
    try:
      st.image(row.image, width=100, caption=f'ID {int(row.product_id)}', use_column_width='auto')
    except:
      st.image(No_Image_Available, width=100, caption=f'ID {int(row.product_id)}', use_column_width='auto')
  else:
    st.write('No image')
  # --- display product_id and product_name ---
  if row.link:
    # display product_name with maximum lines of 3
    st.markdown('[{}]({})'.format(row.product_name, row.link))
  else:
    st.write(row.product_name)

  return


def handle_cb_search_button_click(desc, rec_nums, threshold, isVoice):
  """Handle search button click
  Parameters
  ----------
  desc : str
      Product description
  rec_nums : int
      Number of recommended products
  isVoice : bool
      Whether the input is from voice or not
      

  Returns
  -------
    None
  """
  if isVoice:
    description = utils.takecomand()
  else:
    description = desc

  # Get info of the product
  if description.isdigit():
    product_info_  = pr_.get_product_info_(int(description))
    if product_info_ is not None:
      product_info_display(product_info_.iloc[0])

  # Get top similar products
  results = pr_.recommend_products(description, rec_nums, threshold)
  # Check if the results is empty
  if results.empty:
    st.error('Không tìm thấy sản phẩm tương tự!')
    return
  # Add separator
  st.markdown('---')
  # Display top similar products found
  st.write('Top {} sản phẩm tương tự >= {}:'.format(results.shape[0], round(threshold, 2)))
  # Get info of the product
  results = pr_.get_product_info(results, 'product_id')
  st.write(results[['product_id', 'similarity', 'product_name', 'product_name_description_processed', 'image']])

  # create a grid with four columns and display the product images in 2 columns with the same size
  col1, col2, col3 = st.columns(3)
  for group_num, group in results.groupby((results.index % 3)):
    with col1 if group_num == 0 else col2 if group_num == 1 else col3:
      for row in group.itertuples():
        product_info_display(row)
  return


def handle_cf_user_search_button_click(user_id, rec_nums, threshold):
  """Handle collaborative user-based search button click
  Parameters
  ----------
  user_id : str
      User ID
  rec_nums : int
      Number of recommended products
  threshold : float
      Rating threshold
      
  Returns
  -------
    None
  """
  # --- Get top rating history of that user ---
  df_rating = pr_.get_top_user_rated_items(user_id)
  with st.expander('Xem lịch sử mua hàng của khách hàng'):
    df_rating['link'] = df_rating['link'].apply(utils.make_clickable)
    df_rating = df_rating.to_html(escape=False)
    st.write(df_rating, unsafe_allow_html=True)

  # --- Get top recommended products ---
  results = pr_.get_rec_user_items(user_id, rec_nums, threshold)
  # Check if the results is empty
  if results.empty:
    st.error('Không tìm thấy sản phẩm tương tự!')
    return
  # Add separator
  st.markdown('---')
  # Display top recommended products found
  st.write('Top {} sản phẩm đề xuất với rating >= {}:'.format(results.shape[0], round(threshold, 2)))
  # Get info of the product
  results = pr_.get_product_info(results, 'product_id')
  st.write(results[['product_id', 'product_name', 'rating', 'image', 'link']])
  # create a grid with four columns and display the product images in 2 columns with the same size
  col1, col2, col3 = st.columns(3)
  for group_num, group in results.groupby((results.index % 3)):
    with col1 if group_num == 0 else col2 if group_num == 1 else col3:
      for row in group.itertuples():
        product_info_display(row)
  return


def handle_cf_item_search_button_click(product_id, rec_nums, threshold):
  """Handle collaborative item-based search button click
  Parameters
  ----------
  product_id : str
      Product ID
  rec_nums : int
      Number of recommended products
  threshold : float
      Rating threshold
      
  Returns
  -------
    None
  """
  # Get top potential users
  results, df_rating = pr_.get_rec_item_users(product_id, rec_nums, threshold)
  # Check if the results is empty
  if results is None:
    st.error('Không tìm thấy khách hàng tiềm năng!')
    return
  # Get item info
  item_info_  = pr_.get_product_info_(product_id)
  product_info_display(item_info_.iloc[0])

  # Add separator
  st.markdown('---')
  # Display top potential users found
  st.write('Top {} khách hàng tiềm năng với rating >= {}:'.format(results.shape[0], round(threshold, 2)))
  st.write(results[['user_id', 'user', 'rating']])
  st.markdown('---')
  # Display rating history of users
  with st.expander('Xem lịch sử đánh giá của khách hàng'):
    df_rating['link'] = df_rating['link'].apply(utils.make_clickable)
    df_rating = df_rating.to_html(escape=False)
    st.write(df_rating, unsafe_allow_html=True)
  return


def content_gui(desc, isVoice=False):
  """Content-based filtering GUI
  Parameters
  ----------
  desc : str
      Product description
  isVoice : bool
      Whether the input is from voice or not

  Returns
  -------
    None
  """
  if desc is None:
    if isVoice:
      description = None
    else:
      # Add text input to input product's ID or description
      description = st.text_input('Mã sản phẩm hoặc mô tả',
                                  help='Điền mã sản phẩm hoặc mô tả ở đây.',
                                  )
  else:
    description = desc

  # Add slider and set default number of recommendations to 5
  # Number input threshold selecting
  col1, col2 = st.columns(2)
  # Use col1 and col2 to display the slider
  rec_nums = col1.slider('Số lượng nhận xét',
                          min_value=1,
                          max_value=10,
                          value=5,
                          step=1)
  threshold = col2.number_input('Mức độ tương tự',
                              min_value=0.0,
                              max_value=1.0,
                              value=DEF_SIMILARITY_THRESHOLD,
                              step=0.05,
                              help='Similarity threshold (0.0 ~ 1.0) to filter products')
  # Add button to search
  search_button = st.form_submit_button(label='Search')
  if search_button:
    handle_cb_search_button_click(description, rec_nums, threshold, isVoice)
  return


def user_gui(user_id):
  """User-based filtering GUI
  Parameters
  ----------
  user_id : str
      User ID

  Returns
  -------
    None
  """
  # Add slider and set default number of recommendations to 5
  # Number input threshold selecting
  col1, col2 = st.columns(2)
  # Use col1 and col2 to display the slider
  rec_nums = col1.slider('Số lượng nhận xét',
                          min_value=1,
                          max_value=5,
                          value=5,
                          step=1)
  threshold = col2.number_input('Đánh giá',
                              min_value=0.0,
                              max_value=5.0,
                              value=DEF_RATING_THRESHOLD,
                              step=0.05,
                              help='Rating threshold (0.0 ~ 5.0) to filter products')
  # Add button to search
  search_button = st.form_submit_button(label='Search')
  if search_button:
    handle_cf_user_search_button_click(user_id, rec_nums, threshold)
  return

def search_items(user_input, item_list):
    return [item for item in item_list if user_input.lower() in item.lower()]

def item_gui(item_id):
  """Item-based filtering GUI
  Parameters
  ----------
  product_id : str
      Product ID

  Returns
  -------
    None
  """
  # Add slider and set default number of recommendations to 5
  # Number input threshold selecting
  col1, col2 = st.columns(2)
  # Use col1 and col2 to display the slider
  rec_nums = col1.slider('Số lượng nhận xét',
                          min_value=1,
                          max_value=5,
                          value=5,
                          step=1)
  threshold = col2.number_input('Đánh giá',
                              min_value=0.0,
                              max_value=5.0,
                              value=DEF_RATING_THRESHOLD,
                              step=0.05,
                              help='Rating threshold (0.0 ~ 5.0) to filter products')
  # Add button to search
  search_button = st.form_submit_button(label='Search')
  if search_button:
    handle_cf_item_search_button_click(item_id, rec_nums, threshold)
  return


def content_based_filtering(filter_option, isVoice=False):
  """Content-based filtering
  Parameters
  ----------
  filter_option : str
      Option to filter products

  Returns
  -------
    None
  """
  if filter_option == FilterProdDesc:
    # Stick widgets
    with st.form(key='my_form'):
      content_gui(None, isVoice)
  elif filter_option == FilterProdLst:
    # Stick widgets
    with st.form(key='my_form'):
      #product_info = st.selectbox("Chọn sản phẩm", pr_.get_product_id_name_list())
      # Extract product_id from product_info
      #product_id = product_info.split(' - ')[0]
      #content_gui(product_id)
      item_id_name = st.text_input("Chọn thông tin sản phẩm:",key="item_id_name")
      submit_botton = st.form_submit_button("Tìm kiếm")
      if submit_botton:
        if item_id_name:
          suggestions = search_items(item_id_name, item_list_itembase)
          if suggestions:
            selected_item = st.selectbox("Chọn một đối tượng:", suggestions)
            if selected_item:
              item_id_name = selected_item
              st.write("Giá trị cuối cùng được chọn:", item_id_name)
            else:
              st.write("Không có item nào theo như bạn đang nhập.")
        item_id = int(item_id_name.split(' - ')[0])
        st.write("phân tích trên item-id: ",item_id)
        # Add slider and set default number of recommendations to 5
        # Number input threshold selecting
        col1, col2 = st.columns(2)
        # Use col1 and col2 to display the slider
        rec_nums = col1.slider('Số lượng nhận xét',
                                      min_value=1,
                                      max_value=5,
                                      value=5,
                                      step=1)
        threshold = col2.number_input('Đánh giá',
                                          min_value=0.0,
                                          max_value=5.0,
                                          value=DEF_RATING_THRESHOLD,
                                          step=0.05,
                                          help='Rating threshold (0.0 ~ 5.0) to filter products')
        handle_cf_user_search_button_click(item_id, rec_nums, threshold)



  
  return

def collaborative_based_filtering(filter_option):
  """Collaborative-based filtering
  Parameters
  ----------
  filter_option : str
      Option to filter products

  Returns
  -------
    None
  """
  if filter_option == UserBasedFilter:
    #item_list_userbase = pr_.get_all_user_ids_names()
    #item_list_userbase = item_list_userbase.tolist()
    # --- Get top user with rating ---
    df_rating = pr_.get_top_user_with_rating()
    with st.expander('Xem top khách hàng và rating tương ứng'):
      st.write(df_rating)

    # Stick widgets
    with st.form(key='my_form'):
      #Select user's ID
      # user_id_name = st.selectbox("Chọn mã khách hàng", pr_.get_all_user_ids_names())
      # user_id = int(user_id_name.split(' - ')[0])
      # user_gui(user_id)
        #item_list_userbase = pr_.get_all_user_ids_names()
        #item_list_userbase = item_list_userbase.tolist()
        user_id_name = st.text_input("Chọn mã khách hàng:",key="user_id_name")
        submit_botton = st.form_submit_button("Tìm kiếm")
        if submit_botton:        
            if user_id_name:
                suggestions = search_items(user_id_name, item_list_userbase)
                if suggestions:
                    selected_item = st.selectbox("Chọn một đối tượng:", suggestions)
                    if selected_item:
                        user_id_name = selected_item
                        st.write("Giá trị cuối cùng được chọn:", user_id_name)
                else:
                    st.write("Không có item nào theo như bạn đang nhập.")
            user_id = int(user_id_name.split(' - ')[0])
            st.write("phân tích trên user-id: ",user_id)
            # Add slider and set default number of recommendations to 5
            # Number input threshold selecting
            col1, col2 = st.columns(2)
            # Use col1 and col2 to display the slider
            rec_nums = col1.slider('Số lượng nhận xét',
                                      min_value=1,
                                      max_value=5,
                                      value=5,
                                      step=1)
            threshold = col2.number_input('Đánh giá',
                                          min_value=0.0,
                                          max_value=5.0,
                                          value=DEF_RATING_THRESHOLD,
                                          step=0.05,
                                          help='Rating threshold (0.0 ~ 5.0) to filter products')
            handle_cf_user_search_button_click(user_id, rec_nums, threshold)
            
                

            
        
    

  elif filter_option == ItemBasedFilter:
    #item_list_itembase = pr_.get_all_item_ids_names()
    #item_list_itembase = item_list_itembase.tolist()
    # Stick widgets
    with st.form(key='my_form'):
      # item_id_name  = st.selectbox("Chọn mã sản phẩm",
      #                         pr_.get_all_item_ids_names(),
      #                         # format_func=lambda x: x.split(' - ')[0],
      #                         )
      # item_id = int(item_id_name.split(' - ')[0])
      # item_gui(item_id)
        #item_list_itembase = pr_.get_all_item_ids_names()
        #item_list_itembase = item_list_itembase.tolist()
        item_id_name = st.text_input("Chọn mã sản phẩm:",key="item_id_name")
        submit_botton = st.form_submit_button("Tìm kiếm")
        if submit_botton:        
            if item_id_name:
                suggestions = search_items(item_id_name, item_list_itembase)
                if suggestions:
                    selected_item = st.selectbox("Chọn một đối tượng:", suggestions)
                    if selected_item:
                        item_id_name = selected_item
                        st.write("Giá trị cuối cùng được chọn:", item_id_name)
                else:
                    st.write("Không có item nào theo như bạn đang nhập.")
            item_id = int(item_id_name.split(' - ')[0])
            st.write("phân tích trên item-id: ",item_id)
            # Add slider and set default number of recommendations to 5
            # Number input threshold selecting
            col1, col2 = st.columns(2)
            # Use col1 and col2 to display the slider
            rec_nums = col1.slider('Số lượng nhận xét',
                                      min_value=1,
                                      max_value=5,
                                      value=5,
                                      step=1)
            threshold = col2.number_input('Đánh giá',
                                          min_value=0.0,
                                          max_value=5.0,
                                          value=DEF_RATING_THRESHOLD,
                                          step=0.05,
                                          help='Rating threshold (0.0 ~ 5.0) to filter products')
            handle_cf_user_search_button_click(item_id, rec_nums, threshold)     
  return


def main():
  # --- Sidebar --- 
  # Add title to the sidebar
  st.sidebar.title('Shopee Recommendation System')
  # Use radio button to choose between content-based filltering and collaborative filtering
  page = st.sidebar.radio('Menu', menu_)
  if page == BussinessObjective:
    st.title('Capstone Project')
    st.subheader("Shopee Recommendation System")
    st.image(BrandImg, width=400)
    # Markdown italic with link
    #st.markdown("*(Data used in this project is from https://shopee.vn/Th%E1%BB%9Di-Trang-Nam-cat.11035567)*")
    st.write("## Giới thiệu Project")
    st.write("Shopee là một hệ sinh thái thương mại “all in one”, trong đó có shopee.vn, là một website thương mại điện tử đứng top 1 của Việt Nam và khu vực Đông Nam Á. Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa. Hãy triển khai Recommendation System cho công ty này.")
    # --- Recommendation system ---
    st.write("## Recommendation system")
    st.write("Recommendation system: Là các thuật toán nhằm đề xuất các item có liên quan cho người dùng. nhiệm vụ chính là tối ưu hóa lượng thông tin khổng lồ nhằm đưa đến cho người dùng những thứ phù hợp nhất.")
    st.image(RecommendationImg, width=800)
    # --- Content-based Filtering ---
    st.write("### Content-based Filtering")
    st.write("Content-based Filtering: Gợi ý các item dựa vào hồ sơ (profiles) của người dùng hoặc dựa vào nội dung/thuộc tính (attributes) của những item tương tự như item mà người dùng đã chọn trong quá khứ.")
    st.image(ContentBasedImg, width=800)
    # --- Collaborative Filtering ---
    st.write("### Collaborative Filtering")
    st.write("Collaborative Filtering: Hay còn gọi là lọc tương tác, sử dụng sự tương tác qua lại trong hành vi mua sắm giữa các khách hàng để tìm ra sở thích của một khách hàng đổi với một sản phẩm.")
    st.write("Bao gồm:")
    st.markdown("・*User-based Collaborative Filtering:* Ý tưởng là phân chia các Users tương tự nhau vào chung một nhóm. Nếu một User bất kỳ trong nhóm thích một Item nào đó thì Item đó sẽ được đề xuất cho toàn bộ các Users khác trong nhóm đó.")
    st.markdown("・*Item-based Collaborative Filtering:* Ý tưởng là phân chia Items tương tự nhau vào chung một nhóm. Nếu một User thích bất kỳ một Item nào trong nhóm đó thì tất cả các Item còn lại trong cùng nhóm sẽ được đề xuất cho User đó.")
    st.image(CollaUserItemImg, width=800)
    # --- Đọc dữ liệu và EDA ---
    st.write("## Đọc dữ liệu và EDA")
    df1 = pd.read_csv(ProductFilePath, encoding='utf8', header=0)
    st.write("Đọc dữ liệu Products_ThoiTrangNam_raw:")
    st.write(df1)
    st.write("Mô tả dữ liệu: ")
    a = df1.describe()
    st.write(a)
    b = df1['sub_category'].value_counts()
    st.write("Phân loại hàng hóa: ")
    st.write(b)
    st.write("Số lượng hàng hóa theo sub_category: ")
    st.image(Sub_Category, width=800)
    st.write("Biểu đồ trực quan tỷ lệ các sản phẩm: ")
    st.image(Pie_Chart, width=800)
    
    df2 = pd.read_csv(ProductRatingFilePath, encoding='utf8', header=0, sep='\t')
    st.write("Đọc dữ liệu Products_ThoiTrangNam_rating_raw:")
    st.write(df2)
    st.write("Mô tả dữ liệu: ")
    c = df2.describe()
    st.write(c)
    st.write("Số lượng hàng hóa theo rating: ")
    st.image(Rating, width=800)

  # --- Content-based Filtering ---
  elif page == ContentBasedFiltering:
    desc_ = None
    st.title('Capstone Project')
    st.subheader("Shopee Recommendation System")
    st.image(ContentBasedImg, width=700)
    st.write("# Content-based Filtering")
    # Add select box to choose between product list and manual input
    option = st.sidebar.selectbox("Select option for fitler", [FilterProdDesc, FilterProdLst])
    if option == FilterProdDesc:
      isVoice = st.sidebar.checkbox('By voice (Vnmese) 🎙️')
    
    if option == FilterProdDesc:
      content_based_filtering(option, isVoice)
    elif option == FilterProdLst:
      content_based_filtering(option)

  # --- CollaborativeFiltering ---
  elif page == CollaborativeFiltering:
    st.title('Data Science Capstone Project')
    st.subheader("Shopee Recommendation System")
    st.image(CollaUserItemImg, width=700)
    st.write("# Collaborative Filtering")
    # Add select box to choose between product list and manual input
    option = st.sidebar.selectbox("Select option for fitler", [UserBasedFilter, ItemBasedFilter])
       # Add separator
    st.sidebar.markdown('---')
    # A brief description about the search engine
    st.sidebar.title('Techniques')
    st.sidebar.write("Algorithm: **Alternating Least Squares (ALS)**")
    st.sidebar.subheader('Cross-Validator')
    rank_       = [10, 40]
    max_iter_   = [10]
    reg_param_  = [0.01, 0.1]
    alpha_      = [1.0]
    numFolds_   = 3
    ParamMaps = { 'Parameter': ['rank', 'maxIter', 'regParam', 'alpha', 'numFolds'],
                  'Value': [rank_, max_iter_, reg_param_, alpha_, numFolds_]
                }
    st.sidebar.write(pd.DataFrame(ParamMaps))
    st.sidebar.subheader('Hyperparameters (RMSE = 1.18)')
    df_hyper = pd.DataFrame({'rank': ['40',],
                            'maxIter': ['10',],
                            'regParam': ['0.1',],
                            'alpha': ['1.0',],})
    st.sidebar.table(df_hyper)
    #st.sidebar.markdown(':blue[***Note:***] The hyperparameters are chosen by using cross-validation with 3 folds.\
                        #Increasing value of **rank** will increase the accuracy of the model but will also increase the training time and memory usage.\
                        #Pay attention to this parameter to not make the model overfitting.')
    collaborative_based_filtering(option)


# ====================== Main ====================== #
if __name__ == "__main__":
  main()
