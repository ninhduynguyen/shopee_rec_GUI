"""Library for preprocessing Vietnamese text in E-Commerce
-------
@note   This library provides a set of functions for preprocessing Vietnamese text which is used in the project 2 of the course LDS0_k283.KHTN
@author Duong Nhat Minh
@date   2023-03-29
"""

"""References:
@ref  https://github.com/undertheseanlp/underthesea
@ref  Underthesea-postagging](https://undertheseanlp.com/#!/
"""

"""Import libraries"""
import regex
import underthesea as uts

"""Define global variables"""
stopwords_path      = './Data/vietnamese-stopwords.txt'
# example = '''Áo thun ba lỗ nam tập gym sát nách, áo ba lỗ nam tanktop tập gym thể thao vải cotton thoáng mát co giãn hút mồ hôi'''

"""Define PreprocessLib class"""
class PreprocessLib:
  def __init__(self):
    """Initialize the PreprocessLib class
    """
    self.stopwords_lst  = []

    self.stopwords_lst  = self.load_list(stopwords_path)

  def load_dict(self, file_path, dict_):
    """Load dictionary from file
    Parameters
    ----------
    Arguments:
      file_path {str}   -- [Path to the file]
      dict_     {dict}  -- [Dictionary to be loaded]
    Returns:
      None              -- [The input dictionary (dict_) will be updated]

    """
    with open(file_path, 'r', encoding="utf8") as file:
      lst_ = file.read().split('\n')
      dict_.clear()
      for line in lst_:
        key, value = line.split('\t')
        dict_[key] = str(value)
    return

  def load_list(self, file_path):
    """Load list from file
    Parameters
    ----------
    Arguments:
      file_path {str}   -- [Path to the file]
    Returns:
      lst_      {list}  -- [List to be loaded]
    """
    with open(file_path, 'r', encoding="utf8") as file:
      lst_ = file.read().split('\n')
    return lst_

  def process_text(self, text):
    """Process text
    Parameters
    ----------
    Arguments:
      text {str}          -- [Input text]
    Keyword Arguments:
    Returns:
      document    {str}   -- [Processed text]
    """
    document = text.lower()
    #document = document.replace("'",'')
    #document = regex.sub(r'\.+', ".", document)
    document = regex.sub(r"[',.]", " ", document)
    new_sentence =''
    for sentence in uts.sent_tokenize(document):
      # if not(sentence.isascii()):
      #== CONVERT EMOJICON
      sentence = ''.join(word for word in list(sentence))
      #== DEL Punctuation & Numbers
      pattern = r'(?i)\b[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợíìỉĩịúùủũụưứừửữựýỳỷỹỵđ]+\b'
      sentence = ' '.join(regex.findall(pattern,sentence))
      #== DEL NUMBERS
      sentence = regex.sub(r'\d+', '', sentence)
      #== DEL SPECIAL CHARACTERS (',', '.', '...', '-',':', ';', '?', '%', '_%' , '(', ')', '+', '/', 'g', 'ml')
      sentence = regex.sub(r'[,.;:?%_)(+/-]', ' ', sentence)
      sentence = regex.sub(r'\b[g|kg|ml|cm|dm|m]\b', ' ', sentence)
      
      new_sentence = new_sentence + sentence + '. '
    document = new_sentence  
    #print(document)
    #== DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document

  # Standardize Vietnamese Unicode
  def loaddicchar(self):
    """Load utf8 character
    Parameters
    ----------
    Arguments:
      None
    Returns:
      dict    {str}   -- [utf8 character]
    """
    uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
    unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"
    dic = {}
    char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split('|')
    charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split('|')
    for i in range(len(char1252)):
      dic[char1252[i]] = charutf8[i]
    return dic
 
  # Pass all data through this function to normalize
  def covert_unicode(self, txt):
    """Convert to utf8 character
    Parameters
    ----------
    Arguments:
      text {str}          -- [Input text]
    Keyword Arguments:
    Returns:
      document    {str}   -- [Processed text]
    """
    dicchar = self.loaddicchar()
    return regex.sub(
      r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
      lambda x: dicchar[x.group()], txt)

  def process_special_word(self, text, special_words):
    """Process special word
    Parameters
    ----------
    Arguments:
      text {str}            -- [Text to process]
      special_words {list}  -- [List of special words]
    Returns
      text {str}            -- [Processed text]
    """
    # print('special_words: ', special_words)
    new_text          = ''
    text_lst          = text.split()
    has_special_word = any(word in special_words for word in text_lst)
    if has_special_word:
      # print("Special word found ...")
      i = 0
      while i <= len(text_lst) - 1:
        word = text_lst[i]
        if word in special_words:
          next_idx = i + 1
          if next_idx <= len(text_lst) - 1:
            word = word +'_'+ text_lst[next_idx]
          i = next_idx + 1
        else:
          i = i + 1
        new_text = new_text + word + ' '
    else:
      # print("No special word found!")
      new_text = text

    return new_text.strip()
  
  def process_postag_thesea(self, text):
    """Process postag using underthesea library
    Parameters
    ----------
    Arguments:
      text {str}            -- [Text to process]
    Returns:
      text {str}            -- [Processed text]
    
    @note: Use list comprehension to join, as it is faster and more efficient than loop.
    """
    # Tokenize text
    sentences = [uts.word_tokenize(s.replace('.', ''), format='text') for s in uts.sent_tokenize(text)]
    #== DEL excess blank space
    return regex.sub(r'\s+', ' ', ' '.join(sentences)).strip()

  def remove_stopword(self, text, stopwords=None):
    """Remove stop words
    Parameters
    ----------
    Arguments:
      text {str}        -- [Text to process]
    Keyword Arguments:
      stopwords {list}  -- [List of stop words] (default: {None})
    Returns:
      text {str}        -- [Processed text]
    """
    if stopwords is None:
      stopwords = self.stopwords_lst
    #== REMOVE stop words
    document = ' '.join('' if word in stopwords else word for word in text.split())
    #print(document)
    #== DEL excess blank space
    document = regex.sub(r'\s+', ' ', document).strip()
    return document