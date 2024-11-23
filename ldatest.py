import pandas as pd
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# 下载NLTK数据（如果还没有下载）
nltk.download('punkt')
nltk.download('stopwords')

# 读取数据
data = pd.read_csv("reviews.csv")  # 假设评论数据保存在reviews.csv中
comments = data['review'].tolist()   # 提取评论列为列表

# 数据预处理
def preprocess(text):
    if not isinstance(text, str):  # 检查是否是字符串
        return []  # 如果不是字符串，返回空列表
    # 分词
    tokens = word_tokenize(text.lower())  # 转为小写并分词
    # 去除停用词和非字母字符
    tokens = [word for word in tokens if word.isalpha() and word not in stopwords.words('english')]
    return tokens

# 处理所有评论
processed_comments = [preprocess(comment) for comment in comments]

# 过滤掉空列表的评论
processed_comments = [comment for comment in processed_comments if comment]

# 创建字典和语料库
dictionary = corpora.Dictionary(processed_comments)  # 创建字典
corpus = [dictionary.doc2bow(comment) for comment in processed_comments]  # 创建语料库

# 训练LDA模型
num_topics = 5  # 设置主题数量
lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

# 输出主题
for idx, topic in lda_model.print_topics(-1):
    print(f"主题 {idx + 1}: {topic}")

# 计算一致性得分（可选）
from gensim.models.coherencemodel import CoherenceModel

coherence_model_lda = CoherenceModel(model=lda_model, texts=processed_comments, dictionary=dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print(f'一致性得分: {coherence_lda}')