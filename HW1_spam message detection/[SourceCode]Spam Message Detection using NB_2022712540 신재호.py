# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import numpy as np


# 나이브 베이즈 분류 모델을 구현
class NaiveBayes:
    
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 라플라스 스무딩을 위한 파라미터

    def fit(self, X, y):
        '''
        나이브베이지안 학습을 위한 함수 (확률 계산)
        param:
            X: 학습 X 데이터셋
            y: 학습 y 데이터셋
        return:
            None        
        '''
        print("===========================================================")
        print("=========== NaiveBayesClassifier Fit Start ================")
        print("===========================================================\n\n")
        self.targets = np.unique(y)  # target 변수 = non-spam :0, spam :1
        self.target_probs = {}      # target 변수의 확률 변수
        self.word_probs = {}     # word별 확률 변수

        total_docs = X.shape[0]
        num_features = X.shape[1]

        for c in self.targets:
            X_c = X[y == c]  # 클래스 c에 해당하는 문서들만 추출 

            # target spam 여부 비율 계산
            self.target_probs[c] = X_c.shape[0] / total_docs
            # 단어 빈도로 확률 계산 (라플라스 스무딩 포함)
            self.word_probs[c] = (X_c.sum(axis=0) + self.alpha) / (X_c.sum() + self.alpha * num_features)

    def predict(self, X):
        '''
        나이브베이지안 모델에서 예측을 하기 위한 함수
        param :
            X : 예측하고자 하는 X 데이터셋
        return : 
           pred_list : 예측 결과 리스트 
        '''
        print("===========================================================")
        print("========= NaiveBayesClassifier predict Start ==============")
        print("===========================================================\n\n")

        pred_list = []

        for x in X.toarray():  #배열로 변환
            probs = {}
            for target in self.targets:
                # 현재 단어의 로그 조건부 확률 x 현재 단어의 등장 횟수로 계산하여 확률 누적
                # 확률 (사전 확률) 로그 값
                log_prob = np.log(self.target_probs[target])
                # 확률 로그 값
                word_probs = self.word_probs[target]

                # 로그 확률을 사용해 예측 확률 계산
                log_probs = np.dot(x, np.log(word_probs).T) + np.dot(1-x, np.log(1-word_probs).T)
                probs[target] = log_prob + log_probs

            # 스팸 / Non spam log 값 저장
            spam_prob = probs[1].item()
            non_spam_prob = probs[0].item()
            
            #로그 값 기준으로 예측값 append
            if spam_prob > non_spam_prob: 
                #예측 확률이 높은 것으로 append
                pred_list.append(1)
            else:
                pred_list.append(0)
        
        return pred_list
    

    def print_matrics(self,y_pred,y_test):
        '''
        결과를 한번에 출력하기 위한 함수
        param:
            y_pred: 모델에서 예측한 y 결과
            y_test: 실제 y 결과
        '''

        # metrics 계산
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print('accuracy : ',accuracy )
        print('precision: ',precision)
        print('recall   : ',recall)
        print('f1       : ',f1)

        # metrics을 위한 데이터프레임 생성
        df_metrics = pd.DataFrame({'Accuracy': [accuracy], 'Precision': [precision], 'Recall': [recall], 'F1 Score': [f1]})
        df_metrics = df_metrics.melt(var_name="Metrics", value_name="Values")

        # Plotting metrics
        plt.figure(figsize=(10, 4))
        sns.barplot(x="Metrics", y="Values", data=df_metrics)
        plt.title('Evaluation Metrics')
        plt.show()

    def show_confusion_mat(self, y_pred, y_test):
        '''
        confusion matrix 출력을 위한 함수
        param:
            y_pred: 모델에서 예측한 y 결과
            y_test: 실제 y 결과
        '''

        conf_mat = confusion_matrix(y_test, y_pred)

        # Plotting confusion matrix
        plt.figure(figsize=(10, 4))
        sns.heatmap(conf_mat, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()



def word_to_none(X_train, x_test):
    '''
    기본 전처리 모델
    '''
    vectorizer = CountVectorizer()

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    print(vectorizer.vocabulary_ )
    print('단어장 수 :' ,len(vectorizer.vocabulary_))

    return X_train_transformed, X_test_transformed

def word_to_vec_best(X_train, x_test):
    '''
    가장 높은 성능의 전처리 
    '''
    vectorizer = CountVectorizer(ngram_range=(1,1),analyzer="word",stop_words=['subject','re' , 's' ,'hpl','hou','enron','etc'])

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)
    
    print(vectorizer.vocabulary_ )
    print('단어장 수 :' ,len(vectorizer.vocabulary_))
    return X_train_transformed, X_test_transformed

def word_to_vec_tfidf(X_train, x_test):
    '''
    전처리 실험 : TFIDF
    accuracy :  0.8653350515463918
    precision:  1.0
    recall   :  0.525
    f1       :  0.6885245901639345
    '''
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer()
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    return X_train_transformed, X_test_transformed

def word_to_vec_ngram(X_train, x_test):
    '''
    전처리 실험 : Ngram 2, 3개
    '''
    vectorizer = CountVectorizer(ngram_range=(1,2))
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    print(vectorizer.vocabulary_ )
    print('단어장 수 :' ,len(vectorizer.vocabulary_))
    
    return X_train_transformed, X_test_transformed

def word_to_vec_split_word(X_train, x_test):
    '''
    전처리 실험 : 단어별로 자르기, 문자로 자르기
    '''

    vectorizer = CountVectorizer(analyzer="char_wb")
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    print(vectorizer.vocabulary_ )

    return X_train_transformed, X_test_transformed

def word_to_vec_stop_word(X_train, x_test):
    '''
    전처리 실험 : stop word 추가하기
    '''

    vectorizer = CountVectorizer(stop_words=['subject','re' , 's' ,'hpl','hou','enron','etc'])
    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    print(vectorizer.vocabulary_ )
    print(vectorizer._parameter_constraints)
    print(len(vectorizer.vocabulary_ ))
    return X_train_transformed, X_test_transformed



if __name__ =='__main__':
    FILE_DIR = "./spam_ham_dataset.csv"

    # 데이터 Read
    df = pd.read_csv(FILE_DIR)

    # label 정리
    df[['label','message']] = df[['label_num','text']]
    df.drop(['Unnamed: 0','label_num','text'],axis=1,inplace=True)


    # 학습 및 테스트를 위한 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.3, random_state=23)

    # 전처리 : Transforming text data into vectors 
    X_train_transformed, X_test_transformed = word_to_vec_best(X_train,X_test)


    print('데이터셋 비율')
    print(X_train.shape,X_test.shape, y_train.shape, y_test.shape,'\n\n')


    # 나이브베이지안 함수 초기화
    naive_bayes = NaiveBayes()

    # 나이브베이지안 학습
    naive_bayes.fit(X_train_transformed, y_train)

    # 결과 예측
    y_pred = naive_bayes.predict(X_test_transformed)

    #모델 평가: 정확도, 정밀도, 재현율, F1 점수를 계산
    naive_bayes.print_matrics(y_pred,y_test)

    # Confusion matrix를 통해 결과 Plot
    naive_bayes.show_confusion_mat(y_pred,y_test)
