from gensim.models import LdaModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.corpora.dictionary import Dictionary
from sentence_transformers import SentenceTransformer
import pandas as pd
import re
from nltk.corpus import movie_reviews
from top2vec import Top2Vec
from bertopic import BERTopic
import torch
import torch.nn as nn
import torch.optim as optim
from ast import literal_eval
import numpy as np
from Net import Net
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import math
import pickle
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt


class DocPreprocess:
    def __init__(self):
        self.data = pd.read_csv('Total.csv')
        self.result_path = 'featureData'
        self.Y = []
        self.load_Y()
        X = []
        doc2vec = self.doc2vec(200)
        print('doc2vec done!')
        BERTopic_X = self.BERTopic()
        print('BERTopic done!')
        top2vec_X = self.top2vec()
        print('top2vec done!')
        SBERT_X = self.SBERT()
        print('SBERT done!')

        for b, t1, t2, t3, t4 in zip(doc2vec, self.pca(BERTopic_X, 20), self.pca(top2vec_X, 20), self.pca(SBERT_X, 20), self.LDA(20)):
            X.append(b + t1 + t2 + t3 + t4)
        self.store_result(X, "combination")
        print('combination done')
        X.clear()
        for b, t in zip(doc2vec, self.pca(BERTopic_X, 80)):
            X.append(b + t)
        self.store_result(X, "BertTopic+doc2vec")
        print('BertTopic+doc2vec done')
        X.clear()
        for b, t in zip(doc2vec, self.pca(top2vec_X, 80)):
            X.append(b + t)
        self.store_result(X, "top2vec+doc2vec")
        print('top2vec+doc2vec done')
        X.clear()
        for b, t in zip(doc2vec, self.pca(SBERT_X, 80)):
            X.append(b + t)
        self.store_result(X, "SBERT+doc2vec")
        print('SBERT+doc2vec done')
        X.clear()
        for b, t in zip(doc2vec, self.LDA(80)):
            X.append(b + t)
        self.store_result(X, "LDA+doc2vec")
        print('LDA+doc2vec done')
        X.clear()

    def top2vec(self):
        # nltk_corpus = self.remove_mark(movie_reviews.words())
        corpus = self.data['review'].values
        # model = Top2Vec(corpus, workers=6, speed='deep-learn')
        model = Top2Vec(corpus, workers=12, speed='learn', min_count=50)

        topic_size, _ = model.get_topic_sizes()
        print(f'topic number: {topic_size}')
        topic_num = len(topic_size)
        X = [[0 for j in range(topic_num)] for i in range(len(corpus))]
        for topic_id in range(topic_num):
            _, document_scores, document_ids = model.search_documents_by_topic(topic_num=topic_id,
                                                                               num_docs=topic_size[topic_id])
            for score, doc_id in zip(document_scores, document_ids):
                X[doc_id][topic_id] = score

        return X

    def BERTopic(self):
        corpus = self.data['review'].values
        topic_model = BERTopic(calculate_probabilities=True, n_gram_range=(1, 2), min_topic_size=50)
        topics, prTopic = topic_model.fit_transform(corpus)
        print(f'topic size: {len(topics)}')
        X = prTopic.tolist()

        # self.store_result("BERTopic")
        return X

    def doc2vec(self, size=200):
        X = []
        paragraph_list = self.splitText(self.data.iterrows())

        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(paragraph_list)]
        model = Doc2Vec(documents, vector_size=size, window=2, min_count=1, workers=4)

        for para in paragraph_list:
            X.append(list(model.infer_vector(para)))

        # self.store_result(f'doc2vec_{size}')
        return X

    def SBERT(self):
        X = []
        model = SentenceTransformer('paraphrase-mpnet-base-v2')
        sentences = self.data['review'].values

        model.max_seq_length = 512

        embeddings = model.encode(sentences)

        for em in embeddings.tolist():
            X.append(em)

        # self.store_result('SBERT')
        return X

    def LDA(self, num_topic):
        X = []
        nltk_corpus = list(map(lambda x: self.remove_mark(x), movie_reviews.sents()))
        documents = Dictionary(nltk_corpus)
        nltk_corpus = [documents.doc2bow(text) for text in nltk_corpus]
        lda = LdaModel(nltk_corpus, num_topics=num_topic)

        paragraph_list = self.splitText(self.data.iterrows())
        # documents = Dictionary(paragraph_list)
        corpus = [documents.doc2bow(text) for text in paragraph_list]
        for row in corpus:
            X.append([i[1] for i in lda.get_document_topics(row, minimum_probability=0)])

        # self.store_result(f'LDA_{num_topic}', X)
        return X

    @staticmethod
    def pca(X, size):
        if len(X[0]) <= size:
            return X
        transformer = SparsePCA(n_components=size, random_state=2)
        transformer.fit(X)
        return transformer.transform(X).tolist()

    @staticmethod
    def splitText(rows):
        paragraph_list = []
        for index, row in rows:
            result = []
            tem = re.split('[,?.!:;()"]+', row['review'])
            for i in tem:
                inner = re.split('[ ]+', i)
                for j in inner:
                    if re.sub('[ ]+', '', j):
                        result.append(j)

            paragraph_list.append(result)
        return paragraph_list

    @staticmethod
    def remove_mark(str_list):
        re_list = []
        mark_list = ',?.!:;()\\\n"' + "'"
        for word in str_list:
            if word.strip() not in mark_list:
                re_list.append(word.strip())
        return re_list

    def store_result(self, X, name):
        df = pd.DataFrame(data={
            'x': X,
            'y': self.Y
        })
        df = df.sample(frac=1)
        df.to_csv(f'{self.result_path}/{name}.csv', index=False)

    def load_Y(self):
        for index, row in self.data.iterrows():
            ret = row['sentiment']
            if ret == 'Negative':
                self.Y.append(0)
            else:
                self.Y.append(1)


class classifier_method:
    def __init__(self, train_num, path):
        self.training_data, self.testing_data = self.split_train_test(train_num, path)
        self.dimension = len(self.training_data[0][0])
        self.min_num = 0
        print(f'dimension:{self.dimension}')

    def split_train_test(self, train_num, path):
        data = pd.read_csv(path)
        df = pd.DataFrame(data)
        vectors = []
        labels = []
        vectors_test = []
        labels_test = []
        test_num = len(df) - train_num
        print('succeed in reading csv')
        for i in range(train_num):
            vector = df['x'][i]
            label = df['y'][i]
            vector = literal_eval(vector)
            vectors.append(vector)
            labels.append(label)

        for i in range(test_num):
            vector = df['x'][i + train_num]
            label = df['y'][i + train_num]
            vector = literal_eval(vector)
            vectors_test.append(vector)
            labels_test.append(label)

        training_set = [vectors, labels]
        testing_set = [vectors_test, labels_test]
        print("preparation finished!")
        return [training_set, testing_set]

    """ calculate the test error """

    def test_error(self, clf, test_x, test_y):
        pred_y = clf.predict(test_x)
        error = 0
        for i in range(len(pred_y)):
            if pred_y[i] != test_y[i]:
                error += 1
        error /= len(pred_y)
        return error

    def net_method_train(self, epoch_num):
        """ create the network, optimizer and loss function """
        print('start networks')
        if torch.cuda.is_available():
            net = Net(self.dimension).cuda()
        else:
            net = Net(self.dimension)
        # optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.00001, betas=(0.9, 0.99))
        loss_func = nn.MSELoss(reduce=True, size_average=True)
        xs, ys = self.training_data
        xst, yst = self.testing_data
        error_list = []
        min_error = 0
        for epoch in range(epoch_num):
            error = 0
            for j in range(len(xs)):
                # cuda means the gpu boosting(changing the structure from cpu to gpu)
                """ modify the type of data """
                x, y = xs[j], ys[j]
                x = torch.from_numpy(np.array(x))
                x = x.to(torch.float32)
                y = torch.from_numpy(np.array(y))
                y = y.to(torch.float32)
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                """ training process """
                optimizer.zero_grad()
                y_re = net.forward(x)
                loss = loss_func(y_re[0], y)
                loss.backward()
                optimizer.step()
            """ calculate the test error for each epoch """
            for j in range(len(xst)):
                xt, yt = xst[j], yst[j]
                xt = torch.from_numpy(np.array(xt))
                xt = xt.to(torch.float32)
                if torch.cuda.is_available():
                    xt = xt.cuda()
                y_pred = net.predict(net.forward(xt))
                # print(str(y_re)+' '+str(y))
                if y_pred > yt or y_pred < yt:
                    error += 1
            error /= len(xst)
            error_list.append(error)
            if epoch == 0:
                min_error = error
                self.min_num = 0
            else:
                if min_error > error:
                    min_error = error
                    self.min_num = epoch
            print("epoch" + str(epoch + 1) + " error:" + str(error))
            torch.save(net.state_dict(), './params/net_params_for_epoch_' + str(epoch) + '.pkl')
        count = [i + 1 for i in range(epoch_num)]
        plt.plot(count, error_list)
        plt.xlabel('epoch num')
        plt.ylabel('error rate')
        plt.title('error rate according to epoch')
        plt.show()
        print("Finished Training")
        return net

    def svm_train(self):
        print('start svm')
        vectors, labels = self.training_data
        vectors_test, labels_test = self.testing_data
        svc = SVC(gamma='auto', kernel='linear', class_weight='balanced', tol=0.001, C=1.5)
        clf = svc.fit(vectors, labels)
        error = self.test_error(clf, vectors_test, labels_test)
        print(f'svm error:{error}')
        with open('./sklearn_classifier/svm.pickle', 'wb') as file:
            pickle.dump(clf, file)

    def randomforest_train(self, n_estimator):
        print('start random_forest')
        vectors, labels = self.training_data
        vectors_test, labels_test = self.testing_data
        tree = RandomForestClassifier(n_estimators=n_estimator)
        clf = tree.fit(vectors, labels)
        error = self.test_error(clf, vectors_test, labels_test)
        print(f'randomforest error:{error}')
        with open('./sklearn_classifier/forest.pickle', 'wb') as file:
            pickle.dump(clf, file)

    def bag_of_words(self, n_estimator, n_feature):
        data = pd.read_csv('Review.csv')
        Y = []
        paragraph_list = []
        for index, row in data.iterrows():
            Y.append(row['sentiment'])
            words = re.sub("[^a-zA-Z]", "", row['review']).lower().split()
            meaningful_words = [w for w in words]
            paragraph_list.append(" ".join(meaningful_words))
        print(len(paragraph_list))
        vectorizer = CountVectorizer(analyzer='word', tokenizer=None, preprocessor=None, stop_words=None,
                                     max_features=n_feature)

        data_features = vectorizer.fit_transform(paragraph_list)
        data_features = data_features.toarray()
        train_data_features, test_f, y_train, y_test = self.train_test_split(data_features, Y, test_size=0.2)
        print(train_data_features)
        print(train_data_features.shape)
        forest = RandomForestClassifier(n_estimators=n_estimator)
        forest = forest.fit(train_data_features, y_train)
        error = self.test_error(forest, test_f, y_test)
        print(f'bag of words:{error}')

    def adaboost_al(self, y_pred_total, y_truth):
        D = []
        alpha = []
        for t in range(len(y_pred_total)):
            """ initialize the D, then construct D_t and alpha_t """
            if t == 0:
                D_t = []
                error = 0
                for i in range(len(y_truth)):
                    D_t.append(1 / len(y_truth))
                    if y_pred_total[t][i] != y_truth[i]:
                        error += D_t[i]
                D.append(D_t)
                Z_t = sum(D_t)
                error /= Z_t
                alpha_t = 1 / 2 * math.log((1 - error) / error)
                alpha.append(alpha_t)
            else:
                D_t = []
                for i in range(len(y_truth)):
                    if y_pred_total[t][i] == y_truth[i]:
                        D_t.append(D[t - 1][i] * math.e ** (-alpha_t))
                    if y_pred_total[t][i] != y_truth[i]:
                        D_t.append(D[t - 1][i] * math.e ** alpha_t)
                        error += D_t[i]
                D.append(D_t)
                Z_t = sum(D_t)
                error /= Z_t
                alpha_t = 1 / 2 * math.log((1 - error) / error)
                alpha.append(alpha_t)
        return alpha

    def net_method_test(self, testing_data):
        if torch.cuda.is_available():
            net = Net(self.dimension).cuda()
        else:
            net = Net(self.dimension)
        net.load_state_dict(torch.load('./params/net_params_for_epoch_' + str(self.min_num) + '.pkl'))
        vectors, labels = testing_data
        labels_pred = []
        for vector in vectors:
            vector = torch.from_numpy(np.array(vector)).to(torch.float32).cuda()
            label_pred = net.predict(net.forward(vector))
            labels_pred.append(label_pred)
        return labels_pred

    def svm_test(self, testing_data):
        with open('./sklearn_classifier/svm.pickle', 'rb') as file:
            clf = pickle.load(file)
        vectors, labels = testing_data
        labels_pred = clf.predict(vectors)
        return labels_pred

    def randomforest_test(self, testing_data):
        with open('./sklearn_classifier/forest.pickle', 'rb') as file:
            clf = pickle.load(file)
        vectors, labels = testing_data
        labels_pred = clf.predict(vectors)
        return labels_pred

    def adaboost_train(self):
        print('start adaboost training')
        x_train, y_train = self.training_data
        y_pred_net = self.net_method_test(self.training_data)
        y_pred_svm = self.svm_test(self.training_data)
        y_pred_forest = self.randomforest_test(self.training_data)
        y_pred_total = [y_pred_net, y_pred_svm, y_pred_forest]
        alpha = self.adaboost_al(y_pred_total, y_train)
        return alpha
        # alpha = self.adaboost_al(y_pred_total, y_truth)

    def adaboost(self, alpha):
        x_test, y_test = self.testing_data
        y_pred_net = self.net_method_test(self.testing_data)
        y_pred_svm = self.svm_test(self.testing_data)
        y_pred_forest = self.randomforest_test(self.testing_data)
        y_pred = []
        alpha_sum = sum(alpha)
        error = 0
        print('adaboost testing')
        for i in range(len(y_test)):
            pred = (alpha[0] * y_pred_net[i] + alpha[1] * y_pred_svm[i] + alpha[2] * y_pred_forest[i]) / alpha_sum
            if pred >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)
            if y_pred[i] != y_test[i]:
                error += 1
        error /= len(y_test)
        print(f'adaboost:{error}')


if __name__ == '__main__':
    feature = DocPreprocess()
    method = classifier_method(10000, './featureData/SBERT+doc2vec.csv')
    method.svm_train()
    method.randomforest_train(200)
    method.net_method_train(50)
    method.adaboost(method.adaboost_train())
