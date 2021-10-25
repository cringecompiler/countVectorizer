from collections import defaultdict


class CountVectorizer():
    def __init__(self):
        pass

    def get_feature_names(self) -> list:
        """возвращаем список всех слов из текста"""
        list_of_names = []
        check = set()
        for row in self.corpus:
            row = row.lower().split()
            for word in row:
                word = word.rstrip('.,!?')
                if word not in check:
                    list_of_names.append(word)
                    check.add(word)
        return list_of_names

    def fit_transform(self, corpus) -> list:
        """возвращаем терм-документную матрицу"""
        matrix = []
        self.corpus = corpus
        feature_names = self.get_feature_names()
        pos_name_matrix = tuple(enumerate(feature_names))
        for row in corpus:
            counter = len(feature_names) * [0]

            check = defaultdict(int)
            for word in row.lower().split():
                check[word.rstrip('.,!?')] += 1

            for index, name in pos_name_matrix:
                counter[index] = check[name]
            matrix.append(counter)
        return matrix


if __name__ == '__main__':
    corpus_1 = [
        'Crock Pot Pasta Never boil pasta again',
        'Pasta Pomodoro Fresh ingredients Parmesan to taste',
    ]

    corpus_2 = [
        'Что что что? А ничего.',
        'Ну как так то?!',
        'А вот так, шизофрения дело такое.'
    ]

    corpus_3 = [
        'ого ого ого ого'
    ]

    vectorizer_1 = CountVectorizer()
    vectorizer_2 = CountVectorizer()
    vectorizer_3 = CountVectorizer()
    count_matrix_1 = vectorizer_1.fit_transform(corpus_1)
    count_matrix_2 = vectorizer_2.fit_transform(corpus_2)
    count_matrix_3 = vectorizer_3.fit_transform(corpus_3)
    assert count_matrix_1 == [[1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]]
    assert count_matrix_2 == [[3, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1]]
    assert count_matrix_3 == [[4]]
    print(vectorizer_1.get_feature_names())
    print(vectorizer_2.get_feature_names())
    print(vectorizer_3.get_feature_names())
