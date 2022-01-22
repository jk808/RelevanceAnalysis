import nltk
import random

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

posts = [("Dangerous of escalating climate crisis if we don't take climate action now.", 'rel'), 
("Trees can live without humans but humans can't servive without trees So please understand the importance of trees and do plantation", 'rel'),
("Tornadoes Wyoming. What do you think about this? How it happens? Does it happen every Year? How much Tornadoes are dangerous?", 'rel'),
("Understanding environmental attitudes is vital for addressing many applied environmental problems, ranging from local issues like water pollution to global issues like climate change.", 'rel'),
("Treat yourself for the new year. All personal training sessios are 20 percent off today only", 'irr'),
("Text Club members get free 2-day shipping all day on Cyber Monday, no promo code needed. You'll have all of your holiday shopping done and in your hands by Wednesday!", 'irr')
]

words = []
for post in posts:
    words.extend(post[0].split())

if __name__ == '__main__':
    documents = [(post[0].split(), post[1])
                 for post in posts]

    random.shuffle(documents)

    # Define the feature extractor

    all_words = nltk.FreqDist(w.lower() for w in words)
    word_features = list(all_words)[:2000]

    # Train Naive Bayes classifier
    featuresets = [(document_features(d), c) for (d, c) in documents]

    train_set, test_set = featuresets[2:], featuresets[:2]
    classifier = nltk.NaiveBayesClassifier.train(train_set)

    # Test the classifier
    print(nltk.classify.accuracy(classifier, test_set))

    # Show the most important features as interpreted by Naive Bayes
    classifier.show_most_informative_features(5)