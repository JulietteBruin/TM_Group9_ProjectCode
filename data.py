import pandas as pd
import utils
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier


def main():
    # Import instances and drop unused columns
    twitter_instances = pd.read_csv("dataset/gender-classifier-DFE-791531.csv", encoding='latin-1')
    twitter_instances.drop(
        columns=["_unit_id", "_golden", "_unit_state", "_trusted_judgments", "_last_judgment_at", "gender:confidence",
                 "profile_yn", "profile_yn:confidence", "created", "fav_number", "gender_gold", "link_color",
                 "profile_yn_gold", "profileimage", "retweet_count", "sidebar_color", "text"], inplace=True)

    # remove rows which don't have a gender label
    no_gender = twitter_instances[twitter_instances["gender"].isnull()].index
    unknown_gender = twitter_instances[twitter_instances["gender"] == 'unknown'].index

    twitter_instances.drop(no_gender, inplace=True)
    twitter_instances.drop(unknown_gender, inplace = True)

    # remove rows without description
    no_desciption = twitter_instances[twitter_instances["description"].isnull()].index
    twitter_instances.drop(no_desciption, inplace=True)

    # The remaining dataset has 15522 instances
    # Divide data in training and test sets (division 80/20)
    training_instances = utils.training_instances(utils.split_description(twitter_instances["description"][:12418])) #12418
    test_instances = utils.training_instances(utils.split_description(twitter_instances["description"][len(training_instances):]))

    training_labels = twitter_instances["gender"][:len(training_instances)]
    test_labels = twitter_instances["gender"][len(training_instances):]

    vec = DictVectorizer()
    training_instances_vec = vec.fit_transform(training_instances).toarray()
    test_instances_vec = vec.fit_transform(test_instances).toarray()

    lin_clf = svm.LinearSVC()
    lin_clf.fit(training_instances_vec, training_labels)

    test_predict = lin_clf.predict(test_instances_vec)
    print(classification_report(test_predict, test_labels))

if __name__ == "__main__":
    main()
