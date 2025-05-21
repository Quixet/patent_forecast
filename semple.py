# Standard library imports
import json
import random
from io import StringIO
# Third-party imports
import joblib
import numpy as np
import pandas as pd
import torch
from bertopic import BERTopic
from bertopic.representation import OpenAI
from bertopic.vectorizers import ClassTfidfTransformer
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
# Local application imports
from app import app
from app.learning.models import (
    Model,
    Unsupervised,
    Hierarchy,
    compute_cluster_sizes,
    number_of_desired_hierarchical_clusters
)
from config.config import (
    openai_api_key,
    openai_api_version,
    openai_azure_deployment,
    openai_azure_endpoint,
    openai_chat,
    openai_model,
    openai_prompt,
    openai_nr_docs,
    openai_doc_length,
    openai_diversity,
    openai_tokenizer,
    random_seed,
    hdbscan_core_dist_n_jobs,
    session_texts,
    count_vectorizer_ngram_range,
    transformers_model_name
)


@Model.register_subclass("hierarchical")
class Hierarchical(Unsupervised):
    def __init__(self, *args, **kwargs):
        """
        Initializer for clustering model
        :param k: number of clusters
        :param args: inherited
        :param kwargs: inherited
        """

        self.training_run = False
        self.tagging_dictionary = {}
        self.is_unsupervised = True
        super().__init__(*args, **kwargs)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        with open('app/learning/stopwords.txt', 'r') as file:
        # Read all the lines of the file into a list
        stop_words = file.readlines()
        # strip out space characters and newlines makes all words lowercases
        stop_words = [word.lower().strip() for word in stop_words]
        # New BERTopic specific initialization
        self.embedding_model = SentenceTransformer(transformers_model_name)
        self.vectorizer_model =CountVectorizer(ngram_range=count_vectorizer_ngram_range,
                stop_words=stop_words)
        self.client = AzureOpenAI(
        api_key=openai_api_key,
        api_version=openai_api_version,
        azure_deployment=openai_azure_deployment,
        azure_endpoint=openai_azure_endpoint
        )
        self.representation_model = OpenAI(
            self.client,
            model=openai_model,
            chat=openai_chat,
            prompt=openai_prompt,
            nr_docs=openai_nr_docs,
            doc_length=openai_doc_length,
            diversity=openai_diversity,
            tokenizer=openai_tokenizer,
        )
        self.ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
        self.umap = UMAP(
        n_neighbors=100,
        n_components=100,
        min_dist=0.0,
        metric='cosine',
        low_memory=False,
        random_state=random_seed)


    def save(self, filename, training_id):
        to_store = (self.key, training_id)


        joblib.dump(to_store, filename + '.meta')
        self.pipeline.save(filename, serialization="safetensors", save_ctfidf=True,
                   save_embedding_model=True)


    def load(self, filename):
        self.pipeline = BERTopic.load(filename,
                                  embedding_model=transformers_model_name)


        self.key, training_id = joblib.load(filename + '.meta')
        return self.pipeline, self.key, training_id


def get_estimator(self):
    """
    model for "Hierarchical" use case
    """


    return None


def prediction_to_tags(self, prediction, prob):
    """
    convert a prediction to a tag dictionary
    :param prediction: the cluster number this document belongs to
    :param prob: the probability associated with each class prediction
    :return: a tag description for each positive prediction (1) in the
   prediction vector
    [
    {
    document_tag_id: tag id,
    },
    ...
    ]
    """


    return [{"tag": f"{prediction} \u03B1 {prob:.2f}"}] if prediction else []


def predict(self, docs=None):
    """
    Cluster documents and return clustering; if no pipeline present, train first
    :param docs:
    :return:
    """


if self.pipeline is None:
    self.is_unsupervised = False
self.train(docs)
x = self.get_x(docs)
texts = x[session_texts].tolist()
x[session_texts] = texts
if self.training_run:
    predictions = self.pipeline.get_document_info(texts)
self.training_run = False
else:
topics, probabilities = self.pipeline.transform(texts)
# create a pandas dataframe from predictions
# map probabilities to max probability
probabilities = [max(p) for p in probabilities]
predictions = pd.DataFrame({
    'Topic': topics,
    'Probability': probabilities
})
topics = self.pipeline.get_topic_info()
# add Name column to the dataframe using topic dataframe
predictions = pd.merge(predictions, topics, left_on='Topic',
                       right_on='Topic')
predictions['id'] = x['id']
return self.format_predictions(x, predictions, 1.0)


def format_predictions(self, x, predictions, probs):
    app.logger.info('Formatting predictions...')


tags = predictions[['id']]
# tags['tag'] = predictions['Name'] + " \u03B1 " +
predictions['Probability'].map("{:.2f}".format)


# Function to generate the formatted string
def generate_string(name, prob):
    for key, value_list in self.tagging_dictionary.items():
        if name in value_list:
        return '{' + key + '} - {' + name + " \u03B1 " +


"{:.2f}".format(prob) + '}'
return '{' + name.split("_", 1)[1] + '} - {' + name + " \u03B1 " +
"{:.2f}".format(prob) + '}'

predictions['Name'] = predictions.apply(lambda row:
                                        generate_string(row['Name'], row['Probability']), axis=1)
tags['tag'] = predictions['Name']


def to_tag(tag):
    return [{"tag": tag.values[0]}]


tags = tags.groupby('id')['tag'].apply(to_tag).to_dict()
app.logger.info(tags)
app.logger.info('Done formatting predictions...')
return tags


def train_semi(self, docs=None):
    """
    After training, construct the descriptions for each cluster and return as
   training results
    :param docs: optional set of documents to cluster
    :return: a dictionary of training results containing the descriptions for
   each cluster
    """


print("Training type: Semi-Supervised")
self.training_run = True
self.load_data(docs)
feature_x_count, feature_y_count = self.df.shape
target_field = session_texts
texts = self.df[target_field].tolist()
if target_field == "claims":
# in case claims is empty, use abstract
texts = [text if text else abstract for text, abstract in zip(texts,
                                                              self.df['abstract'].tolist())]
# overwrite the text field with the cleaned text
self.df[target_field] = texts
labels = []
labels = self.pipeline.topic_labels_
labels = list(labels.values())
self.pipeline.fit(texts, y=labels)
topics = self.pipeline.get_topic_info()
topics_json = topics.drop('Representative_Docs', axis=1)  # Drop the
Representative_Docs
column
topics_json = topics_json.to_json(orient='records')
# Load JSON string as Python object
topics_data = json.loads(topics_json)
# Prepare formatted JSON data
formatted_data = {}
for item in topics_data:
    topic_id = str(item['Topic'])
item.pop('Topic')
if topic_id not in formatted_data:
    formatted_data[topic_id] = []
formatted_data[topic_id].append(item)
# Convert formatted data back to JSON string with indentation
formatted_json = json.dumps(formatted_data, indent=4)
formatted_json = json.loads(formatted_json)
# linkage_function = lambda x: sch.linkage(x, 'centroid',
optimal_ordering = True)
# hierarchical_topics = self.pipeline.hierarchical_topics(texts,
linkage_function = linkage_function)
hierarchical_topics = self.pipeline.hierarchical_topics(texts)
hierarchical_topics = hierarchical_topics.reset_index(drop=True)
csv_buffer = StringIO()
hierarchical_topics_csv = hierarchical_topics.to_csv(csv_buffer,
                                                     index=False)
csv_data = csv_buffer.getvalue()
topic_labels = self.pipeline.topic_labels_
hierarchy = Hierarchy(topic_labels, hierarchical_topics)
hierarchy_result = hierarchy.extract_hierarchy()
self.tagging_dictionary = hierarchy_result
result = {
    "feature_x_count": feature_x_count,
    "feature_y_count": feature_y_count,
    "training_results": {
"descriptions": formatted_json,
"csv_data": csv_data,
"category": hierarchy_result
}
}
return result


def train(self, docs=None):
    """
    After training, construct the descriptions for each cluster and return as
   training results
    :param docs: optional set of documents to cluster
    :return: a dictionary of training results containing the descriptions for
   each cluster
    """


print("Training type: Unsupervised")
self.is_unsupervised = False
self.training_run = True
self.load_data(docs)
feature_x_count, feature_y_count = self.df.shape
target_field = session_texts
texts = self.df[target_field].tolist()
if target_field == "claims":
# in case claims is empty, use abstract
texts = [text if text else abstract for text, abstract in zip(texts,
                                                              self.df['abstract'].tolist())]
# overwrite the text field with the cleaned text
self.df[target_field] = texts
min_cluster_size, max_cluster_size = compute_cluster_sizes(self.df.shape[0],
                                                           number_of_desired_hierarchical_clusters)
hdbscan_model = HDBSCAN(core_dist_n_jobs=hdbscan_core_dist_n_jobs,
                        min_cluster_size=min_cluster_size,
                        max_cluster_size=max_cluster_size,
                        gen_min_span_tree=True,
                        prediction_data=True,
                        cluster_selection_method='leaf',
                        min_samples=max(round(min_cluster_size / 7), 1))
self.pipeline = BERTopic(
    vectorizer_model=self.vectorizer_model,
    ctfidf_model=self.ctfidf_model,
    umap_model=self.umap,
    language='english',
    calculate_probabilities=True,
    hdbscan_model=hdbscan_model,
    representation_model=self.representation_model,
    embedding_model=self.embedding_model,
    verbose=True
)
topics, probs = self.pipeline.fit_transform(texts)
new_topics = self.pipeline.reduce_outliers(texts, topics,
                                           probabilities=probs, strategy="probabilities")
self.pipeline.update_topics(texts,
                            topics=new_topics,
                            vectorizer_model=self.vectorizer_model,
                            ctfidf_model=self.ctfidf_model,
                            representation_model=self.representation_model)
topics = self.pipeline.get_topic_info()
topics_json = topics.drop('Representative_Docs', axis=1)  # Drop the
Representative_Docs
column
topics_json = topics_json.to_json(orient='records')
# Load JSON string as Python object
topics_data = json.loads(topics_json)
# Prepare formatted JSON data
formatted_data = {}
for item in topics_data:
    topic_id = str(item['Topic'])
item.pop('Topic')
if topic_id not in formatted_data:
    formatted_data[topic_id] = []
formatted_data[topic_id].append(item)
# Convert formatted data back to JSON string with indentation
formatted_json = json.dumps(formatted_data, indent=4)
formatted_json = json.loads(formatted_json)
hierarchical_topics = self.pipeline.hierarchical_topics(texts)
hierarchical_topics = hierarchical_topics.reset_index(drop=True)
csv_buffer = StringIO()
hierarchical_topics_csv = hierarchical_topics.to_csv(csv_buffer,
                                                     index=False)
csv_data = csv_buffer.getvalue()
topic_labels = self.pipeline.topic_labels_
hierarchy = Hierarchy(topic_labels, hierarchical_topics)
hierarchy_result = hierarchy.extract_hierarchy()
self.tagging_dictionary = hierarchy_result
result = {
    "feature_x_count": feature_x_count,
    "feature_y_count": feature_y_count,
    "training_results": {
        "descriptions": formatted_json,
        "csv_data": csv_data,
        "category": hierarchy_result
    }
}
return result


def validate(self, **kwargs):
    """
    validation not supported for supervised models
    """


raise NotImplementedError
