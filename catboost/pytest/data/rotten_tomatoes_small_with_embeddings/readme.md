This folder contain dataset derived from dataset Rotten Tomatoes Movie Reviews from kaggle
(link: https://www.kaggle.com/rpnuser8182/rotten-tomatoes) which provided with Open Database License (ODbL) v1.0
(link: https://opendatacommons.org/licenses/odbl/1-0/index.html).

The difference is that:

1) small subset of 101 samples is selected from the original data
2) text features have been replaced by the corresponding word embeddings computed using `spacy` library with `en_core_web_md` model.

In binclass problem we're trying to predict is author of review it top critic or not.