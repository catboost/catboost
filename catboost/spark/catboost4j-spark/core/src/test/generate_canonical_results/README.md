This is an utility CLI app to generate canonical data for CatBoost Spark tests using standalone CatBoost
implementation.

##Requirements

- python 3.6+
- Installed `catboost` python package built from revision [2d2b8e0dbc7e59ca9700a8f68db10a7f8b21d110](https://github.com/catboost/catboost/commit/2d2b8e0dbc7e59ca9700a8f68db10a7f8b21d110) or greater.
- CatBoost CLI app built from revision [5b19e2f4340e34382f49caaa68e5f4d67503d9b9](https://github.com/catboost/catboost/commit/5b19e2f4340e34382f49caaa68e5f4d67503d9b9) or greater for the local platform accessible from `config.CATBOOST_APP_PATH`.

##Usage

Run `main.py` from the command line.
Canonical data for the tests will be generated in `config.OUTPUT_DIR` directory.
It is `${project.basedir}/src/test/resources/canondata` by default.